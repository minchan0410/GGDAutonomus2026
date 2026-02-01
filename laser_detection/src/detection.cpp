#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>

// Visualization Markers
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// 좌표 발행을 위한 메시지 헤더
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

// PCL Libraries
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h> 

#include <algorithm>
#include <cmath>
#include <string>
#include <limits>
#include <vector>
#include <chrono>
#include <map>

struct Box {
    double center_x;
    double center_y;
    double width;  // Diameter
    double length; // Diameter
    double heading; // Circle은 0.0
    double z_min;
    double z_max;
};

struct TrackedObject {
    int id;
    Box box;
    pcl::PointCloud<pcl::PointXYZ> points;
    uint8_t r, g, b;
    int no_detection_count; 
    
    double vx; 
    double vy; 
    ros::Time last_update_time; 
};

class LaserClusterNode {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_; 
    
    ros::Subscriber scan_sub_;
    ros::Publisher cluster_pub_; 
    ros::Publisher marker_pub_;  
    
    // 두 개의 PoseArray 퍼블리셔
    ros::Publisher pose_pub_;      // ID 전송용 (Z = ID)
    ros::Publisher pose_viz_pub_;  // Rviz 시각화용 (Z = 0)
    
    laser_geometry::LaserProjection projector_;

    // Parameters
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;

    double max_cluster_extent_threshold_;
    double min_cluster_extent_threshold_;

    bool use_fixed_size_;
    double fixed_width_;
    double fixed_length_;

    double roi_min_range_;
    double roi_max_range_;

    // Tracking Variables
    std::vector<TrackedObject> tracks_; 
    int next_id_;                       
    double tracking_distance_th_;
    double cv_max_speed_;       
    int max_disappeared_frames_;        

public:
    LaserClusterNode() : private_nh_("~"), next_id_(0) { 
        
        // 1. Clustering Params
        private_nh_.param("cluster_tolerance", cluster_tolerance_, 0.2); 
        private_nh_.param("min_cluster_size", min_cluster_size_, 10);
        private_nh_.param("max_cluster_size", max_cluster_size_, 1000); 

        // 2. Filtering Params
        private_nh_.param("max_cluster_extent", max_cluster_extent_threshold_, 1.5);
        private_nh_.param("min_cluster_extent", min_cluster_extent_threshold_, 0.2);

        // 3. Fixed Box Params (원형에서는 fixed_width를 지름으로 사용)
        private_nh_.param("use_fixed_size", use_fixed_size_, true);
        private_nh_.param("fixed_width", fixed_width_, 0.9);
        private_nh_.param("fixed_length", fixed_length_, 0.9);

        // 4. ROI Params
        private_nh_.param("roi_min_range", roi_min_range_, 0.3);
        private_nh_.param("roi_max_range", roi_max_range_, 7.0);

        // 5. Tracking Params
        private_nh_.param("tracking_distance_th", tracking_distance_th_, 1.0); 
        private_nh_.param("max_disappeared_frames", max_disappeared_frames_, 14);
        private_nh_.param("cv_max_speed", cv_max_speed_, 0.8); 

        ROS_INFO("--------------------------------");
        ROS_INFO("Mode: CIRCLE (AABB Center + Radius Correction)");
        ROS_INFO("--------------------------------");

        scan_sub_ = nh_.subscribe("/scan", 1, &LaserClusterNode::scanCallback, this);
        cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/clustered_cloud", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/detection_markers", 1);
        
        pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection_poses", 1);
        pose_viz_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection_poses_viz", 1);
        
        ROS_INFO("Laser Cluster Node Started.");
    }

    double getDistance(const Box& b1, const Box& b2) {
        return std::sqrt(std::pow(b1.center_x - b2.center_x, 2) + std::pow(b1.center_y - b2.center_y, 2));
    }

    void updateTracking(const std::vector<Box>& new_boxes, const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& new_cluster_points, ros::Time current_time) {
        
        std::vector<bool> matched_new_box(new_boxes.size(), false);
        
        for (auto& track : tracks_) {
            
            double min_dist = std::numeric_limits<double>::max();
            int best_match_idx = -1;

            for (size_t i = 0; i < new_boxes.size(); ++i) {
                if (matched_new_box[i]) continue; 

                double dist = getDistance(track.box, new_boxes[i]);

                if (dist < min_dist) {
                    min_dist = dist;
                    best_match_idx = i;
                }
            }

            double dt = (current_time - track.last_update_time).toSec();
            if (dt <= 0.0) dt = 0.1; 

            if (best_match_idx != -1 && min_dist < tracking_distance_th_) {
                // --- 매칭 성공 ---
                double alpha = 0.6; 
                double current_vx = (new_boxes[best_match_idx].center_x - track.box.center_x) / dt;
                double current_vy = (new_boxes[best_match_idx].center_y - track.box.center_y) / dt;
                
                if (std::abs(current_vx) < cv_max_speed_ && std::abs(current_vy) < cv_max_speed_) {
                    track.vx = (1.0 - alpha) * track.vx + alpha * current_vx;
                    track.vy = (1.0 - alpha) * track.vy + alpha * current_vy;
                }

                track.box = new_boxes[best_match_idx];
                track.points = *new_cluster_points[best_match_idx];
                track.no_detection_count = 0; 
                track.last_update_time = current_time;
                
                matched_new_box[best_match_idx] = true;

            } else {
                // --- 매칭 실패 (Prediction) ---
                track.no_detection_count++;
                
                track.box.center_x += track.vx * dt;
                track.box.center_y += track.vy * dt;
                
                track.last_update_time = current_time;
            }
        }

        // 신규 트랙 생성
        for (size_t i = 0; i < new_boxes.size(); ++i) {
            if (!matched_new_box[i]) {
                TrackedObject new_track;
                new_track.id = next_id_++;
                new_track.box = new_boxes[i];
                new_track.points = *new_cluster_points[i];
                new_track.no_detection_count = 0;
                
                new_track.vx = 0.0;
                new_track.vy = 0.0;
                new_track.last_update_time = current_time;

                new_track.r = (new_track.id * 50 + 20) % 255;
                new_track.g = (new_track.id * 100 + 50) % 255;
                new_track.b = (new_track.id * 150 + 100) % 255;
                
                if(new_track.r < 80 && new_track.g < 80 && new_track.b < 80) {
                    new_track.r += 100;
                }

                tracks_.push_back(new_track);
            }
        }

        // 삭제
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const TrackedObject& t) {
                return t.no_detection_count > max_disappeared_frames_;
            }), tracks_.end());
    }

    // [수정] 원형 피팅: Min-Max (AABB) Center + Radius Correction
    Box fittingCircle(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
        Box circle_box;

        // 1. Min/Max 찾기 (AABB)
        // 무게 중심 대신 '범위의 중간'을 사용하여 점 밀도 편향 제거
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        // AABB의 중심 (일단 보이는 것의 중심)
        double box_cx = (min_pt.x + max_pt.x) / 2.0;
        double box_cy = (min_pt.y + max_pt.y) / 2.0;

        circle_box.z_min = min_pt.z;
        circle_box.z_max = max_pt.z;
        circle_box.heading = 0.0;

        double radius = 0.0;

        if (use_fixed_size_) {
            // [중요] 반지름 보정 로직
            // 라이다는 물체의 앞면만 보므로, 보이는 중심(box_cx)은 실제 중심보다 센서 쪽에 가깝습니다.
            // 따라서 '관측된 크기'와 '실제 크기(fixed_width)'의 차이만큼 뒤로 밀어줍니다.

            radius = fixed_width_ / 2.0;

            // 관측된 지름 (X축, Y축 중 큰 것 사용)
            double obs_diam_x = max_pt.x - min_pt.x;
            double obs_diam_y = max_pt.y - min_pt.y;
            double obs_diameter = std::max(obs_diam_x, obs_diam_y); 

            // 보정해야 할 거리 (Offset)
            double diff = (fixed_width_ - obs_diameter) / 2.0;
            
            // 관측된 게 실제보다 작을 때만 보정 수행
            if (diff > 0) {
                // 센서 원점(0,0)에서 AABB 중심까지의 거리
                double dist_to_sensor = std::sqrt(box_cx * box_cx + box_cy * box_cy);
                
                if (dist_to_sensor > 0.001) {
                    // 단위 벡터 (Direction Vector)
                    double ux = box_cx / dist_to_sensor;
                    double uy = box_cy / dist_to_sensor;

                    // 중심을 센서 반대 방향(바깥쪽)으로 밀어줌
                    circle_box.center_x = box_cx + ux * diff;
                    circle_box.center_y = box_cy + uy * diff;
                } else {
                    circle_box.center_x = box_cx;
                    circle_box.center_y = box_cy;
                }
            } else {
                circle_box.center_x = box_cx;
                circle_box.center_y = box_cy;
            }

        } else {
            // 가변 크기: 그냥 AABB 중심 사용 (또는 Enclosing Circle)
            circle_box.center_x = box_cx;
            circle_box.center_y = box_cy;
            
            // 중심에서 가장 먼 점까지를 반지름으로
            double max_dist_sq = 0.0;
            for (const auto& p : cluster->points) {
                double dx = p.x - circle_box.center_x;
                double dy = p.y - circle_box.center_y;
                double dist_sq = dx*dx + dy*dy;
                if (dist_sq > max_dist_sq) max_dist_sq = dist_sq;
            }
            radius = std::sqrt(max_dist_sq);
        }

        circle_box.width = radius * 2.0; 
        circle_box.length = radius * 2.0; 

        return circle_box;
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_in) {

        sensor_msgs::LaserScan scan_filtered = *scan_in;

        size_t range_size = scan_filtered.ranges.size();
        for (size_t i = 0; i < range_size; ++i) {
            double r = scan_filtered.ranges[i];
            if (!std::isfinite(r)) continue;
            if (r < roi_min_range_ || r > roi_max_range_) {
                scan_filtered.ranges[i] = std::numeric_limits<float>::infinity();
            }
        }

        sensor_msgs::PointCloud2 cloud_msg;
        projector_.projectLaser(scan_filtered, cloud_msg); 

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_current(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud_msg, *cloud_current);

        if (cloud_current->empty()) return;

        // Clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_current);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        
        ec.setClusterTolerance(cluster_tolerance_); 
        ec.setMinClusterSize(min_cluster_size_);    
        ec.setMaxClusterSize(max_cluster_size_);   
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_current); 
        ec.extract(cluster_indices);

        // --- Prepare Data for Tracking ---
        std::vector<Box> current_boxes;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> current_cluster_points;

        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (int idx : indices.indices) {
                current_cluster->points.push_back(cloud_current->points[idx]);
            }
            current_cluster->width = current_cluster->points.size();
            current_cluster->height = 1;
            current_cluster->is_dense = true;

            // Extent Filter
            double max_dist_sq = 0.0;
            size_t num_points = current_cluster->points.size();
            for (size_t j = 0; j < num_points; ++j) {
                for (size_t k = j + 1; k < num_points; ++k) {
                    const auto& p1 = current_cluster->points[j];
                    const auto& p2 = current_cluster->points[k];
                    double d_sq = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z);
                    if (d_sq > max_dist_sq) max_dist_sq = d_sq;
                }
            }
            double cluster_max_extent = std::sqrt(max_dist_sq);

            if (cluster_max_extent > max_cluster_extent_threshold_ || 
                cluster_max_extent < min_cluster_extent_threshold_) {
                continue; 
            }
            
            // [수정] 원형 피팅 함수 호출
            Box box = fittingCircle(current_cluster);
            current_boxes.push_back(box);
            current_cluster_points.push_back(current_cluster);
        }

        // --- UPDATE TRACKING ---
        updateTracking(current_boxes, current_cluster_points, scan_in->header.stamp);

        // --- VISUALIZATION & PUBLISHING ---
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        geometry_msgs::PoseArray pose_array_logic;
        pose_array_logic.header = scan_in->header; 

        geometry_msgs::PoseArray pose_array_viz;
        pose_array_viz.header = scan_in->header;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (size_t i = 0; i < tracks_.size(); ++i) {
            
            const TrackedObject& track = tracks_[i];
            bool is_missing = (track.no_detection_count > 0);
            float alpha_val = is_missing ? 0.4 : 1.0; 

            // --- 1. Logic Pose (Z에 ID 탑재) ---
            geometry_msgs::Pose pose_logic;
            pose_logic.position.x = track.box.center_x;
            pose_logic.position.y = track.box.center_y;
            pose_logic.position.z = static_cast<double>(track.id); 
            pose_logic.orientation.w = 1.0; // 원은 회전 없음
            pose_array_logic.poses.push_back(pose_logic);

            // --- 2. Viz Pose (Z = 0.0) ---
            geometry_msgs::Pose pose_viz;
            pose_viz.position.x = track.box.center_x;
            pose_viz.position.y = track.box.center_y;
            pose_viz.position.z = 0.0; 
            pose_viz.orientation.w = 1.0; 
            pose_array_viz.poses.push_back(pose_viz);

            // 3. Marker - Circle (Line Strip)
            visualization_msgs::Marker circle_marker;
            circle_marker.header = scan_in->header;
            circle_marker.ns = "circles";
            circle_marker.id = track.id;
            circle_marker.type = visualization_msgs::Marker::LINE_STRIP;
            circle_marker.action = visualization_msgs::Marker::ADD;
            circle_marker.scale.x = 0.05; 
            
            circle_marker.color.r = track.r / 255.0; 
            circle_marker.color.g = track.g / 255.0; 
            circle_marker.color.b = track.b / 255.0; 
            circle_marker.color.a = alpha_val;
            circle_marker.lifetime = ros::Duration(0.1);
            
            const int circle_points = 36; 
            double radius = track.box.width / 2.0;
            
            for(int k=0; k <= circle_points; ++k) {
                double angle = k * (2.0 * M_PI / circle_points);
                geometry_msgs::Point p;
                p.x = track.box.center_x + radius * std::cos(angle);
                p.y = track.box.center_y + radius * std::sin(angle);
                p.z = track.box.z_min; 
                circle_marker.points.push_back(p);
            }
            marker_array.markers.push_back(circle_marker);

            // 4. Text Marker
            visualization_msgs::Marker text_marker;
            text_marker.header = scan_in->header;
            text_marker.ns = "ids";
            text_marker.id = track.id;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            std::string id_str = "ID: " + std::to_string(track.id);
            if(is_missing) id_str += " (Pred)";
            
            text_marker.text = id_str;
            text_marker.pose.position.x = track.box.center_x - 0.4;
            text_marker.pose.position.y = track.box.center_y;
            text_marker.pose.position.z = track.box.z_max + 0.5; 
            text_marker.scale.z = 0.3; 
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = alpha_val;
            text_marker.lifetime = ros::Duration(0.1);
            marker_array.markers.push_back(text_marker);

            // 5. PointCloud Coloring
            if (!is_missing) {
                for (const auto& pt : track.points.points) {
                    pcl::PointXYZRGB pt_rgb;
                    pt_rgb.x = pt.x; pt_rgb.y = pt.y; pt_rgb.z = pt.z;
                    pt_rgb.r = track.r; pt_rgb.g = track.g; pt_rgb.b = track.b;
                    cloud_colored->points.push_back(pt_rgb);
                }
            }
        }

        marker_pub_.publish(marker_array);
        
        // 두 토픽 발행
        pose_pub_.publish(pose_array_logic);       // /detection_poses (Z=ID)
        pose_viz_pub_.publish(pose_array_viz);     // /detection_poses_viz (Z=0)

        // Publish Colored Cloud
        cloud_colored->header = pcl_conversions::toPCL(scan_in->header);
        cloud_colored->width = cloud_colored->points.size();
        cloud_colored->height = 1;
        cloud_colored->is_dense = true;
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*cloud_colored, output_msg);
        cluster_pub_.publish(output_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "laser_cluster_circle_node");
    LaserClusterNode node;
    ros::spin();
    return 0;
}
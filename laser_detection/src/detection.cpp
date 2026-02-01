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
    double width;  
    double length; 
    double heading; 
    double z_min;
    double z_max;
};

struct TrackedObject {
    int id;
    Box box;
    pcl::PointCloud<pcl::PointXYZ> points; // 할당된 포인트들
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
    
    ros::Publisher pose_pub_;      // Z = ID
    ros::Publisher pose_viz_pub_;  // Z = 0
    
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
    double track_search_radius_; // [신규] 트랙킹 유지를 위해 점을 찾는 반경
    double cv_max_speed_;       
    int max_disappeared_frames_;        

public:
    LaserClusterNode() : private_nh_("~"), next_id_(0) { 
        
        // 1. Clustering Params (신규 탐지용)
        private_nh_.param("cluster_tolerance", cluster_tolerance_, 0.2); 
        private_nh_.param("min_cluster_size", min_cluster_size_, 10);
        private_nh_.param("max_cluster_size", max_cluster_size_, 1000); 

        // 2. Filtering Params
        private_nh_.param("max_cluster_extent", max_cluster_extent_threshold_, 1.5);
        private_nh_.param("min_cluster_extent", min_cluster_extent_threshold_, 0.2);

        // 3. Fixed Box Params
        private_nh_.param("use_fixed_size", use_fixed_size_, true);
        private_nh_.param("fixed_width", fixed_width_, 0.9);
        private_nh_.param("fixed_length", fixed_length_, 0.9);

        // 4. ROI Params
        private_nh_.param("roi_min_range", roi_min_range_, 0.3);
        private_nh_.param("roi_max_range", roi_max_range_, 7.0);

        // 5. Tracking Params
        // track_search_radius: 예측된 위치 주변 이 반경 내의 점들은 해당 트랙으로 흡수함
        private_nh_.param("track_search_radius", track_search_radius_, 0.8); 
        private_nh_.param("max_disappeared_frames", max_disappeared_frames_, 14);
        private_nh_.param("cv_max_speed", cv_max_speed_, 5.0); 

        ROS_INFO("--------------------------------");
        ROS_INFO("Mode: HYBRID TRACKING (Points Association -> Clustering)");
        ROS_INFO("Tracking: Radius=%.2f, MaxLost=%d", track_search_radius_, max_disappeared_frames_);
        ROS_INFO("--------------------------------");

        scan_sub_ = nh_.subscribe("/scan", 1, &LaserClusterNode::scanCallback, this);
        cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/clustered_cloud", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/detection_markers", 1);
        
        pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection_poses", 1);
        pose_viz_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection_poses_viz", 1);
        
        ROS_INFO("Laser Cluster Node Started.");
    }

    double getDistanceSq(const pcl::PointXYZ& p, double cx, double cy) {
        return (p.x - cx)*(p.x - cx) + (p.y - cy)*(p.y - cy);
    }

    // Min-Max Center + Radius Correction
    Box fittingCircle(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
        Box circle_box;
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        double box_cx = (min_pt.x + max_pt.x) / 2.0;
        double box_cy = (min_pt.y + max_pt.y) / 2.0;

        circle_box.z_min = min_pt.z;
        circle_box.z_max = max_pt.z;
        circle_box.heading = 0.0;

        if (use_fixed_size_) {
            circle_box.width = fixed_width_; // Use fixed diameter
            circle_box.length = fixed_width_;

            double obs_diam_x = max_pt.x - min_pt.x;
            double obs_diam_y = max_pt.y - min_pt.y;
            double obs_diameter = std::max(obs_diam_x, obs_diam_y); 
            double diff = (fixed_width_ - obs_diameter) / 2.0;
            
            if (diff > 0) {
                double dist_to_sensor = std::sqrt(box_cx * box_cx + box_cy * box_cy);
                if (dist_to_sensor > 0.001) {
                    double ux = box_cx / dist_to_sensor;
                    double uy = box_cy / dist_to_sensor;
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
            // Variable size
            circle_box.center_x = box_cx;
            circle_box.center_y = box_cy;
            double max_dist_sq = 0.0;
            for (const auto& p : cluster->points) {
                double dx = p.x - box_cx;
                double dy = p.y - box_cy;
                double dist_sq = dx*dx + dy*dy;
                if (dist_sq > max_dist_sq) max_dist_sq = dist_sq;
            }
            double radius = std::sqrt(max_dist_sq);
            circle_box.width = radius * 2.0; 
            circle_box.length = radius * 2.0; 
        }
        return circle_box;
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_in) {
        
        // 1. Convert LaserScan to PointCloud (with Range Filtering)
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

        ros::Time current_time = scan_in->header.stamp;

        // =================================================================================
        // STEP 1: PREDICTION (모든 트랙을 먼저 이동시킴)
        // =================================================================================
        for (auto& track : tracks_) {
            double dt = (current_time - track.last_update_time).toSec();
            if (dt < 0.0) dt = 0.0;
            
            // 예측 위치 업데이트 (CV Model)
            track.box.center_x += track.vx * dt;
            track.box.center_y += track.vy * dt;
            
            // 포인트 초기화 (새로운 점들을 담을 준비)
            track.points.clear();
        }

        // =================================================================================
        // STEP 2: POINTS ASSOCIATION (점들을 트랙에 할당)
        // =================================================================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_unassigned(new pcl::PointCloud<pcl::PointXYZ>);
        double search_r_sq = track_search_radius_ * track_search_radius_;

        for (const auto& pt : cloud_current->points) {
            int best_track_idx = -1;
            double min_dist_sq = std::numeric_limits<double>::max();

            // 이 점이 어떤 트랙의 예측 위치 근처에 있는지 검사
            for (size_t i = 0; i < tracks_.size(); ++i) {
                double dist_sq = getDistanceSq(pt, tracks_[i].box.center_x, tracks_[i].box.center_y);
                
                if (dist_sq < search_r_sq) {
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        best_track_idx = i;
                    }
                }
            }

            if (best_track_idx != -1) {
                // 가장 가까운 트랙에 점 추가
                tracks_[best_track_idx].points.push_back(pt);
            } else {
                // 어떤 트랙에도 속하지 않는 점 -> 신규 탐지 후보
                cloud_unassigned->points.push_back(pt);
            }
        }

        // =================================================================================
        // STEP 3: UPDATE TRACKS (할당된 점들로 피팅 다시 수행)
        // =================================================================================
        for (auto& track : tracks_) {
            // 점이 하나라도 할당되었으면 -> Measurement Update
            if (!track.points.empty()) {
                
                // 점들의 Z값 범위 갱신
                double min_z = std::numeric_limits<double>::max();
                double max_z = -std::numeric_limits<double>::max();
                for(const auto& p : track.points) {
                    if(p.z < min_z) min_z = p.z;
                    if(p.z > max_z) max_z = p.z;
                }
                track.box.z_min = min_z;
                track.box.z_max = max_z;

                // [중요] 점들로 다시 피팅 수행
                // shared_ptr 변환 필요
                pcl::PointCloud<pcl::PointXYZ>::Ptr track_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>(track.points));
                Box new_box = fittingCircle(track_cloud_ptr);

                // 속도 업데이트 (Measurement - Predicted) / dt
                // 주의: 위에서 이미 Prediction으로 box를 이동시켰으므로,
                // 현재 new_box는 실제 관측치, track.box는 예측치에 가까움
                double dt = (current_time - track.last_update_time).toSec();
                if (dt > 0.001) {
                    double alpha = 0.6;
                    double measured_vx = (new_box.center_x - (track.box.center_x - track.vx*dt)) / dt; 
                    double measured_vy = (new_box.center_y - (track.box.center_y - track.vy*dt)) / dt;
                    
                    // 단순하게는 (new - old_measured) / dt 를 써도 됨. 여기선 위치 갱신 차이 이용
                    double raw_vx = (new_box.center_x - (track.box.center_x - track.vx*dt)) / dt; 
                    // 간단히 이전 위치 저장 변수가 없으므로, 현재 위치 차이로 속도 보정
                    
                    // 위치는 관측값으로 강제 보정
                    track.box = new_box;
                    
                    // 속도 필터링
                    if (std::abs(raw_vx) < cv_max_speed_) {
                         track.vx = (1.0 - alpha) * track.vx + alpha * raw_vx;
                         track.vy = (1.0 - alpha) * track.vy + alpha * (new_box.center_y - (track.box.center_y - track.vy*dt)) / dt;
                    }
                } else {
                    track.box = new_box;
                }

                track.no_detection_count = 0;
                track.last_update_time = current_time;

            } else {
                // 점이 하나도 할당 안됨 -> Prediction 유지 (이미 Step 1에서 이동함)
                track.no_detection_count++;
                track.last_update_time = current_time; // 시간은 흐름
            }
        }

        // =================================================================================
        // STEP 4: DETECT NEW OBJECTS (남은 점들로 클러스터링)
        // =================================================================================
        if (!cloud_unassigned->empty()) {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(cloud_unassigned);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(cluster_tolerance_);
            ec.setMinClusterSize(min_cluster_size_);
            ec.setMaxClusterSize(max_cluster_size_);
            ec.setSearchMethod(tree);
            ec.setInputCloud(cloud_unassigned);
            ec.extract(cluster_indices);

            for (const auto& indices : cluster_indices) {
                // Extent Check
                pcl::PointCloud<pcl::PointXYZ>::Ptr new_cluster(new pcl::PointCloud<pcl::PointXYZ>);
                for (int idx : indices.indices) {
                    new_cluster->points.push_back(cloud_unassigned->points[idx]);
                }
                
                pcl::PointXYZ min_pt, max_pt;
                pcl::getMinMax3D(*new_cluster, min_pt, max_pt);
                double dist_sq = (max_pt.x - min_pt.x)*(max_pt.x - min_pt.x) + 
                                 (max_pt.y - min_pt.y)*(max_pt.y - min_pt.y) + 
                                 (max_pt.z - min_pt.z)*(max_pt.z - min_pt.z);
                double extent = std::sqrt(dist_sq);

                if (extent > max_cluster_extent_threshold_ || extent < min_cluster_extent_threshold_) {
                    continue; 
                }

                // 신규 트랙 생성
                Box new_box = fittingCircle(new_cluster);
                
                TrackedObject new_track;
                new_track.id = next_id_++;
                new_track.box = new_box;
                new_track.points = *new_cluster;
                new_track.no_detection_count = 0;
                new_track.vx = 0.0;
                new_track.vy = 0.0;
                new_track.last_update_time = current_time;
                
                new_track.r = (new_track.id * 50 + 20) % 255;
                new_track.g = (new_track.id * 100 + 50) % 255;
                new_track.b = (new_track.id * 150 + 100) % 255;
                if(new_track.r < 80 && new_track.g < 80 && new_track.b < 80) new_track.r += 100;

                tracks_.push_back(new_track);
            }
        }

        // =================================================================================
        // STEP 5: REMOVE DEAD TRACKS
        // =================================================================================
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const TrackedObject& t) {
                return t.no_detection_count > max_disappeared_frames_;
            }), tracks_.end());


        // =================================================================================
        // STEP 6: VISUALIZATION & PUBLISH
        // =================================================================================
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        geometry_msgs::PoseArray pose_array_logic;
        pose_array_logic.header = scan_in->header; 
        geometry_msgs::PoseArray pose_array_viz;
        pose_array_viz.header = scan_in->header;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (const auto& track : tracks_) {
            bool is_missing = (track.no_detection_count > 0);
            float alpha_val = is_missing ? 0.4 : 1.0; 

            // 1. Pose
            geometry_msgs::Pose pose_logic;
            pose_logic.position.x = track.box.center_x;
            pose_logic.position.y = track.box.center_y;
            pose_logic.position.z = static_cast<double>(track.id); 
            pose_logic.orientation.w = 1.0;
            pose_array_logic.poses.push_back(pose_logic);

            geometry_msgs::Pose pose_viz;
            pose_viz.position = pose_logic.position;
            pose_viz.position.z = 0.0;
            pose_viz.orientation.w = 1.0;
            pose_array_viz.poses.push_back(pose_viz);

            // 2. Marker (Circle)
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

            // 3. Text
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

            // 4. Colored Cloud (Missing이 아닌 경우에만)
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
        pose_pub_.publish(pose_array_logic);
        pose_viz_pub_.publish(pose_array_viz);

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
    ros::init(argc, argv, "laser_cluster_hybrid_node");
    LaserClusterNode node;
    ros::spin();
    return 0;
}
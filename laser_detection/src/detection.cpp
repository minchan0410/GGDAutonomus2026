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
#include <deque>
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

// [수정] 속도와 시간 정보를 포함한 트래킹 객체
struct TrackedObject {
    int id;
    Box box;
    pcl::PointCloud<pcl::PointXYZ> points;
    uint8_t r, g, b;
    int no_detection_count; 
    
    // [추가] 등속 운동 예측을 위한 변수
    double vx; // x축 속도 (m/s)
    double vy; // y축 속도 (m/s)
    ros::Time last_update_time; // 마지막으로 위치가 갱신된 시간
};

class LaserClusterNode {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_; 
    
    ros::Subscriber scan_sub_;
    ros::Publisher cluster_pub_; 
    ros::Publisher marker_pub_;  
    ros::Publisher pose_pub_;
    ros::Publisher accumulated_cloud_pub_; 
    
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

    // Accumulation Variables
    int accumulate_frames_;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_queue_;

    // Tracking Variables
    std::vector<TrackedObject> tracks_; 
    int next_id_;                       
    double tracking_distance_th_;       
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

        // 3. Fixed Box Params
        private_nh_.param("use_fixed_size", use_fixed_size_, true);
        private_nh_.param("fixed_width", fixed_width_, 0.9);
        private_nh_.param("fixed_length", fixed_length_, 0.9);

        // 4. ROI Params
        private_nh_.param("roi_min_range", roi_min_range_, 0.3);
        private_nh_.param("roi_max_range", roi_max_range_, 7.0);

        // 5. Accumulation Params
        private_nh_.param("accumulate_frames", accumulate_frames_, 3); 

        // 6. Tracking Params
        private_nh_.param("tracking_distance_th", tracking_distance_th_, 1.0); 
        private_nh_.param("max_disappeared_frames", max_disappeared_frames_, 5); 

        ROS_INFO("--------------------------------");
        ROS_INFO("Mode: CIRCULAR FITTING + CV PREDICTION");
        ROS_INFO("Cluster Params: Tol=%.2f, Min=%d, Max=%d", cluster_tolerance_, min_cluster_size_, max_cluster_size_);
        ROS_INFO("Circle Params: Fixed=%s, Diameter=%.2f", use_fixed_size_ ? "True" : "False", fixed_width_);
        ROS_INFO("--------------------------------");

        scan_sub_ = nh_.subscribe("/scan", 1, &LaserClusterNode::scanCallback, this);
        cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/clustered_cloud", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/detection_markers", 1);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection_poses", 1);
        accumulated_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/accumulated_cloud", 1);
        
        ROS_INFO("Laser Cluster Node Started.");
    }

    double getDistance(const Box& b1, const Box& b2) {
        return std::sqrt(std::pow(b1.center_x - b2.center_x, 2) + std::pow(b1.center_y - b2.center_y, 2));
    }

    // [수정] 현재 시간(scan_time)을 인자로 받음
    void updateTracking(const std::vector<Box>& new_boxes, const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& new_cluster_points, ros::Time current_time) {
        
        std::vector<bool> matched_new_box(new_boxes.size(), false);
        
        // 1. 기존 트랙 매칭
        for (auto& track : tracks_) {
            
            // 예측(Prediction) 위치를 기준으로 매칭을 시도하는 것이 더 정확함
            // 여기서는 간단히 기존 위치 기준으로 매칭하되, 뒤에서 Update함.

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

            // 시간 차이 계산 (dt)
            double dt = (current_time - track.last_update_time).toSec();
            if (dt <= 0.0) dt = 0.1; // 0으로 나누기 방지용 안전장치

            if (best_match_idx != -1 && min_dist < tracking_distance_th_) {
                // --- 매칭 성공 (Observation Update) ---
                
                // 1. 속도 업데이트 (Low Pass Filter 적용하여 노이즈 감소)
                double alpha = 0.6; // 1.0이면 현재 속도 100% 반영, 낮을수록 부드럽게 변함
                double current_vx = (new_boxes[best_match_idx].center_x - track.box.center_x) / dt;
                double current_vy = (new_boxes[best_match_idx].center_y - track.box.center_y) / dt;
                
                // 튀는 값 방지 (ex: 매칭 오류로 인해 순간이동 하는 경우)
                if (std::abs(current_vx) < 10.0 && std::abs(current_vy) < 10.0) {
                    track.vx = (1.0 - alpha) * track.vx + alpha * current_vx;
                    track.vy = (1.0 - alpha) * track.vy + alpha * current_vy;
                }

                // 2. 위치 및 정보 업데이트
                track.box = new_boxes[best_match_idx];
                track.points = *new_cluster_points[best_match_idx];
                track.no_detection_count = 0; 
                track.last_update_time = current_time;
                
                matched_new_box[best_match_idx] = true;

            } else {
                // --- 매칭 실패 (Prediction Step only) ---
                track.no_detection_count++;
                
                // [핵심] 등속 운동 모델 적용 (위치 예측)
                // 사라진 동안에도 기존 속도(vx, vy)만큼 이동시킴
                track.box.center_x += track.vx * dt;
                track.box.center_y += track.vy * dt;
                
                // 시간도 갱신해줘야 다음 루프에서 dt가 올바르게 계산됨
                track.last_update_time = current_time;
            }
        }

        // 2. 신규 트랙 생성
        for (size_t i = 0; i < new_boxes.size(); ++i) {
            if (!matched_new_box[i]) {
                TrackedObject new_track;
                new_track.id = next_id_++;
                new_track.box = new_boxes[i];
                new_track.points = *new_cluster_points[i];
                new_track.no_detection_count = 0;
                
                // 초기 속도 0, 초기 시간 설정
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

        // 3. 삭제
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const TrackedObject& t) {
                return t.no_detection_count > max_disappeared_frames_;
            }), tracks_.end());
    }

    Box fittingCircle(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
        Box circle_box;

        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        double box_cx = (min_pt.x + max_pt.x) / 2.0;
        double box_cy = (min_pt.y + max_pt.y) / 2.0;

        circle_box.z_min = min_pt.z;
        circle_box.z_max = max_pt.z;
        circle_box.heading = 0.0;

        double radius = 0.0;

        if (use_fixed_size_) {
            radius = fixed_width_ / 2.0;
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
            circle_box.center_x = box_cx;
            circle_box.center_y = box_cy;
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

        // Queue Update
        cloud_queue_.push_back(cloud_current);
        if (cloud_queue_.size() > accumulate_frames_) {
            cloud_queue_.pop_front();
        }

        // Accumulate
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_accumulated(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& cloud : cloud_queue_) {
            *cloud_accumulated += *cloud; 
        }

        if (accumulated_cloud_pub_.getNumSubscribers() > 0) {
            sensor_msgs::PointCloud2 accum_msg;
            pcl::toROSMsg(*cloud_accumulated, accum_msg);
            accum_msg.header = scan_in->header; 
            accumulated_cloud_pub_.publish(accum_msg);
        }

        if (cloud_accumulated->empty()) return;

        // Clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_accumulated);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        
        ec.setClusterTolerance(cluster_tolerance_); 
        ec.setMinClusterSize(min_cluster_size_);    
        ec.setMaxClusterSize(max_cluster_size_);   
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_accumulated); 
        ec.extract(cluster_indices);

        // --- Prepare Data for Tracking ---
        std::vector<Box> current_boxes;
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> current_cluster_points;

        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (int idx : indices.indices) {
                current_cluster->points.push_back(cloud_accumulated->points[idx]);
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
            
            Box box = fittingCircle(current_cluster);
            current_boxes.push_back(box);
            current_cluster_points.push_back(current_cluster);
        }

        // --- UPDATE TRACKING (Pass Current Time) ---
        // scan_in->header.stamp를 사용하여 정확한 측정 시간을 전달합니다.
        updateTracking(current_boxes, current_cluster_points, scan_in->header.stamp);

        // --- VISUALIZATION ---
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        geometry_msgs::PoseArray pose_array_msg;
        pose_array_msg.header = scan_in->header; 

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (size_t i = 0; i < tracks_.size(); ++i) {
            
            const TrackedObject& track = tracks_[i];
            bool is_missing = (track.no_detection_count > 0);
            
            // 놓친 물체는 투명도 0.4로 흐릿하게, 잡힌 건 1.0
            float alpha_val = is_missing ? 0.4 : 1.0; 

            // 1. Pose (예측된 위치가 반영됨)
            geometry_msgs::Pose pose;
            pose.position.x = track.box.center_x;
            pose.position.y = track.box.center_y;
            pose.position.z = (track.box.z_min + track.box.z_max) / 2.0;
            pose.orientation.w = 1.0; 
            pose_array_msg.poses.push_back(pose);

            // 2. Marker - Circle
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

            // 3. Text Marker (ID)
            visualization_msgs::Marker text_marker;
            text_marker.header = scan_in->header;
            text_marker.ns = "ids";
            text_marker.id = track.id;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            std::string id_str = "ID: " + std::to_string(track.id);
            // 사라진 상태면 (Pred) 라고 표시해줌
            if(is_missing) id_str += " (Pred)";
            
            text_marker.text = id_str;
            text_marker.pose.position.x = track.box.center_x - 0.3;
            text_marker.pose.position.y = track.box.center_y;
            text_marker.pose.position.z = track.box.z_max + 0.5; 
            text_marker.scale.z = 0.3; 
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = alpha_val;
            text_marker.lifetime = ros::Duration(0.1);
            marker_array.markers.push_back(text_marker);

            // 4. PointCloud Coloring (놓치지 않았을 때만 그림)
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
        pose_pub_.publish(pose_array_msg);

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
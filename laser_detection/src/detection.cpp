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
    double width;
    double length;
    double heading; // radian
    double z_min;
    double z_max;
};

// [추가] 트래킹 되는 객체 정보를 담는 구조체
struct TrackedObject {
    int id;
    Box box;
    pcl::PointCloud<pcl::PointXYZ> points; // 시각화를 위해 포인트 저장
    
    // 고유 색상
    uint8_t r, g, b;

    // 사라짐 감지용 (몇 프레임 동안 놓쳤는지)
    int no_detection_count; 
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

    // [추가] Tracking Variables
    std::vector<TrackedObject> tracks_; // 현재 추적 중인 객체 리스트
    int next_id_;                       // 다음 부여할 ID
    double tracking_distance_th_;       // 같은 물체로 판단할 거리 기준 (m)
    int max_disappeared_frames_;        // 몇 프레임 안 보이면 삭제할지

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
        private_nh_.param("fixed_length", fixed_length_, 0.45);

        // 4. ROI Params
        private_nh_.param("roi_min_range", roi_min_range_, 0.3);
        private_nh_.param("roi_max_range", roi_max_range_, 7.0);

        // 5. Accumulation Params
        private_nh_.param("accumulate_frames", accumulate_frames_, 3); 

        // 6. [추가] Tracking Params
        private_nh_.param("tracking_distance_th", tracking_distance_th_, 1.0); // 1m 이내면 같은 물체로 간주
        private_nh_.param("max_disappeared_frames", max_disappeared_frames_, 5); // 5프레임 놓치면 삭제

        ROS_INFO("--------------------------------");
        ROS_INFO("Cluster Params: Tol=%.2f, Min=%d, Max=%d", cluster_tolerance_, min_cluster_size_, max_cluster_size_);
        ROS_INFO("Tracking Params: DistTh=%.2f, MaxLost=%d", tracking_distance_th_, max_disappeared_frames_);
        ROS_INFO("--------------------------------");

        scan_sub_ = nh_.subscribe("/scan", 1, &LaserClusterNode::scanCallback, this);
        cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/clustered_cloud", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/detection_markers", 1);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection_poses", 1);
        accumulated_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/accumulated_cloud", 1);
        
        ROS_INFO("Laser Cluster Node Started.");
    }

    // [추가] 유클리드 거리 계산
    double getDistance(const Box& b1, const Box& b2) {
        return std::sqrt(std::pow(b1.center_x - b2.center_x, 2) + std::pow(b1.center_y - b2.center_y, 2));
    }

    // [추가] 트래킹 로직 업데이트 함수
    void updateTracking(const std::vector<Box>& new_boxes, const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& new_cluster_points) {
        
        std::vector<bool> matched_new_box(new_boxes.size(), false);
        
        // 1. 기존 트랙들과 새로운 박스 매칭 (Greedy Nearest Neighbor)
        for (auto& track : tracks_) {
            double min_dist = std::numeric_limits<double>::max();
            int best_match_idx = -1;

            for (size_t i = 0; i < new_boxes.size(); ++i) {
                if (matched_new_box[i]) continue; // 이미 매칭된 박스는 패스

                double dist = std::abs(track.box.center_x - new_boxes[i].center_x) + std::abs(track.box.center_y - new_boxes[i].center_y);
                // 유클리드 거리 대신 맨해튼 거리로 1차 필터링 후 정밀계산 해도 됨 (여기선 간단히)
                dist = getDistance(track.box, new_boxes[i]);

                if (dist < min_dist) {
                    min_dist = dist;
                    best_match_idx = i;
                }
            }

            if (best_match_idx != -1 && min_dist < tracking_distance_th_) {
                // 매칭 성공: 정보 업데이트
                track.box = new_boxes[best_match_idx];
                track.points = *new_cluster_points[best_match_idx];
                track.no_detection_count = 0; // 카운트 초기화
                matched_new_box[best_match_idx] = true;
            } else {
                // 매칭 실패: 사라짐 카운트 증가
                track.no_detection_count++;
            }
        }

        // 2. 매칭되지 않은 새로운 박스는 신규 트랙 생성
        for (size_t i = 0; i < new_boxes.size(); ++i) {
            if (!matched_new_box[i]) {
                TrackedObject new_track;
                new_track.id = next_id_++;
                new_track.box = new_boxes[i];
                new_track.points = *new_cluster_points[i];
                new_track.no_detection_count = 0;

                // ID 기반 고유 색상 생성 (랜덤하지만 ID에 고정됨)
                // HSV to RGB 변환 흉내 혹은 단순 해싱
                new_track.r = (new_track.id * 50 + 20) % 255;
                new_track.g = (new_track.id * 100 + 50) % 255;
                new_track.b = (new_track.id * 150 + 100) % 255;
                
                // 너무 어두우면 밝게 보정
                if(new_track.r < 80 && new_track.g < 80 && new_track.b < 80) {
                    new_track.r += 100;
                }

                tracks_.push_back(new_track);
            }
        }

        // 3. 오래동안 감지 안된 트랙 삭제
        // remove_if를 사용하여 no_detection_count가 임계치를 넘으면 삭제
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const TrackedObject& t) {
                return t.no_detection_count > max_disappeared_frames_;
            }), tracks_.end());
    }

    Box fittingLShape(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
        Box best_box;
        double min_score = std::numeric_limits<double>::max(); 

        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        best_box.z_min = min_pt.z;
        best_box.z_max = max_pt.z;

        double step = 2.0; 
        
        double box_long_side = std::max(fixed_width_, fixed_length_);
        double box_short_side = std::min(fixed_width_, fixed_length_);

        for (double theta = 0.0; theta < 90.0; theta += step) {
            double rad = theta * M_PI / 180.0;
            double cos_t = std::cos(rad);
            double sin_t = std::sin(rad);

            double min_x = std::numeric_limits<double>::max();
            double max_x = -std::numeric_limits<double>::max();
            double min_y = std::numeric_limits<double>::max();
            double max_y = -std::numeric_limits<double>::max();

            std::vector<std::pair<double, double>> rotated_points;
            rotated_points.reserve(cluster->size());

            for (const auto& p : cluster->points) {
                double x_prime = p.x * cos_t + p.y * sin_t;
                double y_prime = -p.x * sin_t + p.y * cos_t;
                rotated_points.push_back({x_prime, y_prime});

                if (x_prime < min_x) min_x = x_prime;
                if (x_prime > max_x) max_x = x_prime;
                if (y_prime < min_y) min_y = y_prime;
                if (y_prime > max_y) max_y = y_prime;
            }

            double current_score = (max_x - min_x) * (max_y - min_y); // Area

            if (current_score < min_score) {
                min_score = current_score;
                best_box.heading = rad;

                double obs_len_x = max_x - min_x;
                double obs_len_y = max_y - min_y;
                double applied_w, applied_l;

                if (use_fixed_size_) {
                    if (obs_len_x > obs_len_y) {
                        applied_w = box_long_side; applied_l = box_short_side;
                    } else {
                        applied_w = box_short_side; applied_l = box_long_side;
                    }
                } else {
                    applied_w = obs_len_x; applied_l = obs_len_y;
                }
                
                best_box.width = applied_w;
                best_box.length = applied_l;

                double cx_1 = min_x + applied_w / 2.0; double cy_1 = min_y + applied_l / 2.0;
                double cx_2 = min_x + applied_w / 2.0; double cy_2 = max_y - applied_l / 2.0;
                double cx_3 = max_x - applied_w / 2.0; double cy_3 = min_y + applied_l / 2.0;
                double cx_4 = max_x - applied_w / 2.0; double cy_4 = max_y - applied_l / 2.0;

                double d1 = cx_1*cx_1 + cy_1*cy_1;
                double d2 = cx_2*cx_2 + cy_2*cy_2;
                double d3 = cx_3*cx_3 + cy_3*cy_3;
                double d4 = cx_4*cx_4 + cy_4*cy_4;

                double best_cx_prime, best_cy_prime;
                double min_d = std::numeric_limits<double>::max();

                if(d1 < min_d) { min_d = d1; best_cx_prime = cx_1; best_cy_prime = cy_1; }
                if(d2 < min_d) { min_d = d2; best_cx_prime = cx_2; best_cy_prime = cy_2; }
                if(d3 < min_d) { min_d = d3; best_cx_prime = cx_3; best_cy_prime = cy_3; }
                if(d4 < min_d) { min_d = d4; best_cx_prime = cx_4; best_cy_prime = cy_4; }

                best_box.center_x = best_cx_prime * cos_t - best_cy_prime * sin_t;
                best_box.center_y = best_cx_prime * sin_t + best_cy_prime * cos_t;

                if (best_box.width < best_box.length) {
                    best_box.heading += M_PI_2; 
                    std::swap(best_box.width, best_box.length); 
                }
            }
        }
        return best_box;
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
            
            // Fitting
            Box box = fittingLShape(current_cluster);
            current_boxes.push_back(box);
            current_cluster_points.push_back(current_cluster);
        }

        // --- UPDATE TRACKING ---
        updateTracking(current_boxes, current_cluster_points);

        // --- VISUALIZATION (Based on Tracks) ---
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        geometry_msgs::PoseArray pose_array_msg;
        pose_array_msg.header = scan_in->header; 

        // [수정] Coloring Cloud를 위한 준비
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_colored(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (size_t i = 0; i < tracks_.size(); ++i) {
            
            const TrackedObject& track = tracks_[i];
            
            // [수정] 놓친 물체(Ghost)인지 확인
            bool is_missing = (track.no_detection_count > 0);
            
            // 놓쳤다면 투명도를 0.3(흐릿함), 잡혔다면 1.0(선명함)
            float alpha_val = is_missing ? 0.3 : 1.0; 

            // 1. Pose (놓쳤어도 마지막 위치 발행)
            geometry_msgs::Pose pose;
            pose.position.x = track.box.center_x;
            pose.position.y = track.box.center_y;
            pose.position.z = (track.box.z_min + track.box.z_max) / 2.0;
            pose.orientation.w = cos(track.box.heading * 0.5);
            pose.orientation.z = sin(track.box.heading * 0.5);
            pose_array_msg.poses.push_back(pose);

            // 2. Marker - Box
            visualization_msgs::Marker box_marker;
            box_marker.header = scan_in->header;
            box_marker.ns = "boxes";
            box_marker.id = track.id;
            box_marker.type = visualization_msgs::Marker::LINE_STRIP;
            box_marker.action = visualization_msgs::Marker::ADD;
            box_marker.scale.x = 0.05; 
            
            box_marker.color.r = track.r / 255.0; 
            box_marker.color.g = track.g / 255.0; 
            box_marker.color.b = track.b / 255.0; 
            box_marker.color.a = alpha_val; // [수정] 투명도 적용
            box_marker.lifetime = ros::Duration(0.1);
            
            // ... (박스 좌표 계산 로직 동일) ...
            // (위 코드 복사해서 cx, cy 계산 부분 그대로 사용)
            double cos_t = std::cos(track.box.heading);
            double sin_t = std::sin(track.box.heading);
            double hw = track.box.width / 2.0;
            double hl = track.box.length / 2.0;
            double cx[5] = {hw, -hw, -hw, hw, hw};
            double cy[5] = {hl, hl, -hl, -hl, hl};

            for(int k=0; k<5; ++k) {
                geometry_msgs::Point p;
                p.x = track.box.center_x + (cx[k] * cos_t - cy[k] * sin_t);
                p.y = track.box.center_y + (cx[k] * sin_t + cy[k] * cos_t);
                p.z = track.box.z_min;
                box_marker.points.push_back(p);
            }
            marker_array.markers.push_back(box_marker);

            // 3. Text Marker (ID)
            // ... (기존 코드 동일, alpha만 적용해주면 좋음) ...
            // 3. Marker - Text (ID 표시)
            visualization_msgs::Marker text_marker;
            text_marker.header = scan_in->header;
            text_marker.ns = "ids";
            text_marker.id = track.id;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            std::string id_str = "ID: " + std::to_string(track.id);
            text_marker.text = id_str;
            text_marker.pose.position.x = track.box.center_x;
            text_marker.pose.position.y = track.box.center_y;
            text_marker.pose.position.z = track.box.z_max + 0.5; 
            text_marker.scale.z = 0.3; 
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = 1.0;
            text_marker.lifetime = ros::Duration(0.1);
            marker_array.markers.push_back(text_marker);

            // 4. PointCloud Coloring
            // [중요] 놓친 물체는 현재 포인트 클라우드가 없으므로(과거 데이터임),
            // 포인트를 그리지 않거나, 과거 포인트라도 그리고 싶다면 아래 로직 유지.
            // 보통 놓친 물체의 '과거 포인트'를 현재 씬에 그리면 
            // 실제로는 없는 자리에 점이 찍혀 헷갈리므로 PointCloud는 안 그리는 게 좋습니다.
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
    ros::init(argc, argv, "laser_cluster_tracking_node");
    LaserClusterNode node;
    ros::spin();
    return 0;
}
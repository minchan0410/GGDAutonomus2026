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

struct Box {
    double center_x;
    double center_y;
    double width;
    double length;
    double heading; // radian
    double z_min;
    double z_max;
};

class LaserClusterNode {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_; 
    
    ros::Subscriber scan_sub_;
    ros::Publisher cluster_pub_; 
    ros::Publisher marker_pub_;  
    ros::Publisher pose_pub_;    
    
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

public:
    LaserClusterNode() : private_nh_("~") { 
        
        // 1. Clustering Params
        private_nh_.param("cluster_tolerance", cluster_tolerance_, 0.15);
        private_nh_.param("min_cluster_size", min_cluster_size_, 10);
        private_nh_.param("max_cluster_size", max_cluster_size_, 500);

        // 2. Filtering Params
        private_nh_.param("max_cluster_extent", max_cluster_extent_threshold_, 1.2);
        private_nh_.param("min_cluster_extent", min_cluster_extent_threshold_, 0.4);

        // 3. Fixed Box Params
        private_nh_.param("use_fixed_size", use_fixed_size_, true);
        private_nh_.param("fixed_width", fixed_width_, 0.9);
        private_nh_.param("fixed_length", fixed_length_, 0.45);

        ROS_INFO("--------------------------------");
        ROS_INFO("Cluster Params: Tol=%.2f, Min=%d, Max=%d", cluster_tolerance_, min_cluster_size_, max_cluster_size_);
        ROS_INFO("Filter Params: MaxExtent=%.2f, MinExtent=%.2f", max_cluster_extent_threshold_, min_cluster_extent_threshold_);
        ROS_INFO("Box Params: Fixed=%s, W=%.2f, L=%.2f", use_fixed_size_ ? "True" : "False", fixed_width_, fixed_length_);
        ROS_INFO("--------------------------------");

        scan_sub_ = nh_.subscribe("/scan", 1, &LaserClusterNode::scanCallback, this);
        cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/clustered_cloud", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/detection_markers", 1);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detection_poses", 1);
        
        ROS_INFO("Laser Cluster Node Started.");
    }

    Box fittingLShape(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
        Box best_box;
        double min_score = std::numeric_limits<double>::max(); 

        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        best_box.z_min = min_pt.z;
        best_box.z_max = max_pt.z;

        double step = 1.0; 
        
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

            double current_score = 0.0;
            for (const auto& p_rot : rotated_points) {
                double d_xmin = std::abs(p_rot.first - min_x);
                double d_xmax = std::abs(p_rot.first - max_x);
                double d_ymin = std::abs(p_rot.second - min_y);
                double d_ymax = std::abs(p_rot.second - max_y);
                current_score += std::min({d_xmin, d_xmax, d_ymin, d_ymax});
            }

            if (current_score < min_score) {
                min_score = current_score;
                best_box.heading = rad;

                double obs_len_x = max_x - min_x;
                double obs_len_y = max_y - min_y;

                double applied_w, applied_l;

                if (use_fixed_size_) {
                    if (obs_len_x > obs_len_y) {
                        applied_w = box_long_side;
                        applied_l = box_short_side;
                    } else {
                        applied_w = box_short_side;
                        applied_l = box_long_side;
                    }
                } else {
                    applied_w = obs_len_x;
                    applied_l = obs_len_y;
                }
                
                best_box.width = applied_w;
                best_box.length = applied_l;

                // Anchor Strategy
                double cx_1 = min_x + applied_w / 2.0; double cy_1 = min_y + applied_l / 2.0;
                double cx_2 = min_x + applied_w / 2.0; double cy_2 = max_y - applied_l / 2.0;
                double cx_3 = max_x - applied_w / 2.0; double cy_3 = min_y + applied_l / 2.0;
                double cx_4 = max_x - applied_w / 2.0; double cy_4 = max_y - applied_l / 2.0;

                double d1 = cx_1*cx_1 + cy_1*cy_1;
                double d2 = cx_2*cx_2 + cy_2*cy_2;
                double d3 = cx_3*cx_3 + cy_3*cy_3;
                double d4 = cx_4*cx_4 + cy_4*cy_4;

                double best_cx_prime, best_cy_prime;
                double max_d = -1.0;

                if(d1 > max_d) { max_d = d1; best_cx_prime = cx_1; best_cy_prime = cy_1; }
                if(d2 > max_d) { max_d = d2; best_cx_prime = cx_2; best_cy_prime = cy_2; }
                if(d3 > max_d) { max_d = d3; best_cx_prime = cx_3; best_cy_prime = cy_3; }
                if(d4 > max_d) { max_d = d4; best_cx_prime = cx_4; best_cy_prime = cy_4; }

                best_box.center_x = best_cx_prime * cos_t - best_cy_prime * sin_t;
                best_box.center_y = best_cx_prime * sin_t + best_cy_prime * cos_t;

                // [핵심 수정] 화살표(Pose X축)가 항상 긴 쪽(Long Side)을 향하도록 조정
                // 현재 Width(X축 길이)가 Length(Y축 길이)보다 짧다면, 
                // Heading을 90도 돌리고 Width와 Length를 스왑함.
                if (best_box.width < best_box.length) {
                    best_box.heading += M_PI_2; // +90도 회전
                    std::swap(best_box.width, best_box.length); // 가로, 세로 길이 변경
                }
            }
        }
        return best_box;
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_in) {
        ros::Time start_time = ros::Time::now(); 

        sensor_msgs::PointCloud2 cloud_msg;
        projector_.projectLaser(*scan_in, cloud_msg); 

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud_msg, *cloud_raw);

        if (cloud_raw->empty()) return;

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_raw);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        
        ec.setClusterTolerance(cluster_tolerance_); 
        ec.setMinClusterSize(min_cluster_size_);    
        ec.setMaxClusterSize(max_cluster_size_);   
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_raw);
        ec.extract(cluster_indices);

        std::vector<std::pair<double, int>> cluster_distance_map;
        
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        geometry_msgs::PoseArray pose_array_msg;
        pose_array_msg.header = scan_in->header; 

        for (int i = 0; i < cluster_indices.size(); ++i) {
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            const auto& indices = cluster_indices[i].indices;
            for (int idx : indices) {
                current_cluster->points.push_back(cloud_raw->points[idx]);
            }
            current_cluster->width = current_cluster->points.size();
            current_cluster->height = 1;
            current_cluster->is_dense = true;

            // --- Brute-force Filter ---
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
            
            // --- Fitting ---
            Box box = fittingLShape(current_cluster);
            double dist = std::sqrt(box.center_x * box.center_x + box.center_y * box.center_y);
            cluster_distance_map.push_back(std::make_pair(dist, i));

            // PoseArray 추가
            geometry_msgs::Pose pose;
            pose.position.x = box.center_x;
            pose.position.y = box.center_y;
            pose.position.z = (box.z_min + box.z_max) / 2.0;
            pose.orientation.w = cos(box.heading * 0.5);
            pose.orientation.z = sin(box.heading * 0.5);
            pose_array_msg.poses.push_back(pose);

            // Visualization (Marker) - Box
            visualization_msgs::Marker box_marker;
            box_marker.header = scan_in->header;
            box_marker.ns = "boxes";
            box_marker.id = i;
            box_marker.type = visualization_msgs::Marker::LINE_STRIP;
            box_marker.action = visualization_msgs::Marker::ADD;
            box_marker.pose.orientation.w = 1.0;
            box_marker.scale.x = 0.05; 
            box_marker.color.r = 1.0; box_marker.color.g = 1.0; box_marker.color.b = 0.0; 
            box_marker.color.a = 1.0;
            box_marker.lifetime = ros::Duration(0.1);

            double cos_t = std::cos(box.heading);
            double sin_t = std::sin(box.heading);
            double hw = box.width / 2.0;
            double hl = box.length / 2.0;
            double cx[5] = {hw, -hw, -hw, hw, hw};
            double cy[5] = {hl, hl, -hl, -hl, hl};

            for(int k=0; k<5; ++k) {
                geometry_msgs::Point p;
                p.x = box.center_x + (cx[k] * cos_t - cy[k] * sin_t);
                p.y = box.center_y + (cx[k] * sin_t + cy[k] * cos_t);
                p.z = box.z_min;
                box_marker.points.push_back(p);
            }
            marker_array.markers.push_back(box_marker);

            // Visualization (Marker) - Center Sphere
            visualization_msgs::Marker center_marker;
            center_marker.header = scan_in->header;
            center_marker.ns = "centers";
            center_marker.id = i; 
            center_marker.type = visualization_msgs::Marker::SPHERE;
            center_marker.action = visualization_msgs::Marker::ADD;
            center_marker.pose.position.x = box.center_x;
            center_marker.pose.position.y = box.center_y;
            center_marker.pose.position.z = (box.z_min + box.z_max) / 2.0;
            center_marker.pose.orientation.w = 1.0;
            center_marker.scale.x = 0.15; 
            center_marker.scale.y = 0.15;
            center_marker.scale.z = 0.15;
            center_marker.color.r = 1.0; center_marker.color.g = 0.0; center_marker.color.b = 0.0; center_marker.color.a = 1.0;
            center_marker.lifetime = ros::Duration(0.1);
            marker_array.markers.push_back(center_marker);

            // Visualization (Marker) - Text
            visualization_msgs::Marker text_marker;
            text_marker.header = scan_in->header;
            text_marker.ns = "coords";
            text_marker.id = i; 
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            
            std::string coords_str = "(" + std::to_string(box.center_x).substr(0, 4) + ", " + 
                                     std::to_string(box.center_y).substr(0, 4) + ")";
            text_marker.text = coords_str;
            text_marker.pose.position.x = box.center_x;
            text_marker.pose.position.y = box.center_y;
            text_marker.pose.position.z = box.z_max + 0.3; 
            text_marker.scale.z = 0.2; 
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = 1.0;
            text_marker.lifetime = ros::Duration(0.1);
            marker_array.markers.push_back(text_marker);
        }

        marker_pub_.publish(marker_array);
        pose_pub_.publish(pose_array_msg);

        // Coloring & Publish Cloud (기존 유지)
        std::sort(cluster_distance_map.begin(), cluster_distance_map.end());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (const auto& pair_info : cluster_distance_map) {
            double dist = pair_info.first;
            int original_idx = pair_info.second;
            
            uint8_t r, g, b;
            if (dist > 5.0) { r=200; g=200; b=200; }
            else {
                double ratio = dist / 5.0; 
                if (ratio < 0.5) {
                    r = static_cast<uint8_t>(255 * (1.0 - 2 * ratio));
                    g = static_cast<uint8_t>(255 * (2 * ratio));
                    b = 0;
                } else {
                    r = 0;
                    g = static_cast<uint8_t>(255 * (2.0 - 2 * ratio));
                    b = static_cast<uint8_t>(255 * (2 * ratio - 1.0));
                }
            }
            for (const auto& point_idx : cluster_indices[original_idx].indices) {
                pcl::PointXYZRGB point;
                point.x = cloud_raw->points[point_idx].x;
                point.y = cloud_raw->points[point_idx].y;
                point.z = cloud_raw->points[point_idx].z;
                point.r = r; point.g = g; point.b = b;
                cloud_clustered->points.push_back(point);
            }
        }

        cloud_clustered->header = pcl_conversions::toPCL(scan_in->header);
        cloud_clustered->width = cloud_clustered->points.size();
        cloud_clustered->height = 1;
        cloud_clustered->is_dense = true;
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*cloud_clustered, output_msg);
        cluster_pub_.publish(output_msg);

        ros::Time end_time = ros::Time::now();
        ROS_INFO_THROTTLE(1.0, "Time: %.4f s", (end_time - start_time).toSec());
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "laser_cluster_final_full_node");
    LaserClusterNode node;
    ros::spin();
    return 0;
}
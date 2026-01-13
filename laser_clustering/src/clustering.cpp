#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>

// Visualization Markers
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// PCL Libraries
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <algorithm>
#include <cmath>
#include <string> // std::to_string

class LaserClusterNode {
private:
    ros::NodeHandle nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher cluster_pub_;
    ros::Publisher marker_pub_; // [추가] 마커 Publisher
    
    laser_geometry::LaserProjection projector_;

    const double MAX_CLUSTER_EXTENT_THRESHOLD = 1.2; 
    const double MIN_CLUSTER_EXTENT_THRESHOLD = 0.4;

public:
    LaserClusterNode() {
        scan_sub_ = nh_.subscribe("/scan", 1, &LaserClusterNode::scanCallback, this);
        cluster_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/clustered_cloud", 1);
        
        // [추가] MarkerArray 토픽 발행
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/cluster_markers", 1);
        
        ROS_INFO("Laser Cluster Node Started with Markers...");
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_in) {
        sensor_msgs::PointCloud2 cloud_msg;
        projector_.projectLaser(*scan_in, cloud_msg); 

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_raw(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud_msg, *cloud_raw);

        if (cloud_raw->empty()) return;

        // KD-Tree
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_raw);

        // Clustering
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        
        ec.setClusterTolerance(0.15); 
        ec.setMinClusterSize(10);    
        ec.setMaxClusterSize(500);   
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_raw);
        ec.extract(cluster_indices);

        std::vector<std::pair<double, int>> cluster_distance_map;
        
        // [추가] 마커 배열 생성
        visualization_msgs::MarkerArray marker_array;

        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);

        for (int i = 0; i < cluster_indices.size(); ++i) {
            
            // --- 크기 필터링 로직 ---
            double max_dist_sq = 0.0;
            const auto& indices = cluster_indices[i].indices;
            size_t num_points = indices.size();

            for (size_t j = 0; j < num_points; ++j) {
                for (size_t k = j + 1; k < num_points; ++k) {
                    const auto& p1 = cloud_raw->points[indices[j]];
                    const auto& p2 = cloud_raw->points[indices[k]];

                    double d_sq = (p1.x - p2.x) * (p1.x - p2.x) +
                                  (p1.y - p2.y) * (p1.y - p2.y) +
                                  (p1.z - p2.z) * (p1.z - p2.z);

                    if (d_sq > max_dist_sq) max_dist_sq = d_sq;
                }
            }

            double cluster_max_extent = std::sqrt(max_dist_sq);

            if (cluster_max_extent > MAX_CLUSTER_EXTENT_THRESHOLD || 
                cluster_max_extent < MIN_CLUSTER_EXTENT_THRESHOLD) {
                continue; 
            }
            // ---------------------

            // 중심점(Centroid) 계산
            double sum_x = 0, sum_y = 0, sum_z = 0;
            for (int idx : indices) {
                sum_x += cloud_raw->points[idx].x;
                sum_y += cloud_raw->points[idx].y;
                sum_z += cloud_raw->points[idx].z;
            }
            
            double cx = sum_x / num_points;
            double cy = sum_y / num_points;
            double cz = sum_z / num_points;

            // [추가] Marker 생성 (Sphere - 위치 표시)
            visualization_msgs::Marker marker;
            marker.header = scan_in->header; // 같은 프레임 사용
            marker.ns = "cluster_centroids";
            marker.id = i; // ID는 유니크해야 함
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = cx;
            marker.pose.position.y = cy;
            marker.pose.position.z = cz;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.1; // 지름 20cm 구
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;
            marker.color.a = 1.0; // 투명도
            marker.color.r = 1.0; // 노란색
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            marker.lifetime = ros::Duration(0.1); // 0.1초 뒤 사라짐 (잔상 방지)
            marker_array.markers.push_back(marker);

            // [추가] Marker 생성 (Text - 좌표 값 표시)
            visualization_msgs::Marker text_marker = marker;
            text_marker.id = i + 1000; // ID 겹치지 않게 오프셋
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.text = "(" + std::to_string(cx).substr(0,4) + ", " + std::to_string(cy).substr(0,4) + ")";
            text_marker.pose.position.z += 0.2; // 구 위에 띄우기
            text_marker.scale.z = 0.15; // 글자 크기
            text_marker.color.r = 1.0; 
            text_marker.color.g = 1.0; 
            text_marker.color.b = 1.0; // 흰색 글씨
            marker_array.markers.push_back(text_marker);


            // 거리 계산 및 저장 (기존 로직)
            double dist = std::sqrt(cx * cx + cy * cy + cz * cz);
            cluster_distance_map.push_back(std::make_pair(dist, i));
        }

        // 마커 발행
        marker_pub_.publish(marker_array);

        // --- (이하 색상 입히기 및 PointCloud 발행 로직 기존과 동일) ---
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
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "laser_cluster_node");
    LaserClusterNode node;
    ros::spin();
    return 0;
}
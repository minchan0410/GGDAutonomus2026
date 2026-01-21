#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <vector>
#include <cmath>
#include <algorithm>

class ScanToCostmap {
private:
    ros::NodeHandle nh_;
    ros::Subscriber scan_sub_;
    ros::Publisher map_pub_;

    // 맵 파라미터
    double map_resolution_; 
    int map_width_;         
    int map_height_;        
    
    // Inflation 파라미터
    double inflation_radius_; 

public:
    ScanToCostmap() {
        // 파라미터 설정
        map_resolution_ = 0.05; // 5cm 단위
        map_width_ = 400;       // 20m 폭
        map_height_ = 400;      // 20m 높이
        
        // [요청사항 1] Inflation 반경 0.1m 설정
        inflation_radius_ = 0.1; 

        scan_sub_ = nh_.subscribe("/scan", 1, &ScanToCostmap::scanCallback, this);
        map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/local_costmap", 1);
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan) {
        nav_msgs::OccupancyGrid map;
        
        // 1. 헤더 설정
        map.header = scan->header; 
        map.info.resolution = map_resolution_;
        map.info.width = map_width_;
        map.info.height = map_height_;

        // 맵의 원점을 로봇이 중앙에 오도록 오프셋 설정 (확인 완료)
        double origin_x = -(map_width_ * map_resolution_) / 2.0;
        double origin_y = -(map_height_ * map_resolution_) / 2.0;

        map.info.origin.position.x = origin_x;
        map.info.origin.position.y = origin_y;
        map.info.origin.position.z = 0.0;
        map.info.origin.orientation.w = 1.0;

        // 맵 데이터 초기화 (0: Free)
        map.data.resize(map_width_ * map_height_, 0);

        std::vector<int> obstacle_indices;

        // 2. Scan 데이터를 Grid로 변환
        float angle = scan->angle_min;
        for (size_t i = 0; i < scan->ranges.size(); ++i) {
            float range = scan->ranges[i];

            if (range >= scan->range_min && range <= scan->range_max && !std::isinf(range)) {
                double x = range * cos(angle);
                double y = range * sin(angle);

                int grid_x = (int)((x - origin_x) / map_resolution_);
                int grid_y = (int)((y - origin_y) / map_resolution_);

                if (grid_x >= 0 && grid_x < map_width_ && grid_y >= 0 && grid_y < map_height_) {
                    int index = grid_y * map_width_ + grid_x;
                    map.data[index] = 100; // 실제 장애물은 100
                    obstacle_indices.push_back(index);
                }
            }
            angle += scan->angle_increment;
        }

        // 3. Inflation 적용 (Flat 99)
        applyInflation(map, obstacle_indices);

        map_pub_.publish(map);
    }

    void applyInflation(nav_msgs::OccupancyGrid& map, const std::vector<int>& obstacles) {
        // 반경을 셀 단위로 변환
        int inflation_cells = (int)ceil(inflation_radius_ / map_resolution_);
        
        for (int idx : obstacles) {
            int c_x = idx % map_width_;
            int c_y = idx / map_width_;

            for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
                for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
                    // 유클리드 거리 계산
                    double dist_cells = sqrt(dx*dx + dy*dy);
                    double dist_meters = dist_cells * map_resolution_;

                    // 설정한 반경보다 크면 패스
                    if (dist_meters > inflation_radius_) continue;

                    int n_x = c_x + dx;
                    int n_y = c_y + dy;

                    // 맵 범위 체크
                    if (n_x >= 0 && n_x < map_width_ && n_y >= 0 && n_y < map_height_) {
                        int n_idx = n_y * map_width_ + n_x;
                        
                        // [요청사항 2] 실제 장애물(100)이 아닌 경우에만 99로 설정
                        // 기존 값이 100이면 건드리지 않음
                        if (map.data[n_idx] != 100) {
                            map.data[n_idx] = 99; // 그라데이션 없이 무조건 99
                        }
                    }
                }
            }
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "scan_to_flat_costmap_node");
    ScanToCostmap node;
    ros::spin();
    return 0;
}
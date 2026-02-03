# 2D LiDAR Car Detection

2D LiDAR를 활용하여 차량을 검출 및 tracking 하는 패키지.
![Demo GIF](./imgs/tracking1.gif)
<br><br>

## File Structure

```text
laser_detection/
├── CMakeLists.txt
├── config
│   └── params.yaml
├── launch
│   ├── combined.launch
│   └── run.launch
├── package.xml
├── README.md
├── rviz
│   └── detection.rviz 
└── src
    ├── detection.cpp
    └── occ_grid.cpp
```

<br>

## How to run
- 각각 launch

  for publish **/scan** topic
  ```shell
  roslaunch rplidar_ros rplidar_a1.launch
  ```
  run **laser_detection** package
  ```shell
  roslaunch laser_detection run.launch
  ```
- 한번에 launch 
  ```shell
  roslaunch laser_detection combined.launch
  ```
  alias "lidar" on **vicbook**

<br>

## Topics

### Input Topic
| Name | Type | Uses |
| :--- | :--- | :--- |
| `/scan` | `sensor_msgs/LaserScan` | raw data |

### Output Topics
| Name | Type | Uses |
| :--- | :--- | :--- |
| `/local_costmap` | `nav_msgs/OccupancyGrid` | occupancy grid |
| `/clustered_cloud` | `sensor_msgs/PointCloud2` | visualization & clustering |
| `/detection_markers` | `visualization_msgs/MarkerArray` | visualization circle |
| `/detection_poses` | `geometry_msgs/PoseArray` | center point, ID (x, y, ID) |
| `/detection_poses` | `geometry_msgs/PoseArray` | center point for visualization (z = 0) |

<br>

## Process
1. 유클리디안 클러스터링.
2. 클러스터 내에서 두 점을 있는 가장 긴 직선의 길이를 구하고, 해당 거리 기반으로 차량으로 추정되는 클러스터를 filtering
3. 각 클러스터에 ID를 부여하여 tracking 할 수 있도록 함.
4. 한번 clustering 조건에 부합하면 유클리디안 클러스터링 조건에 맞지 않아도 pointcloud가 완전히 사라지지 않으면 계속 tracking.

5. 완전히 사라지더라도 객체의 마지막 속도 기반으로 Constant Velocity Model을 적용하여 10 프레임 동안 유지.

- **why not L-shape fitting?** (AABB) : lidar에 찍히는 실제 차량의 pointcloud가 sparse하고 L-Shape이 아닌 경우가 많기 때문에 사각형 fitting이 흔들려 center point도 흔들리게 된다. parking planner 에서 stanley path의 안정성이 중요함으로, 클러스터의 중점을 기반으로 한 피팅 방법을 사용.

### ToDo..
- fitting 과정 중 정말 차가 아닌 것은 제거할 수 있도록. 하는 알고리즘을 추가.
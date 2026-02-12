# 2D LiDAR Car Detection

### 2D LiDAR를 활용하여 주차 미션을 위한 차량을 Detection 및 Tracking 하는 패키지.
![Demo GIF](./assets/tracking1.gif)
<br>
- Id : 4, 6 is parked car
<br>

## Process
![process block](./assets/flow.png)
1. **/scan**을 받아 전처리(ROI) 및 **PointCloud**로 변환 
2. 이전 프레임에 감지된 객체로부터 **Constant Velocity** 모델을 적용하여 예상 위치 지정. <br>만약 감지된 프레임이 없다면 곧바로 **Euclidean Clustering**을 통해 객체 감지. 
3. 각 객체의 예상 위치로부터 주변의 **Pointcloud를 객체에 할당**. 
4. 객체가 할당된 Pointcloud를 가지고 있으면 해당 PointCloud에 **Circle fitting**.<br> PointCloud가 할당되지 않았으면 **Pred 상태 유지**(ex 10프레임 동안 CV로 Prediction)
5. 객체에 할당되지 않은 나머지 **PointCloud에 대해서 Euclidean Clustering** 진행.


### Key Strategies
- 2D Lidar의 불안정성과 검출되는 Point의 부족함을 고려하여 **Tracking**이 가능하도록 하였음. <br> 또한, 한번 감지한 물체는 처음 Euclidean clustering의 조건에 맞지 않더라도 남아있는 PointCloud가 하나라도 있다면 **지속해서 감지**할 수 있도록 설계함.

- **why not L-shape fitting?** <br>
Lidar에 찍히는 실제 차량의 PointCloud가 sparse하고 L-Shape이 아닌 상황 많기 때문에 사각형 fitting이 흔들려 center point도 흔들리게 된다. 해당 알고리즘은 parking planner 에서 stanley path을 만드는데 사용됨으로, detection되는 객체의 center point 안정성이 중요하다. 따라서 PointCloud의 중심점을 기반으로 한 피팅 방법을 사용하였다.

- **Circle fitting method** <br>
Clustering된 Point Cloud를 통해 차량의 중심으로 예상되는 지점을 차 크기를 고려한 원으로 추정. <br>
  
  1. Cluster의 무게 중심 m을 계산.
  2. lidar -> cluster 벡터(v)를 구하고 Cluster의 point 들을 정사영시킨다.
  3. 센서로부터 가까운 하위 10% 지점을 구하고 해당 지점 기준 v 방향으로 반지름만큼 떨어진 위치를 중점으로 정한다. 
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

### ToDo..
- fitting 과정 중 정말 차가 아닌 것(옷, 다리 등)은 제거할 수 있도록. 하는 알고리즘을 추가.
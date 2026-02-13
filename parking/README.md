# Parking

### LiDAR, Camera 기반 FSM Parking Planner

---

### Simulation / Test Video
![Parking Demo 1](./assets/parking1.gif)

- ID 5, 7 : parked car

---

### Competition Day Run
![Parking Demo 2](./assets/parking2.gif)

- 대회 당일 주행 영상

---

## System Process

![Process Diagram](./assets/parking.png)

1. LiDAR 기반 차량 검출  
2. 차선 정보 기반 lateral control  
3. FSM 상태 전이 기반 Parking 전략 수행  
4. Stanley Controller 기반 경로 추종  

---

## FSM Explanation

### 1. lane_driving:
진행방향 좌측 평행주차 구간의 선을 검출해 조향각 '/parking_lane_steer'을 받아 주행. <br>
'/detection_poses' 토픽으로부터 fitting된 객체들의 local 좌표를 받으면서 우측 사각형 ROI 내의 영역에서 첫번째로 주차된 차량을 확정함.
<br>

### 2. full_left_steer:
첫번째 차량을 확정 후, 18 degree 고정 조향각으로 좌측으로 꺾으며 주행.
<br>
확정된 첫번째 차량을 기준으로 새로운 부채꼴 ROI 생성, 이후 같은 방식으로 fitting된 객체들 중 두번째로 주차된 차량을 확정함.
<br>
주행중인 차량과 주차되어있는 차량들 간의 기하학적 조건, 그리고 최소한으로 보장된 좌측 조향 시간 조건을 고려해 적절한 위치에 정차.

### 3. pause_after_left:
적절한 시간 정지하며, 확정된 두 점 정보를 바탕으로 후진하는 stanley path를 생성함.

### 4. stanley:
정차된 두 차량을 잇는 선의 중점을 지나고 이 선에 수직인 path를 지속적으로 생성해 **stanley.py** 노드에 '/stanley_path'를 publish함.
<br>
**stanley.py** 노드는 e1, e2 에러를 기반으로 Closed-loop feedback 제어를 통해 적절한 후진 조향각을 지속적으로 생성함.

### 5. stop:
주차에서 차량 중심은 라이다 센서의 중심인 laser frame으로 정의됨. 
<br>
차량 중심이 path 위에서 목표점의 local 좌표를 넘어서는 순간에 정지함.

### 6. pull_out1, pull_out2, finishing:
Open-loop 제어.
<br>
실험적으로 결정된 조향각 및 종방향 속도를 사용해 목적지까지 도달.
<br>
---

## Key Strategies

### 1. Why Stanley Control?
처음에 전통적인 방식인 pure pursuit 방식을 사용해봤으나 e2 에러를 좁히기 쉽지 않다는 판단을 내렸고
<br> 목표점에서 차량이 정차된 차량과 동일하게 주차선에 정렬되도록 e1, e2 에러를 명시적으로 사용하는 stanley 제어기를 사용함. <br> 후진 전, 정차한 위치에 크게 상관없이 주차공간에 위치한 두 차량이 detection 및 tracking 되기만 한다면 유연하고 정확한 path 추종이 가능했음.

### 2. Tracking both objects
fitting된 두 객체가 주차공간에 정차된 두 차량인지 확인하고, 두 좌표를 지속적으로 업데이트하는 게 중요했음. <br> 다음 조건을 만족할 경우 안정적인 path 생성이 가능함.
  - 두 객체가 같은 frame에 동시에 업데이트되어야 함.
  - fitting된 중심점이 흔들리지 않아야 함.
  - 중심점이 실제 정차된 차량의 중심점과 일치해야 함.
  - 3 프레임을 초과하여 끊기지 않아야 함. (**Planner**의 메인 루프는 20Hz)

---

## Topics

### Input Topics

| Name | Type | Description |
| :--- | :--- | :--- |
| `/detection_poses` | `geometry_msgs/PoseArray` | 검출된 차량 Pose 정보 |
| `/parking_lane_steer` | `std_msgs/Int16` | 차선 기반 조향 정보 |

---

### Output Topics

| Name | Type | Description |
| :--- | :--- | :--- |
| `/des_steer` | `std_msgs/Int16` | 최종 조향 명령 |
| `/motor_cmd_long` | `std_msgs/Int16` | 종방향 속도 제어 |
| `/stanley_path` | `nav_msgs/Path` | 생성된 주차 경로 |
| `/parking_viz` | `visualization_msgs/MarkerArray` | RViz 디버그 마커 |
| `/roi_marker` | `visualization_msgs/Marker` | ROI 시각화 |
| `/debug_overlay_text` | `jsk_rviz_plugins/OverlayText` | 디버그 텍스트 표시 |

---

## File Structure

```text
parking/
├── CMakeLists.txt
├── config/
│   └── parking.yaml
├── launch/
│   └── combined.launch
├── package.xml
├── README.md
├── rviz/
│   └── detection.rviz 
└── scripts/
    ├── parking.py
    └── stanley.py
```

<br>

## How to Run

- Launch File 실행
  ```shell
  roslaunch parking combined.launch
  ``` 
  해당 combined.launch 파일은 다음 노드들을 모두 실행

    * RPLIDAR
    * laser_detector
    * lane_detector
    * Stanley Controller
    * Parking FSM Planner
    * RViz

- Alias 등록

  .bashrc에 alias를 등록하여 간편하게 실행
  ```shell
  gedit ~/.bashrc
  ``` 
  아래 내용 추가
  ```shell
  alias parking='roslaunch parking combined.launch'
  ``` 
  변경 사항 적용
  ```shell
  source ~/.bashrc
  ``` 
  이후 터미널에서 아래 명령어로 실행
  ```shell
  parking
  ``` 

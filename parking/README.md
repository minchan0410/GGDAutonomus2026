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

## Key Strategies

- LiDAR 기반 ROI 필터링 및 차량 클러스터링
- FSM 기반 주차 상태 전이 설계
- Stanley Controller를 이용한 곡선 경로 추종
- RViz Marker 및 OverlayText를 활용한 디버깅 시각화

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

roslaunch parking combined.launch

해당 combined.launch 파일은 다음 노드들을 모두 실행

- RPLIDAR
- laser_detector
- lane_detector
- Stanley Controller
- Parking FSM Planner
- RViz

- Alias 등록

.bashrc에 alias를 등록하여 간편하게 실행
```shell
gedit ~/.bashrc
``` 
아래 내용 추가:
```shell
alias parking='roslaunch parking combined.launch'
``` 
변경 사항 적용:
```shell
source ~/.bashrc
``` 
이후 터미널에서 아래 명령어로 실행:
```shell
parking
``` 
# Parking

### LiDAR parked-car detection 결과와 Stanley path 추종을 이용해 평행주차 FSM을 수행하는 parking package

![Parking Demo 1](./assets/parking1.gif)

![Parking Demo 2](./assets/parking2.gif)

## System Process

![Parking Process Diagram](./assets/parking.png)

1. `laser_detector` 가 parked car 후보를 검출해 `/detection_poses` 를 제공한다.
2. 주차 진입 구간에서는 차선 기반 조향 입력 `/parking_lane_steer` 를 사용한다.
3. `parking.py` 가 parked car tracking 과 FSM 전이를 수행하며 필요 시 `/stanley_path` 를 생성한다.
4. `stanley.py` 가 `/stanley_path` 를 추종해 `/parking_stanley_steer` 를 만들고, `parking.py` 가 최종 `/des_steer`, `/motor_cmd_long` 을 출력한다.

## Package Role

`parking` 패키지는 주차 미션 구간에서 parked car detection 결과와 조향 입력을 받아 평행주차 상태기계를 수행하고, 주차 및 탈출 구간의 조향/종방향 명령을 생성하는 역할을 담당한다.  
본 패키지의 핵심 책임은 `/detection_poses` 로부터 주차 대상 차량을 추적하고, 필요 시 `/stanley_path` 를 생성하며, 최종적으로 `/des_steer` 와 `/motor_cmd_long` 을 publish 하는 것이다.

## System Boundary

- 입력:
  `/detection_poses`, `/parking_lane_steer`, `/parking_stanley_steer`, `/rosserial_check`
- 주요 출력:
  `/des_steer`, `/motor_cmd_long`, `/stanley_path`, `/parking_viz`, `/roi_marker`, `/debug_overlay_text`, `/stanley_debug`, `/stanley_debug_text`
- 책임 범위:
  주차 대상 차량 추적, 주차 시작 판단, 주차 FSM 전이, Stanley path 생성, 주차 중 조향/속도 명령 publish, 주차 시각화
- 책임 범위 아님:
  parked car 검출 자체, 저수준 steering PID, Arduino 하드웨어 구동, 일반 차선 검출
- 상위/하위 관계:
  입력 : `laser_detector`, `lane_detector`, `support`, `stanley.py`
  출력 : `lateral_controller`, `arduino_motor_bridge`, RViz 및 debug 확인 환경

## Interface Summary

| Direction | Topic | Type | Description | Used by |
| :--- | :--- | :--- | :--- | :--- |
| Input | `/detection_poses` | `geometry_msgs/PoseArray` | parked car 중심 좌표 목록 | `parking.py` |
| Input | `/parking_lane_steer` | `std_msgs/Int16` | 주차 진입 구간 차선 기반 조향 입력 | `parking.py` |
| Input | `/parking_stanley_steer` | `std_msgs/Int16` | Stanley path 추종 조향 입력 | `parking.py` |
| Input | `/rosserial_check` | `std_msgs/Int16` | rosserial link 상태 | `parking.py` |
| Output | `/stanley_path` | `nav_msgs/Path` | 주차용 후진 path | `stanley.py`, RViz |
| Output | `/des_steer` | `std_msgs/Int16` | 최종 조향 명령 | `lateral_controller` |
| Output | `/motor_cmd_long` | `std_msgs/Int16` | 종방향 PWM 명령 | `arduino_motor_bridge` |
| Output | `/parking_viz` | `visualization_msgs/MarkerArray` | parked car / destination 시각화 | RViz |
| Output | `/roi_marker` | `visualization_msgs/Marker` | parking ROI 시각화 | RViz |
| Output | `/debug_overlay_text` | `jsk_rviz_plugins/OverlayText` | parking FSM 디버그 텍스트 | RViz |
| Output | `/stanley_debug` | `visualization_msgs/Marker` | Stanley debug marker | RViz |
| Output | `/stanley_debug_text` | `jsk_rviz_plugins/OverlayText` | Stanley debug text | RViz |

## Node Summary

- `parking.py`
  - `parking` 패키지의 메인 FSM planner 노드다.
  - parked car 추적, ROI 관리, parking path 생성, 최종 `/des_steer`, `/motor_cmd_long` 출력을 담당한다.
- `stanley.py`
  - `/stanley_path` 를 받아 후진 주차용 조향각을 계산하고 `/parking_stanley_steer` 를 publish 하는 노드다.
  - marker와 overlay text를 함께 publish 한다.
- `lane.py`
  - package 내부에 존재하는 단순 lane steering 노드다.
  - 다만 현재 `parking/launch/combined.launch` 는 이 노드를 실행하지 않고 `lane_detector/run.py` 를 사용해 `/parking_lane_steer` 를 공급한다.
- `cam_test.py`
  - 실험용 스크립트 성격의 테스트 노드다.
  - 현재 주차 메인 파이프라인에는 포함되지 않는다.

## Requirements Summary

### `parking.py` node

parked car 추적 및 주차 시작 판단
- Description:
  `parking.py` 는 `/detection_poses` 를 입력으로 받아 첫 번째 parked car 와 두 번째 parked car 를 추적하고, 두 객체가 유효하게 갱신될 때만 주차 시작 가능 상태와 목적 지점을 계산해야 한다.
- Interface:

  | Direction | Topic | Type |
  | :--- | :--- | :--- |
  | Input | `/detection_poses` | `geometry_msgs/PoseArray` |
  | Output | `/parking_viz` | `visualization_msgs/MarkerArray` |
  | Output | `/roi_marker` | `visualization_msgs/Marker` |

- Verification:
  parked car 가 포함된 rosbag replay 또는 실시간 입력에서 `/parking_viz` 와 `/roi_marker` 가 생성되고, 두 객체가 모두 유효한 구간에서 주차 목적 지점이 안정적으로 갱신되는지 확인한다.
- Constraint / Fault Note:
  detection 결과가 끊기거나 두 차량 중 하나라도 추적이 불안정하면 Stanley path 생성과 상태 전이가 지연될 수 있다.

주차 FSM 전이 및 path 생성
- Description:
  메인 노드는 `lane_driving`, `full_left_steer`, `pause_after_left`, `stanley`, `stop`, `pull_out1`, `pull_out2`, `finishing` 상태를 관리하고, 주차 시작 조건이 만족되면 `/stanley_path` 를 생성해야 한다.
- Interface:

  | Direction | Topic | Type |
  | :--- | :--- | :--- |
  | Input | `/parking_lane_steer` | `std_msgs/Int16` |
  | Input | `/parking_stanley_steer` | `std_msgs/Int16` |
  | Output | `/stanley_path` | `nav_msgs/Path` |
  | Output | `/debug_overlay_text` | `jsk_rviz_plugins/OverlayText` |

- Verification:
  주차 시나리오 입력에서 상태 전이에 따라 `/stanley_path` 가 생성되고, RViz 또는 overlay text 로 현재 상태 변화를 확인할 수 있는지 점검한다.
- Constraint / Fault Note:
  `full_left_steer`, `pull_out1`, `pull_out2`, `finishing` 구간은 open-loop timing 기반이므로 차량 속도나 노면 조건 변화에 민감하다.

조향/종방향 명령 출력 및 serial freeze
- Description:
  `parking.py` 는 lane steer, Stanley steer, 고정 조향 명령 중 현재 상태에 맞는 소스를 선택해 `/des_steer` 와 `/motor_cmd_long` 을 publish 해야 하며, serial 이상 시 freeze 조건에서 0 명령으로 안전 정지해야 한다.
- Interface:

  | Direction | Topic | Type |
  | :--- | :--- | :--- |
  | Input | `/rosserial_check` | `std_msgs/Int16` |
  | Output | `/des_steer` | `std_msgs/Int16` |
  | Output | `/motor_cmd_long` | `std_msgs/Int16` |

- Verification:
  parking mode 실행 중 `/des_steer` 와 `/motor_cmd_long` 이 상태에 따라 갱신되는지 확인하고, `/rosserial_check` 비정상 조건에서 0 명령이 유지되는지 확인한다.
- Constraint / Fault Note:
  현재 종방향은 dedicated longitudinal controller가 아니라 planner 내부의 상태 기반 고정 PWM 명령이다.

### `stanley.py` node

주차 path 기반 조향각 계산
- Description:
  `stanley.py` 는 `/stanley_path` 를 입력으로 받아 lateral error 와 heading error 기반 Stanley steering 을 계산하고 `/parking_stanley_steer` 를 publish 해야 한다.
- Interface:

  | Direction | Topic | Type |
  | :--- | :--- | :--- |
  | Input | `/stanley_path` | `nav_msgs/Path` |
  | Output | `/parking_stanley_steer` | `std_msgs/Int16` |
  | Output | `/stanley_debug` | `visualization_msgs/Marker` |
  | Output | `/stanley_debug_text` | `jsk_rviz_plugins/OverlayText` |

- Verification:
  유효한 `/stanley_path` 입력 시 `/parking_stanley_steer` 가 생성되고, 빈 path 입력 시 조향 명령이 0으로 떨어지며 marker가 정리되는지 확인한다.
- Constraint / Fault Note:
  현재 Stanley 구현은 고정 상수 `K`, `L`, `V`, `k_h`, `k_l` 를 사용하며, motion yaw 를 후진 기준으로 가정한다.

## Verification Scenario

- 실행 준비 / 확인 topic:
  parked car detection 과 `/parking_lane_steer` 가 들어오는 rosbag replay 또는 실차 환경 준비, 확인 topic 은 `/detection_poses`, `/stanley_path`, `/parking_stanley_steer`, `/des_steer`, `/motor_cmd_long`, `/parking_viz`, `/roi_marker`
- 확인 방법:
  `rostopic echo`, RViz marker / overlay 확인, 상태별 output 변화 확인

권장 bag:
- `kkd_parking1.bag`
- `kkd_parking2.bag`
- `kkd_parking3.bag`

통과 판단 기준:
- parked car 2대가 안정적으로 추적될 때 `/stanley_path` 가 생성된다.
- Stanley 구간에서 `/parking_stanley_steer` 와 최종 `/des_steer` 가 갱신된다.
- 상태에 따라 `/motor_cmd_long` 이 정지, 전진, 후진 값으로 바뀐다.
- serial 이상 조건에서 `/des_steer`, `/motor_cmd_long` 이 0으로 제한된다.

## Parameters

주요 설정은 `config/parking.yaml` 에서 읽고, 일부 freeze 관련 값과 Stanley gain은 코드 상수를 사용한다.

| Source | Name | Value | Meaning |
| :--- | :--- | :--- | :--- |
| `parking.yaml` | `vehicle/full_left_steer` | `18` | 좌측 고정 조향 command |
| `parking.yaml` | `vehicle/full_right_steer` | `-22.5` | 우측 고정 조향 command |
| `parking.yaml` | `vehicle/forward_speed` | `175` | 전진 PWM |
| `parking.yaml` | `vehicle/backward_speed` | `-100` | 후진 PWM |
| `parking.yaml` | `timing/pause` | `0.5` | left steer 후 정지 시간 |
| `parking.yaml` | `timing/parking_pause` | `3` | 주차 완료 후 정지 시간 |
| `parking.yaml` | `timing/going_right` | `4.6` | pull_out2 우회전 유지 시간 |
| `parking.yaml` | `timing/detection_start` | `2` | detection 반영 시작 지연 |
| `parking.yaml` | `timing/pull_out` | `1` | pull_out1 유지 시간 |
| `parking.yaml` | `timing/left_steer` | `4` | full left steer 유지 시간 |
| `parking.yaml` | `threshold/can_park` | `-4` | 주차 시작 가능 판정 threshold |
| `parking.yaml` | `threshold/track_max_dist` | `1.0` | parked car tracking 최대 거리 |
| `parking.yaml` | `roi/lane` | `x[-1,1], y[-3,-1]` | lane driving ROI |
| `parking.yaml` | `roi/full_left` | `r[1.2,2.8], angle 60` | full-left 단계 ROI |
| `parking.yaml` | `roi/lost` | `r[1.2,2.8], angle 180` | lost target 재탐색 ROI |
| `parking.yaml` | `stanley_path/back_len` | `2.0` | 후방 path 길이 |
| `parking.yaml` | `stanley_path/front_len` | `5.0` | 전방 path 길이 |
| `parking.yaml` | `stanley_path/step` | `0.1` | path sampling step |
| `parking.yaml` | `stanley_path/frame_id` | `laser` | path 기준 frame |
| code default | `freeze_on_serial_loss` | `true` | serial 이상 시 freeze 여부 |
| code constant | `SERIAL_OK_TICKS / SERIAL_BAD_TICKS` | `20 / 4` | parking serial hysteresis |
| code constant | `Stanley K / L / V / k_h / k_l` | `1.0 / 0.8 / 1.0 / 1.0 / 1.0` | Stanley steering 계산 상수 |

## Limitations / Fault Cases

- 주차 성능은 `/detection_poses` 품질과 두 parked car 추적 안정성에 크게 의존한다.
- `pull_out1`, `pull_out2`, `finishing` 은 open-loop 제어다.
- `combined.launch` 는 `rplidar_ros`, `laser_detector`, `lane_detector` 같은 외부 패키지 의존을 함께 가진다.

## Implementation Notes

1. `parking.py` 는 두 parked car 를 추적해 목적 지점을 계산하고 상태별로 lane steer, Stanley steer, 고정 steer 중 하나를 선택한다.
2. `parking.py` 는 parked car 두 점의 평균과 법선 방향을 이용해 `/stanley_path` 를 생성한다.
3. `stanley.py` 는 `/stanley_path` 를 따라 후진 조향각을 계산하고 debug marker와 overlay text 를 함께 publish 한다.
4. serial 오류가 누적되면 freeze 모드로 들어가고 복구 후 상태를 이어서 진행한다.

## Run Guide

- parking 전체 실행: `roslaunch parking combined.launch`
- `combined.launch` 포함 노드:
  - `rplidar_ros/rplidarNode`
  - `laser_detector/detection`
  - `lane_detector/run.py`
  - `parking/stanley.py`
  - `parking/parking.py`
  - `rviz`

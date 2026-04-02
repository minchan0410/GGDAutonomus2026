# Object Detector

### 전방/신호등 카메라 영상을 이용해 차량 검출, 지면 좌표 투영, 신호 상태 인식을 제공하는 perception 패키지

## System Process

![Object Detector Diagram](./assets/object%20detector%20diagram.png)

## Package Role

`object_detector` 패키지는 카메라 입력으로부터 전방 차량과 신호등 상태를 추정하고, planner가 사용할 수 있는 형태의 perception 결과를 publish 하는 역할을 담당한다.  
본 패키지의 핵심 책임은 `/cam1/usb_cam/image_raw` 에서 차량을 검출해 `/car_projected` 로 변환하고, `/cam2/usb_cam/image_raw` 에서 신호 상태를 검출해 `/traffic` 로 제공하는 것이다.

## System Boundary

- 입력:
  `/cam1/usb_cam/image_raw`, `/cam2/usb_cam/image_raw`
- 주요 출력:
  `/car_detection`, `/car_projected`, `/car_projected_markers`, `/traffic`, `/yolo_overlay/image`, `/traffic_overlay/image`
- 책임 범위:
  차량 detection, 차량 bbox 기반 지면 좌표 투영, 신호 상태 분류, RViz/debug overlay publish
- 책임 범위 아님:
  장애물 회피 판단, 차량 추종 제어, 경로 생성, 최종 정지/출발 판단
- 상위/하위 관계:
  입력 : `usb_cam`
  출력 : `pre_final_planner`, RViz 및 debug 확인 환경

## 인터페이스

| Direction | Topic | Type | Description | Used by |
| :--- | :--- | :--- | :--- | :--- |
| Input | `/cam1/usb_cam/image_raw` | `sensor_msgs/Image` | 전방 차량 검출용 camera image | `object_detector` |
| Input | `/cam2/usb_cam/image_raw` | `sensor_msgs/Image` | 신호등 검출용 camera image | `object_detector` |
| Output | `/car_detection` | `vision_msgs/Detection2DArray` | 차량 bbox detection 결과 | `object_projection.py` |
| Output | `/car_projected` | `geometry_msgs/PoseArray` | planner용 차량 지면 좌표 | `pre_final_planner` |
| Output | `/car_projected_markers` | `visualization_msgs/MarkerArray` | projected 차량 marker | RViz |
| Output | `/traffic` | `std_msgs/Int16` | 신호 상태 결과 (`0` NONE, `1` GREEN, `2` RED, `3` YELLOW) | `pre_final_planner` |
| Output | `/yolo_overlay/image` | `sensor_msgs/Image` | 차량 detection overlay image | RViz / debug |
| Output | `/traffic_overlay/image` | `sensor_msgs/Image` | 신호 detection overlay image | RViz / debug |

## Node

- `object_detection.py`
  - `object_detector.launch`에서 실행되는 차량 검출 노드다.
  - YOLO 결과 중 `car_class_name` 에 해당하는 bbox만 `/car_detection` 으로 publish 한다.
- `object_projection.py`
  - `/car_detection` 을 받아 지면으로 투영하여 `/car_projected` 를 생성하는 노드다.
  - projected marker를 `/car_projected_markers` 로 함께 publish 할 수 있다.
- `traffic_detection.py`
  - 신호등 ROI 내부 YOLO 결과를 이용해 `/traffic` 상태를 publish 하는 노드다.
  - overlay를 통해 현재 프레임의 박스와 상태를 함께 표시한다.

## 요구사항

**대상 노드:** `object_detection.py`, `object_projection.py`

### 기능 요구사항

| 기능 | 설명 | Input | Output |
| :--- | :--- | :--- | :--- |
| 전방 차량 bbox 검출 및 publish | YOLO 결과 중 `car_class_name` 에 해당하는 객체만 선택해 publish 해야 한다 | `/cam1/usb_cam/image_raw` | `/car_detection` |
| 차량 detection 결과의 지면 좌표 투영 | bbox 하단 중심을 카메라 파라미터 기반으로 지면에 투영하여 `/car_projected` 를 생성해야 한다 | `/car_detection` | `/car_projected`, `/car_projected_markers` |

### 비기능 요구사항

| 항목 | 설명 | 기준 |
| :--- | :--- | :--- |
| 처리 주기 | `object_detection.py` 와 `object_projection.py` 는 `/cam1/usb_cam/image_raw` 입력을 누락 없이 처리할 수 있도록 전방 카메라 입력 주기를 따라가야 한다. | `/cam1/usb_cam/image_raw` 입력 `30 Hz`, 1 frame 처리 시간 `33 ms` 이내 |
| CPU 사용률 | vehicle perception 파이프라인은 다른 주행 노드와 병행 실행 가능하도록 시스템 자원을 과도하게 점유하지 않아야 한다. | CPU 사용률 `40%` 이내 |
| 출력 좌표계 일관성 | 지면 투영 결과는 planner가 바로 사용할 수 있도록 단일 기준 좌표계로 publish 되어야 한다. | `/car_projected`, `/car_projected_markers` 의 `frame_id` 는 `base_link` |

**대상 노드:** `traffic_detection.py`

### 기능 요구사항

| 기능 | 설명 | Input | Output |
| :--- | :--- | :--- | :--- |
| 신호등 상태 분류 및 publish | ROI 내부 신호등 YOLO 결과를 분류하여 `/traffic` 상태값을 publish 해야 한다 | `/cam2/usb_cam/image_raw` | `/traffic`, `/traffic_overlay/image` |

### 비기능 요구사항

| 항목 | 설명 | 기준 |
| :--- | :--- | :--- |
| 처리 주기 | `traffic_detection.py` 는 `/cam2/usb_cam/image_raw` 입력을 누락 없이 처리할 수 있도록 신호등 카메라 입력 주기를 따라가야 한다. | `/cam2/usb_cam/image_raw` 입력 `10 Hz`, 1 frame 처리 시간 `100 ms` 이내 |
| CPU 사용률 | 신호등 검출 처리는 다른 perception 및 planner 노드와 병행 실행 가능하도록 시스템 자원을 과도하게 점유하지 않아야 한다. | CPU 사용률 `40%` 이내 |
| 실행 환경 | `traffic_detection.py` 는 YOLO 추론을 위해 CUDA 사용 가능 환경에서만 동작해야 하며, GPU가 없으면 시작 단계에서 종료되어야 한다. | CUDA 사용 가능 GPU 필요 |

## 검증 bag

- 실행 준비 / 확인 topic:
  `/cam1/usb_cam/image_raw`, `/cam2/usb_cam/image_raw` 가 포함된 rosbag replay 또는 실시간 camera 환경 준비, 확인 topic 은 `/car_detection`, `/car_projected`, `/car_projected_markers`, `/traffic`, `/yolo_overlay/image`, `/traffic_overlay/image`
- 확인 방법:
  `rostopic echo`, RViz image / marker 확인, planner 입력 topic 생성 여부 확인

권장 bag :
- `lane_change_1.bag`
- `lane_change_2.bag`
- `traffic_1.bag`
- `traffic_2.bag` 

통과 판단 기준:
- 차량이 보이는 구간에서 `/car_detection` 과 `/car_projected` 가 생성된다.
- detection이 없을 때 `/car_projected` 는 빈 `PoseArray` 로 유지되고 marker가 정리된다.
- 신호등이 보이는 구간에서 `/traffic` 상태가 갱신되며, red light / else 에 대한 구분이 가능하다.
- RViz 또는 image topic에서 `/yolo_overlay/image`, `/traffic_overlay/image` 확인이 가능하다.

## Parameters


| Source | Name | Value | Meaning |
| :--- | :--- | :--- | :--- |
| `object_detection.yaml` | `image_topic` | `/cam1/usb_cam/image_raw` | 차량 검출용 camera topic |
| `object_detection.yaml` | `model_path` | `/home/vic/kkdws/src/object_detector/model/yolo26s.pt` | 차량 검출 YOLO weight 경로 |
| `object_detection.yaml` | `conf_thres` | `0.1` | 차량 detection confidence threshold |
| `object_detection.yaml` | `device` | `"0"` | YOLO inference device 설정 |
| `object_detection.yaml` | `car_class_name` | `"car"` | publish 대상으로 남길 class 이름 |
| `object_detection.yaml` | `car_topic` | `/car_detection` | 차량 detection 결과 topic |
| code default | `roi/bottom_exclude_ratio` | `0.1` | 이미지 하단 ROI 제외 비율 |
| code default | `pub_rate` | `20.0` | `object_detection.py` publish loop 주기 |
| `object_detection.yaml` | `overlay/topic` | `/yolo_overlay/image` | 차량 overlay image topic |
| `object_projection.yaml` | `sub_car_topic` | `/car_detection` | detection 입력 topic |
| `object_projection.yaml` | `pub_car_projected_topic` | `/car_projected` | projected pose output topic |
| `object_projection.yaml` | `pub_markers_topic` | `/car_projected_markers` | projected marker topic |
| `object_projection.yaml` | `frame_id` | `base_link` | projection 결과 기준 frame |
| `object_projection.yaml` | `camera/fx fy cx cy` | `505.0742 / 531.2114 / 352.6202 / 135.9729` | 640x480 기준 camera intrinsics |
| `object_projection.yaml` | `camera/height` | `0.75` | camera height |
| `object_projection.yaml` | `camera/pitch_deg` | `-14.5` | camera pitch |
| code default | `object_projection pub_rate` | `20.0` | `object_projection.py` publish loop 주기 |
| `traffic_detection.yaml` | `image_topic` | `/cam2/usb_cam/image_raw` | 신호등 검출용 camera topic |
| `traffic_detection.yaml` | `state_topic` | `/traffic` | 신호 상태 topic |
| `traffic_detection.yaml` | `weights` | `/home/vic/kkdws/src/object_detector/model/traffic26_2.pt` | traffic YOLO weight 경로 |
| `traffic_detection.yaml` | `conf_th` | `0.35` | 신호 detection confidence threshold |
| `traffic_detection.yaml` | `class_id_green/red/yellow` | `0 / 1 / 2` | 신호등 class id mapping |
| `traffic_detection.yaml` | `roi_x y w h` | `0.0 / 0.1 / 0.8 / 0.6` | 신호등 관심영역 ROI 비율 |
| code default | `queue_len` | `3` | traffic 결과 smoothing queue 길이 |
| code default | `overlay_topic` | `/traffic_overlay/image` | traffic overlay image topic |


## Limitations / Fault Cases

- `traffic_detection.py` 는 CUDA가 없으면 시작 단계에서 종료된다.
- `object_projection.py` 는 평지 가정의 단순 투영이므로 실제 장애물 거리와 오차가 생길 수 있다.
- `image_saver.py` 는 수집용 유틸리티일 뿐 planner 입력 파이프라인에는 연결되지 않는다.

## Implementation Notes

1. `object_detection.py` 는 YOLO 결과에서 `car` class만 남기고, 하단 ROI 제외 조건을 통과한 bbox만 `/car_detection` 으로 publish 한다.
2. `object_projection.py` 는 bbox 하단 중심 pixel을 지면으로 투영해 `/car_projected` 를 만들고 marker를 함께 관리한다.
3. `traffic_detection.py` 는 ROI 내부 신호등 bbox를 class별로 집계해 상태를 결정하고 `/traffic` 및 overlay를 publish 한다.

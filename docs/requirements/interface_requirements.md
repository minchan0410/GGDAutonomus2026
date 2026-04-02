# Interface Requirements

## 1. 목적
- 본 문서는 현재 요구사항 작업에 필요한 핵심 인터페이스만 간단히 정리한다.
- 우선은 `laser_detector`와 직접 연결되는 인터페이스를 중심으로 작성한다.

## 2. 인터페이스 요구사항

### 2.1 `/scan` -> `laser_detector`
- 설명:
  `laser_detector`는 `/scan` 토픽을 입력으로 받아 parked car detection에 사용해야 한다.
- 형식:
  `sensor_msgs/LaserScan`
- 검증:
  bag replay 후 `/scan` 수신 여부와 `/detection_poses` 생성 여부 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`
- 합격 기준:
  `/scan`이 들어오면 `laser_detector` 출력 토픽이 생성된다.

### 2.2 `laser_detector` -> `parking`
- 설명:
  `laser_detector`는 parking이 사용할 수 있는 parked car 중심 좌표를 `/detection_poses`로 publish해야 한다.
- 형식:
  `geometry_msgs/PoseArray`
- 검증:
  bag replay 후 `/detection_poses`와 parking 쪽 반응 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`
- 합격 기준:
  parking이 사용할 수 있는 pose 정보가 replay 중 확인된다.

### 2.3 `laser_detector` debug interface
- 설명:
  `laser_detector`는 bag replay 분석을 위해 debug 토픽을 제공해야 한다.
- 대상 토픽:
  `/clustered_cloud`, `/detection_markers`, `/detection_poses_viz`
- 검증:
  bag replay 후 debug 토픽 생성 여부 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`
- 합격 기준:
  주요 debug 토픽이 replay 중 확인된다.

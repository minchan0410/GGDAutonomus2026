# Laser Detector Requirements

## 1. 목적
- `laser_detector`는 2D lidar를 이용해 주차 미션용 parked car를 검출하고 tracking하는 유닛이다.
- 이 유닛의 출력은 `parking` 패키지의 주차 판단 입력으로 사용된다.

## 2. 상위 시스템 요구 연계
- 시스템은 주차 구간에서 parked car 정보를 활용해야 한다.
- 시스템은 bag replay로 검증 가능해야 한다.

## 3. 입력

### 3.1 입력 데이터
- `/scan`
  - 형식: `sensor_msgs/LaserScan`
  - 의미: raw lidar scan

### 3.2 입력 조건
- `/scan`이 정상적으로 들어와야 한다.
- ROI 범위를 벗어난 값은 제외한다.

## 4. 출력

### 4.1 출력 데이터
- `/detection_poses`
  - 형식: `geometry_msgs/PoseArray`
  - 의미: parking이 사용할 parked car 중심점
- `/clustered_cloud`
  - 형식: `sensor_msgs/PointCloud2`
  - 의미: clustering 결과 확인용
- `/detection_markers`
  - 형식: `visualization_msgs/MarkerArray`
  - 의미: RViz debug
- `/detection_poses_viz`
  - 형식: `geometry_msgs/PoseArray`
  - 의미: visualization용 pose

### 4.2 출력 조건
- 유효한 객체가 있으면 detection 결과를 publish해야 한다.
- 유효한 객체가 없으면 이전 결과를 stale하게 유지하지 않아야 한다.

## 5. 기능 요구사항

### 5.1 scan 수신 및 ROI filtering
- 설명:
  `laser_detector`는 `/scan`을 수신하고 ROI 밖의 point는 제외해야 한다.
- 검증:
  bag replay 후 `/clustered_cloud` 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`
- 합격 기준:
  ROI 내 point만 기반으로 clustering 결과가 생성된다.

### 5.2 clustering 기반 객체 후보 생성
- 설명:
  `laser_detector`는 유효한 point들로부터 clustering을 수행해 parked car 후보를 만들어야 한다.
- 검증:
  bag replay 후 `/clustered_cloud`, `/detection_markers` 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`
- 합격 기준:
  parked car 위치에서 객체 후보가 확인된다.

### 5.3 tracking 유지
- 설명:
  `laser_detector`는 이전 프레임의 객체를 기반으로 tracking을 유지해야 한다.
- 검증:
  bag replay 후 `/detection_poses`에서 같은 객체의 ID 유지 여부 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`, `bag_parking_lidar_sparse_01`
- 합격 기준:
  같은 parked car가 연속 프레임에서 불필요하게 다른 ID로 바뀌지 않는다.

### 5.4 parked car pose publish
- 설명:
  `laser_detector`는 parking이 사용할 수 있는 parked car 중심 좌표를 `/detection_poses`로 publish해야 한다.
- 검증:
  bag replay 후 `/detection_poses` 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`
- 합격 기준:
  parked car가 존재할 때 `/detection_poses`가 생성된다.

### 5.5 입력 누락 및 미검출 처리
- 설명:
  입력이 끊기거나 유효한 객체가 없을 때 이전 detection 결과를 계속 유지하지 않아야 한다.
- 검증:
  bag replay 후 입력 공백 구간과 출력 상태 확인
- 시험 데이터:
  `bag_system_fallback_01`, `bag_parking_lidar_empty_01`
- 합격 기준:
  입력 이상 또는 미검출 상황에서 stale output이 남지 않는다.

## 6. 제약 및 가정
- 입력 lidar topic은 `/scan`이라고 가정한다.
- parking은 `/detection_poses`를 사용할 수 있어야 한다.
- 현재 검증 기준은 bag replay 중심으로 작성한다.

## 7. 검증에 사용할 bag
- `bag_parking_lidar_basic_01`
  - 기본 parked car detection 확인용
- `bag_parking_lidar_sparse_01`
  - sparse point 상황에서 tracking 확인용
- `bag_parking_lidar_empty_01`
  - 객체가 없을 때 출력 처리 확인용
- `bag_system_fallback_01`
  - 입력 누락 또는 fallback 상황 확인용

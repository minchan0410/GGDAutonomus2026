# System Requirements

## 1. 목적
- 본 문서는 차량 전체 수준에서 필요한 요구사항을 간단히 정리한다.
- 본 문서는 `laser_detector` 같은 유닛 문서의 상위 기준으로 사용한다.

## 2. 시스템 요구사항

### 2.1 차선 기반 주행
- 설명:
  시스템은 차선 정보를 기반으로 자율주행을 수행해야 한다.
- 검증:
  bag replay 중 `/lane_steer`, `/des_steer`, `/motor_cmd_long` 확인
- 시험 데이터:
  `bag_lane_basic_01`
- 합격 기준:
  차선 주행 구간에서 조향 및 속도 명령이 계속 출력된다.

### 2.2 장애물 대응
- 설명:
  시스템은 장애물 정보를 받아 감속, 차선 변경, 정지 중 하나의 동작으로 대응해야 한다.
- 검증:
  bag replay 중 planner 상태와 제어 명령 확인
- 시험 데이터:
  `bag_obstacle_basic_01`
- 합격 기준:
  장애물 이벤트 발생 시 planner 상태 또는 제어 출력이 바뀐다.

### 2.3 주차 기능
- 설명:
  시스템은 주차 구간에서 주차용 인지 결과를 사용해 주차 동작을 수행해야 한다.
- 검증:
  bag replay 중 `/detection_poses`, `/stanley_path`, `/des_steer` 확인
- 시험 데이터:
  `bag_parking_lidar_basic_01`
- 합격 기준:
  주차 구간에서 주차 관련 토픽이 순서대로 생성된다.

### 2.4 안전 동작
- 설명:
  시스템은 필수 입력이 끊기거나 제어 경로가 비정상일 때 계속 진행하지 말고 안전 동작으로 전환해야 한다.
- 검증:
  bag replay 중 입력 누락 상황에서 상태 및 제어 명령 확인
- 시험 데이터:
  `bag_system_fallback_01`
- 합격 기준:
  입력 이상 시 명령이 멈추거나 안전 상태로 전환된다.

### 2.5 재현 가능성
- 설명:
  시스템 요구사항은 bag replay로 다시 확인할 수 있어야 한다.
- 검증:
  같은 bag를 다시 재생해 주요 출력 토픽 확인
- 시험 데이터:
  `bag_lane_basic_01`, `bag_parking_lidar_basic_01`
- 합격 기준:
  같은 replay 절차로 같은 종류의 출력 토픽을 확인할 수 있다.

## 3. 관련 문서
- `docs/requirements/interface_requirements.md`
- `docs/requirements/units/laser_detector.md`
- `docs/test/bag_catalog.md`

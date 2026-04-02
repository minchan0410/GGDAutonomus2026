# Bag Catalog

## bag_parking_lidar_basic_01
- 시나리오:
  주차 구간에서 parked car가 정상적으로 보이는 기본 상황
- 포함 데이터:
  `/scan`
- 사용 대상:
  `laser_detector`, `parking`

## bag_parking_lidar_sparse_01
- 시나리오:
  parked car point가 sparse하게 보이는 상황
- 포함 데이터:
  `/scan`
- 사용 대상:
  `laser_detector`

## bag_parking_lidar_empty_01
- 시나리오:
  ROI 안에 유효한 parked car가 없는 상황
- 포함 데이터:
  `/scan`
- 사용 대상:
  `laser_detector`

## bag_system_fallback_01
- 시나리오:
  입력 누락 또는 topic 공백이 있는 상황
- 포함 데이터:
  `/scan` 또는 관련 상태 토픽
- 사용 대상:
  `laser_detector`

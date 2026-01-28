## object_detector

YOLO 기반 차량/신호등 인식과 차량 위치 투영을 제공하는 ROS 패키지.
데이터 수집용 이미지 클릭 저장 노드도 포함됨.

---

### 구성
```
config/
  object_detection.yaml
  object_projection.yaml
  traffic_detection.yaml
launch/
  object_detector.launch
scripts/
  object_detection.py
  object_projection.py
  traffic_detection.py
  image_saver.py
```

---

### 주요 노드

#### 1) `object_detection.py`
차량만 필터링하는 YOLO 검출 노드.

**Topics**
- Sub: `/cam1/usb_cam/image_raw` (`sensor_msgs/Image`)
- Pub: `/car_detection` (`vision_msgs/Detection2DArray`)
- Pub: `/yolo_overlay/image` (`sensor_msgs/Image`, optional)

**Params (private)**
- `image_topic` : 입력 이미지 토픽
- `model_path` : YOLO 가중치 파일 경로
- `conf_thres` : confidence 임계값
- `device` : CUDA 디바이스 (예: `"0"`)
- `car_class_name` : 유지할 클래스 이름 (기본 `"car"`)
- `car_topic` : 차량 검출 결과 토픽
- `overlay/*` : 오버레이 표시 설정
- `roi/bottom_exclude_ratio` : 하단 ROI 제외 비율
- `pub_rate` : 퍼블리시 주기 (Hz)

---

#### 2) `object_projection.py`
검출된 차량을 지면 좌표계로 투영하여 가장 가까운 차량을 퍼블리시.

**Topics**
- Sub: `/car_detection` (`vision_msgs/Detection2DArray`)
- Pub: `/car_projected` (`geometry_msgs/PointStamped`)
- Pub: `/car_projected_markers` (`visualization_msgs/Marker`, optional)

**Params**
- `sub_car_topic`, `pub_car_projected_topic`, `pub_markers_topic`
- `frame_id` : 기준 프레임 (예: `base_link`)
- `select_nearest` : 가장 가까운 차량만 선택
- `camera/*` : 카메라 파라미터 (fx, fy, cx, cy, height, pitch_deg)
- `marker/*` : RViz 마커 설정

---

#### 3) `traffic_detection.py`
신호등 색상 검출 노드 (CUDA GPU 필수).

**Topics**
- Sub: `/cam2/usb_cam/image_raw` (`sensor_msgs/Image`)
- Pub: `/traffic` (`std_msgs/Int16`)
- Pub: `/traffic_overlay/image` (`sensor_msgs/Image`, optional)

**Params**
- `weights` : YOLO 가중치 파일 경로
- `conf_th`
- `class_id_green/red/yellow`
- `roi_x/roi_y/roi_w/roi_h` : ROI 비율
- `queue_len` : 결과 평활화 길이
- `overlay_enable`, `overlay_topic`

**State Mapping**
- `0`: NONE
- `1`: GREEN
- `2`: RED
- `3`: YELLOW

---

#### 4) `image_saver.py`
마우스 클릭 시 이미지 저장하는 유틸 노드.

**Params**
- `image_topic`
- `compressed`
- `save_dir`
- `prefix`
- `jpeg_quality`

---

### 실행
```
roslaunch object_detector object_detector.launch
```

---

### 설정 파일 안내
- `config/object_detection.yaml` : 차량 검출 설정 및 가중치 경로
- `config/traffic_detection.yaml` : 신호등 검출 설정 및 가중치 경로
- `config/object_projection.yaml` : 카메라 내/외부 파라미터

---

### 의존성
- ROS: `rospy`, `vision_msgs`, `sensor_msgs`, `geometry_msgs`, `visualization_msgs`
- OpenCV / `cv_bridge`
- Ultralytics YOLO
- PyTorch (CUDA 필요: `traffic_detection.py`)

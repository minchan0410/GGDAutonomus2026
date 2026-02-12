# Lane Detection by CV

### 전방 카메라 센서를 통해 차선 인식 및 중심점의 offset을 계산하는 패키지

<br>

- **run2.py**

<div align="center">
  <img src="./assets/run2.gif" alt="run" width="350" style="margin-right: 20px;">
  <img src="./assets/run2_crossline.gif" alt="run2" width="350">
</div>

<br>

- **run.py**
<div align="center">
  <img src="./assets/run2.gif" alt="run" width="350">
</div>

<br>



## Process

### run.py
![process block](./assets/flow_run2.png)
1. asdfasdf
2. asdfasdf
3. asdasdfasdf

### Key Strategies
- asdfasdf
---
### run2.py
![process block](./assets/flow_run2.png)
1. **전처리 :** 수신된 이미지를 GPU로 업로드하여 ROI(관심 영역)를 설정하고, 전처리 및 Canny Edge를 통한 Edge 영역 검출을 통해 차선으로 추정되는 영역을 검출.
2. **직선 검출 :** 전처리된 Edge 영상에 CUDA 기반 허프 변환(Hough Transform)을 적용하여 차선으로 추정되는 직선 성분들을 검출.
3. **직선 필터링**
    - 기울기 기반 차선이 아닌 가로 선 제외.
    - 기울기와 x축 좌표 기반 스코어링을 통해 좌측 차선, 우측 차선으로 예상되는 직선을 분리.
    - 각 차선 리스트에서 x축 좌표 기반 노이즈 제거. (논리적으로 불가능한 영역에 있는 차선 제거)
    - 대부분의 경우에서 좌측 우측 모두 가장 중앙점에 가까운 직선들이 차선임으로 좌측 차선은 x 좌표 상위 n%, 우측 차선은 x 좌표 하위 n%에 해당하는 직선을 추출.
    - 추출한 직선들은 평균을 내어 대표직선을 지정.  
4. 타겟 및 조향 산출: 대표직선과 ROI의 y축 중앙 좌표와의 교점을 왼쪽, 오른쪽 차선의 point로 하고, 그 중심점을 midpoint로 하며 midpoint가 이미지의 중심 픽셀 좌표와 얼마나 떨어져 있는지를 계산한다. 이후 차량의 입력 스티어링 범위 (-22 ~ 22) 내에서 주행할 수 있도록 k배(k < 1) 하여 offset를 목표 조향으로 사용한다. <br> <br>이때 왼쪽 및 오른쪽 point는 정상적인 차선 주행을 가정할 때 일정 범위 이상 붙거나 왼쪽 point가 오른쪽 point를 넘어갈 수 없음으로 이에 대한 제약 조건을 추가로 부여한다.또한 산출된 offset의 high frequency를 제거하기 위해 moving average를 사용한다.


### Key Strategies
- **Using cv2.cuda**<br>
전방 이미지 토픽인 `/cam1/usb_cam/image_raw`는 30fps로 발행된다. 따라서 run2.py는 한 loop에 최대 30ms, 안정적으로는 10ms 이내의 processing 시간을 보장해야 한다. 동일 알고리즘을 gpu를 사용하지 않는 cv2.를 가동하여 사용하였을 때는 computing power의 한계로 처리에 30ms를 넘는 경우가 다수 생겨 steering에 lag가 발생, 정상적인 주행이 어려운 경우가 생겼다. cv2.cuda를 사용하였을 때는 cpu보다 안정적은 fps와 적은 processing time을 보여 주었다.

- **why not deep learning method? (yolo segmentation..)** <br>
실도로 데이터을 사용하여 미리 학습된 모델을 사용하였을 때는 안정적인 성능을 보여주지 못하였기 때문에 융합기술원에서 주행 데이터를 만들고 해당 데이터를 통해 학습을 시켜야 안정적인 성능을 보여줄 것으로 예상되었다. 하지만 가장 베이스가 되는 lane_dection이 늦게 완성되면 다른 알고리즘의 개발 일정과 테스트 일정이 밀리게 되어 딥러닝을 통한 차선 검출 알고리즘은 사용하기 어려웠다.  

- **why Canny Edge?** <br>
HSV를 사용하는 방법도 있지만 이는 다양한 조도와 환경에서 균일하게 적용하기가 매우 어려워 상대적으로 강인한 Edge 탐지 방법을 사용. <br>

## Topics

### run.py
---
### run2.py

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


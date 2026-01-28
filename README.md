# **2026 경기도 대학생 자율주행 경진대회**
<div align="center">
  <img src="./assets/poster.jpg" alt="poster" width="600">
</div>

### Team   : **AJOU NICE**
### Result :  ~ prize


| Name | developement Role (package..) |
| :--- | :--- |
| `변지훈` | `하드웨어 세팅` / `lower_controller` / `pre_final_planner` / `object_detector` |
| `한민규` | aasdfasdf |
| `정민찬` | `lane_detector` / `laser_detector` / `object_detector (traffic)` |
| `구동열` | `parking` / `lower_controller` |
| `전희중` | `lane_detection` / `pre_final_planner` |

<br>

## **Hardware Info** 

<div align="center">
  <img src="./assets/hardwareinfo.png" alt="poster" width="600">
</div>

<br>

## **System Structure**
<div align="center">
  <img src="./assets/아키텍처 다이어그램 ver3.jpg" alt="poster" width="600">
</div>

<br>

## **Workspace Structure**

```text
~ 
```

<br>

## Package Overview

###For mission, system
- **arduino_motor_bridge**
    Arduino ROS serial bridge for motor PWM control + potentiometer/ultrasonic IO.

- **lane_detection**
    Lane detect by OpenCV, additional filtering algorithm (DBSCAN..)
    Crosswalk detect by OpenCV

- **laser_detection**
    Detect parked cars by lidar (parking mission) 

- **lateral_controller**
    Lower-level steering PID: /des_steer + potentiometer -> /motor_cmd_steer.

- **object_detector**
    YOLO car detection + ROI filter, plus 2D->ground projection for nearest car.
    YOLO traffic detection

- **pre_final_planner**
    Pre/Final planner modes: lane following + obstacle/traffic logic (with RViz HUD).

###For sensors
- **rplidar_ros**
    to run rplidar's 2d lidar

- **usb_cam**
    to run logitech c920 cameras

<br>

## How to run
- ~~asdfasd

  asdfasdf
  ```shell
  asdfasdf
  ```
  

<br>

## Table

### Input table
| Name | Type | Uses |
| :--- | :--- | :--- |
| `asdf` | `asdf` | asdf |

<br>

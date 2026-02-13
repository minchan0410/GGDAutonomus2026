# **2026 경기도 대학생 자율주행 경진대회**
<div align="center">
  <img src="./assets/poster.jpg" alt="poster" width="600">
</div>

### Team Name   : **AJOU NICE**
###  Result :  **2nd prize** 🥈 

<br>


| Name | developement Role (package..) |
| :--- | :--- |
| `변지훈` | `하드웨어 세팅` / `lower_controller` / `pre_final_planner` / `object_detector` |
| `한민규` |  `Team leader` /  `Paper Works`  |
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

## Package Overview

### For mission, system
- **arduino_motor_bridge**
    <br>
    Arduino ROS serial bridge for motor PWM control + potentiometer/ultrasonic IO.

- **lane_detection**
    <br>
    Lane detect by OpenCV, additional filtering algorithm
    Crosswalk detect by OpenCV

    <div align="left">
        <img src="./lane_detector/assets/run2.gif" alt="poster" width="400">
    </div>

- **laser_detector**
    <br>
    Detect parked cars by lidar (parking mission)
    <div align="left">
        <img src="./laser_detector/assets/tracking1.gif" alt="poster" width="400">
    </div> 

- **lateral_controller**
    <br>
    Lower-level steering PID: /des_steer + potentiometer -> /motor_cmd_steer.

- **object_detector**
    <br>
    YOLO car detection + ROI filter, plus 2D->ground projection for nearest car.
    YOLO traffic detection

- **pre_final_planner**
    <br>
    Pre/Final planner modes: lane following + obstacle/traffic logic (with RViz HUD).

- **parking**
    <br>
    Parking planner: using lane + laser detection information, FSM based planner

    <div align="left">
        <img src="./parking/assets/parking.gif" alt="poster" width="400">
    </div>
    
### For sensors
- **rplidar_ros**
    <br>
    to run rplidar's 2d lidar

- **usb_cam**
    <br>
    to run logitech c920 cameras

<br>

# Mission Vids

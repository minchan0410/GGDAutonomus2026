# **2026 경기도 대학생 자율주행 경진대회**
<div align="center">
  <img src="./assets/poster.jpg" alt="poster" width="600">
</div>

### Team Name   : **AJOU NICE**
###  Result :  **2nd prize** 🥈 

<br>


| Name | developement Role (package..) |
| :--- | :--- |
| `변지훈` | `Hardware Setting` / `lower_controller` / `pre_final_planner` / `object_detector` |
| `한민규` |  `Team leader` /  `Paper Works`  |
| `정민찬` | `lane_detection` / `laser_detector` / `object_detector (traffic)` |
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
    
    Arduino ROS serial bridge for motor PWM control + potentiometer/ultrasonic IO.
    <br>
- **lane_detection**
    
    Lane detect by OpenCV, additional filtering algorithm
    Crosswalk detect by OpenCV

    <div align="left">
        <img src="./lane_detector/assets/run2.gif" alt="poster" width="400">
    </div>
    <br>

- **laser_detector**

    Detect parked cars by lidar (parking mission)
    <div align="left">
        <img src="./laser_detector/assets/tracking1.gif" alt="poster" width="400">
    </div> 

- **lateral_controller**
    Lower-level steering PID: /des_steer + potentiometer -> /motor_cmd_steer.
    <div align="left">
        <img src="./lateral_controller/assets/lower_controller.gif" alt="poster" width="400">
    </div> 
    <br>

- **object_detector**
    YOLO car detection + ROI filter, plus 2D->ground projection for nearest car.
    YOLO traffic detection
    <br>

- **pre_final_planner**
    Pre/Final planner modes: lane following + obstacle/traffic logic (with RViz HUD).
    <div align="left">
        <img src="./pre_final_planner/assets/final_planner.gif" alt="poster" width="400">
    </div> 
    <br>

- **parking**
    Parking planner: using lane + laser detection information, FSM based planner

    <div align="left">
        <img src="./parking/assets/parking1.gif" alt="poster" width="400">
    </div>
    <br>

### For sensors
- **rplidar_ros**
    to run rplidar's 2d lidar

- **usb_cam**
    to run logitech c920 cameras

<br>

# Mission Vids

<div align="left">
  <img src="./pre_final_planner/assets/final_competition_run.gif" alt="pre_final_planner competition run" width="400">
</div>

<div align="left">
  <img src="./parking/assets/parking2.gif" alt="parking competition run" width="400">
</div>

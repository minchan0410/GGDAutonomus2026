# pre_final_planner
---
This package contains pre_planner / final_planner state machine nodes.

---
### Simulation / Test Video
final_planner (Obstacle avoidance) - faster x2
![final Demo 1](./assets/final_planner.gif)


---

### Competition  Run
![final Demo 2](./assets/final_competition_run.gif)


---



### `pre_planner.py`
Switches **DEFAULT/PRE** modes via keyboard input and outputs simple steering/motor commands.

- INPUT
  - `/lane_steer` (std_msgs/Int16)
- OUTPUT
  - `/des_steer` (std_msgs/Int16)
  - `/motor_cmd_long` (std_msgs/Int16)
- Keyboard
  - `d`: DEFAULT 
  - `p`: PRE 
---

### `final_planner.py`
This is the **final driving logic** including lane following, obstacle avoidance (YOLO), and crossline/signal response.

- INPUT
  - `/lane_steer` (std_msgs/Int16)
  - `/cur_lane` (std_msgs/Int16)
  - `/car_projected` (geometry_msgs/PoseArray)
  - `/traffic` (std_msgs/Int16)
  - `/crossline` (std_msgs/Int16)
- OUTPUT
  - `/des_steer` (std_msgs/Int16)
  - `/motor_cmd_long` (std_msgs/Int16)
  - `/final_planner/state` (std_msgs/String)
  - `/final_planner/yolo_crash` (std_msgs/Bool)
  - `/final_planner/lane_change_reason` (std_msgs/String)
- Keyboard
  - `d`: DEFAULT 
  - `f`: FINAL 
  - `1`: CASE1 (`state = case1`) -Emergency state entery
  - `2`: CASE2 (`state = case2`) -Emergency state entery
  - `3`: CASE3 (`state = crossline`) -Emergency state entery

- State Machine (runtime)
  - `start` 
  - `lane_driving1`
  - `lane_change_to_left` 
  - `lane_driving2` 
  - `lane_change_to_right` 
  - `crossline` 
  - `traffic`
  - On serial loss in `FINAL`, planner enters freeze mode (keeps state, stops command updates) until serial recovers.

![final_planner state machine](./assets/final_planner_state_machine.png.png)

---

### `final_planner_rviz.py`
Visualize status of Final planner in RIVZ

- INPUT
  - `/final_planner/state` (std_msgs/String)
  - `/final_planner/yolo_crash` (std_msgs/Bool)
  - `/final_planner/lane_change_reason` (std_msgs/String)
- OUTPUT
  - `/final_planner/markers` (visualization_msgs/MarkerArray)
  - `/final_planner/hud` (jsk_rviz_plugins/OverlayText)
---

### `lane_bev_rviz.py`
Converts lane pixel coordinates into a BEV (bird's-eye view) OccupancyGrid and displays it in RViz.

- INPUT
  - `/lane_lines_px` (std_msgs/Int32MultiArray)
  - `/lane_target_px` (geometry_msgs/PointStamped)
  - `/lane_steer` (std_msgs/Int16)
- OUTPUT
  - `/lane_bev/grid` (nav_msgs/OccupancyGrid)
  - `/lane_bev/markers` (visualization_msgs/Marker)
---

## How to launch

### Pre_planner
```
roslaunch pre_final_planner pre_planner.launch
```

### Final_planner + RViz
```
roslaunch pre_final_planner final_planner.launch
```


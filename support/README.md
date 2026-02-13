# Tele Operation by keyboard

| key | action |
| :--- | :--- |
| `key UP` | move forward |
| `key DOWN` | move backward |
| `q` | turn left |
| `w` | turn right |

<br>

- Because of PID parameters, the command topics should not change rapidly **(overshoot)**.
<br>
Setting the rising and falling steps is important.

---

# rosserial_checker

`scripts/rosserial_checker.py` monitors rosserial link health.

## Overview

- Subscribes to `/potentiometer` (`std_msgs/Int16`) and tracks last receive time.
- Publishes `/heart_beat` (`std_msgs/Int16`) with value `1` to the Arduino board

- Publishes `/rosserial_check` (`std_msgs/Int16`) as status output to planner nodes.

## `/rosserial_check` Consumers (Planners)

- `pre_final_planner/scripts/pre_planner.py` (`PreFinalPlanner`)
- `pre_final_planner/scripts/final_planner.py` (`FinalPlanner`)
- `parking/scripts/parking.py` (`Parking`)

## Status Logic

- Uses `timeout = 0.5s` as communication watchdog threshold.
- If `/potentiometer` is received within timeout: `rosserial_check = 0` (OK).
- If no recent message is received: `rosserial_check = 1` (ERROR).
- Runs monitor loop at `50Hz` and prints throttled ROS logs.

## Run

```bash
rosrun support rosserial_checker.py
```

# Lateral Controller (Lower-Level Steering PID)

조향용 하위 제어기 패키지. 목표 조향각(`/des_steer`)과 조향 포텐셔미터(`/potentiometer`)를 받아 PID로 모터 PWM(`/motor_cmd_steer`)을 출력합니다.

<br>

## File Structure

```text
lateral_controller/
├── CMakeLists.txt
├── config
│   ├── lower_controller_multiplot.perspective
│   └── lower_controller_multiplot.xml
├── launch
│   ├── lateral_lower_controller.launch
│   └── lateral_lower_controller_test.launch
├── package.xml
├── README.md
└── scripts
    ├── lateral_lower_controller.py
    ├── lower_controller_test.py
    └── lower_controller_test_step.py
```

<br>

## Nodes

### `lateral_lower_controller.py`
- **주기**: 20 Hz 타이머 기반
- **제어**: PID + 미분 저역통과 필터 + anti-windup
- **안전**: 현재 조향각이 범위를 벗어나면 출력 0

### `lower_controller_test.py`
- 사인파 조향 입력 publish

### `lower_controller_test_step.py`
- 사각파(스텝) 조향 입력 publish

<br>

## Topics

### Input Topics
| Name | Type | Uses |
| :--- | :--- | :--- |
| `/des_steer` | `std_msgs/Int16` | 목표 조향각(deg) |
| `/potentiometer` | `std_msgs/Int16` | 조향 포텐셔미터 raw 값 |

### Output Topics
| Name | Type | Uses |
| :--- | :--- | :--- |
| `/motor_cmd_steer` | `std_msgs/Int16` | 조향 모터 PWM |
| `/des_steer_deg` | `std_msgs/Float32` | 목표 조향각(deg) 모니터링 |
| `/cur_steer_deg` | `std_msgs/Float32` | 현재 조향각(deg) 모니터링 |

<br>

## Key Constants (code)

`scripts/lateral_lower_controller.py` 내 상수는 현재 하드코딩되어 있습니다.

| Name | Meaning | Default |
| :--- | :--- | :--- |
| `GT_LEFT_MAX`, `GT_RIGHT_MAX` | 조향 각도 제한(deg) | `+22.5`, `-22.5` |
| `Kp`, `Ki`, `Kd` | PID 게인 | `14.0`, `0.0`, `1.0` |
| `u_max` | PWM 제한 | `255` |
| `MARGIN` | 안전 범위 여유(deg) | `1.5` |
| `POT_LEFT_MAX`, `POT_RIGHT_MAX` | 포텐셔미터 범위 | `576`, `422` |

<br>

## How to Run

### 1) Lower controller
```shell
roslaunch lateral_controller lateral_lower_controller.launch
```

`with_rqt`를 끄면 rqt 멀티플롯이 실행되지 않습니다.
```shell
roslaunch lateral_controller lateral_lower_controller.launch with_rqt:=false
```

### 2) Test input (사인파/스텝)
테스트 노드들은 **/lane_steer** 토픽을 publish 합니다.  
실제 제어 입력인 `/des_steer`로 쓰려면 remap이 필요합니다.

```shell
rosrun lateral_controller lower_controller_test.py /lane_steer:=/des_steer
```

```shell
rosrun lateral_controller lower_controller_test_step.py /lane_steer:=/des_steer
```

또는 `launch/lateral_lower_controller_test.launch`를 수정하여 remap을 추가하세요.

<br>

## Notes
- `/potentiometer`는 0~1023 범위에서 조향각으로 변환됩니다.
- `out_of_range()` 조건을 만족하면 PWM 출력이 0으로 고정됩니다.


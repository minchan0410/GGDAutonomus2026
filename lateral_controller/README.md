# Lateral Controller

### `/des_steer`와 `/potentiometer`를 이용해 steering PWM을 생성하는 하위 조향 PID 패키지

![Lower controller demo](./assets/lower_controller.gif)

RED : `cur_steer_deg`  
BLUE : `desired_steer_deg`  
MINT : `lane_steer`

## Package Role

`lateral_controller` 패키지는 상위 planner 또는 parking 패키지가 생성한 목표 조향각을 실제 steering actuator가 사용할 수 있는 PWM 명령으로 변환하는 하위 제어 패키지이다.  
본 패키지의 핵심 책임은 `/des_steer` 와 `/potentiometer` 를 입력으로 받아 `/motor_cmd_steer` 를 publish 하고, 디버그용 조향 상태 topic을 함께 제공하는 것이다.

## System Boundary

- 입력:
  `/des_steer`, `/potentiometer`
- 주요 출력:
  `/motor_cmd_steer`, `/des_steer_deg`, `/cur_steer_deg`
- 책임 범위:
  목표 조향각 수신, potentiometer 기반 현재 조향각 계산, PID 제어, steering PWM publish
- 책임 범위 아님:
  목표 조향각 생성, 종방향 제어, Arduino 하드웨어 구동, rosserial heartbeat 감시
- 상위/하위 관계:
  입력 : `pre_final_planner`, `parking`, `support/keyboard_control`, `arduino_motor_bridge`
  출력 : `arduino_motor_bridge`, `rqt_multiplot` 및 debug 확인 환경

## 인터페이스

| Direction | Topic | Type | Description | Used by |
| :--- | :--- | :--- | :--- | :--- |
| Input | `/des_steer` | `std_msgs/Int16` | 목표 조향각 command | `lateral_controller` |
| Input | `/potentiometer` | `std_msgs/Int16` | steering potentiometer raw ADC 값 | `lateral_controller`, `support` |
| Output | `/motor_cmd_steer` | `std_msgs/Int16` | steering motor PWM command | `arduino_motor_bridge` |
| Output | `/des_steer_deg` | `std_msgs/Float32` | 목표 조향각 monitor | `rqt_multiplot` / debug |
| Output | `/cur_steer_deg` | `std_msgs/Float32` | 현재 조향각 monitor | `rqt_multiplot` / debug |

## Node

- `lateral_lower_controller.py`
  - `lateral_lower_controller.launch`에서 실행되는 메인 steering controller 노드.
  - 20 Hz timer loop에서 PID를 계산하고 `/motor_cmd_steer` 를 publish.
- `lower_controller_test.py`
  - sine 형태의 steering 입력을 publish 하는 테스트 노드.
  - 기본 출력 topic은 `/lane_steer` 이므로 실제 controller 검증 시 `/des_steer` 로 remap 이 필요.
- `lower_controller_test_step.py`
  - step 형태의 steering 입력을 publish 하는 테스트 노드다.
  - 기본 출력 topic은 `/lane_steer` 이므로 실제 controller 검증 시 `/des_steer` 로 remap 이 필요.

## 요구사항

### 대상 노드: `lateral_lower_controller.py`

#### 기능 요구사항

| 기능 | 설명 | Input | Output |
| :--- | :--- | :--- | :--- |
| 조향각 feedback 수신 및 변환 | `/potentiometer` raw 값을 steering degree로 변환하여 제어 계산에 사용해야 한다 | `/des_steer`, `/potentiometer` | `/cur_steer_deg` |
| PID 기반 steering PWM 계산 및 publish | 20Hz loop에서 목표/현재 조향각 오차로 PID 출력을 계산하고 `/motor_cmd_steer` 를 publish 해야 한다 | `/des_steer`, `/potentiometer` | `/motor_cmd_steer`, `/des_steer_deg` |
| 범위 이탈 시 안전 출력 제한 | 현재 조향각이 허용 범위를 벗어난 경우 `out_of_range()` 조건에 따라 `/motor_cmd_steer` 를 0으로 강제해야 한다 | - | `/motor_cmd_steer` |

#### 비기능 요구사항

| 항목 | 설명 | 기준 |
| :--- | :--- | :--- |
| 제어 주기 | `lateral_lower_controller.py` 는 일정 주기의 제어 루프를 유지하며 조향 제어를 수행해야 한다. | 제어 루프 주기 `20 Hz` |
| 조향 PWM 출력 제한 | steering motor command 는 하드웨어 허용 범위를 넘지 않도록 항상 제한되어야 한다. | `/motor_cmd_steer` 출력 범위 `-255` ~ `255` |

## 검증 bag

- 실행 준비 : 실차 rosserial 환경 준비, 확인 topic 은 `/motor_cmd_steer`, `/des_steer_deg`, `/cur_steer_deg`
- 확인 방법: `rostopic echo`, `rqt_multiplot`, step 또는 sine steering 입력에 대한 응답 확인

통과 판단 기준:
- `/des_steer` 입력 시 `/motor_cmd_steer` 가 생성된다.
- `/des_steer_deg` 와 `/cur_steer_deg` 를 통해 목표/현재 조향각 비교가 가능하다.
- 허용 범위 밖 조건에서 `/motor_cmd_steer` 가 0으로 제한된다.

## Parameters

현재 구현은 YAML parameter가 아니라 `scripts/lateral_lower_controller.py` 의 code constant 기반으로 동작한다.

| Source | Name | Value | Meaning |
| :--- | :--- | :--- | :--- |
| code constant | `POT_LEFT_MAX` | `600` | potentiometer 좌측 끝 보정값 |
| code constant | `POT_RIGHT_MAX` | `445` | potentiometer 우측 끝 보정값 |
| code constant | `POT_TOTAL_RANGE_DEGREE` | `270` | raw ADC 값을 degree로 환산할 때 사용하는 전체 회전 범위 |
| derived | `POT_CENTER` | `(POT_LEFT_MAX + POT_RIGHT_MAX) / 2.0` | steering center 기준값 |
| code constant | `GT_LEFT_MAX` | `22.5` | 허용 좌측 조향각 limit |
| code constant | `GT_RIGHT_MAX` | `-22.5` | 허용 우측 조향각 limit |
| code constant | `Kp` | `14.0` | PID proportional gain |
| code constant | `Ki` | `0.0` | PID integral gain |
| code constant | `Kd` | `1.0` | PID derivative gain |
| code constant | `u_max` | `255` | PWM saturation limit |
| code constant | `MARGIN` | `1.5` | out-of-range 판정 margin |
| code constant | `alpha` | `0.9` | derivative low-pass filter 계수 |
| code constant | `hz` | `20.0` | controller timer loop 주기 |

메모:
- tuning 값은 현재 launch 또는 YAML에서 조정되지 않고 코드에 직접 박혀 있다.
- 테스트 노드는 `/lane_steer` 를 publish 하므로 실제 controller 입력에는 remap 이 필요하다.

## Limitations / Fault Cases

- `/potentiometer` feedback 이 없으면 closed-loop steering 동작 검증이 불가능하다.
- potentiometer calibration 이 실제 차량과 다르면 `/cur_steer_deg` 와 안전 조건이 모두 틀어질 수 있다.
- 현재 구현에는 `/des_steer` timeout 이 없어 입력이 끊겨도 마지막 목표 조향각이 남을 수 있다.


## Implementation Notes

1. `/des_steer` 를 받아 허용 조향각 범위 안으로 clamp 한다.
2. `/potentiometer` raw 값을 `POT_CENTER` 와 `POT_TOTAL_RANGE_DEGREE` 기준으로 현재 조향각 degree로 변환한다.
3. 목표값과 현재값 오차로 PID 출력을 계산하고, derivative 성분에는 low-pass filter를 적용한다.
4. 출력이 saturation 되면 integral 항을 되감는 방식으로 anti-windup 을 적용한다.
5. `out_of_range()` 조건이 참이면 steering PWM을 0으로 강제하고, debug topic을 함께 publish 한다.


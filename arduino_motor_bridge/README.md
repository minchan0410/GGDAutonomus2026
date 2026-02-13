# Arduino Motor Bridge

ROS-serial bridge package for Arduino-based motor and steering control.

## Package Layout

```text
arduino_motor_bridge/
|-- CMakeLists.txt
|-- package.xml
`-- scripts/
    `-- arduino_bridge/
        `-- arduino_bridge.ino
```

## What It Does

`arduino_bridge.ino` handles three things:

- Subscribes to motor command topics from ROS (`/motor_cmd_long`, `/motor_cmd_steer`)
- Drives DC motor outputs (longitudinal + steering) through PWM pins
- Publishes steering potentiometer raw value to ROS (`/potentiometer`)

It also includes a heartbeat watchdog that latches E-STOP when rosserial link health is bad.

## ROS Topics

### Subscribed

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/motor_cmd_long` | `std_msgs/Int16` | Longitudinal motor PWM command (`-255..255`) |
| `/motor_cmd_steer` | `std_msgs/Int16` | Steering motor PWM command (`-255..255`) |
| `/heart_beat` | `std_msgs/Int16` | Link keep-alive signal from checker node |

### Published

| Topic | Type | Description |
| :--- | :--- | :--- |
| `/potentiometer` | `std_msgs/Int16` | Steering potentiometer raw ADC value |

## Watchdog / E-STOP Logic

The bridge starts in E-STOP state and only releases after heartbeat is stable.

- Heartbeat expected rate: `50 Hz`
- Timeout threshold: `4 frames` (`80 ms`)
- Stable threshold to recover from E-STOP: `20 frames` (`400 ms`)

Behavior:

- If heartbeat times out, E-STOP is latched immediately.
- While E-STOP is latched, both motor outputs are forced to `0`.
- After stable heartbeat duration, E-STOP is automatically released.

## Pin Map (from current sketch)

| Signal | Pin |
| :--- | :--- |
| Potentiometer input | `A1` |
| Left motor IN1 / IN2 | `13` / `12` |
| Right motor IN1 / IN2 | `9` / `8` |
| Steering motor IN1 / IN2 | `11` / `10` |

Note: pins are currently hardcoded in `scripts/arduino_bridge/arduino_bridge.ino`.

## Timing

- Main ROS communication: every loop (`nh.spinOnce()`)
- Potentiometer publish period: `20 Hz` (`50 ms`)

## Typical Run Flow

1. Flash `scripts/arduino_bridge/arduino_bridge.ino` to the Arduino board.
2. Start rosserial bridge on PC (for example, `rosrun rosserial_python serial_node.py ...`).
3. Start heartbeat checker node:

```bash
rosrun support rosserial_checker.py
```

4. Start upstream controller nodes that publish:
- `/motor_cmd_long`
- `/motor_cmd_steer`

## Related Component

- `src/support/scripts/rosserial_checker.py`: publishes `/heart_beat` and monitors rosserial link status.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import threading
import rospy
from std_msgs.msg import Int16

class PreFinalPlanner:
    def __init__(self):
        rospy.init_node("pre_final_planner", anonymous=False)

        # ---- params ----
        self.rate_hz = rospy.get_param("~rate_hz", 20)
        self.pre_motor_cmd = int(rospy.get_param("~pre_motor_cmd_long", 255))
        self.default_steer = int(rospy.get_param("~default_steer", 0))
        self.default_motor = int(rospy.get_param("~default_motor_cmd_long", 0))
        self.serial_timeout_sec = 0.5

        # ---- state ----
        self.mode = "DEFAULT"  # "DEFAULT" | "PRE"
        self.last_lane_steer = 0
        self.lane_steer_received = False
        self.lock = threading.Lock()
        self.serial_ok = False
        self.serial_received = False
        self.serial_last_time = None
        self._last_log_serial_ready = None
        self._last_log_mode = None

        # ---- pubs/subs ----
        self.sub_lane = rospy.Subscriber("/lane_steer", Int16, self.cb_lane_steer, queue_size=1)
        self.sub_serial = rospy.Subscriber("/rosserial_check", Int16, self.cb_serial_check, queue_size=1)
        self.pub_des_steer = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.pub_motor = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)

        # ---- keyboard thread ----
        th = threading.Thread(target=self.keyboard_loop, daemon=True)
        th.start()

        self._log(force=True)

    def cb_lane_steer(self, msg: Int16):
        with self.lock:
            self.last_lane_steer = int(msg.data)
            self.lane_steer_received = True

    def cb_serial_check(self, msg: Int16):
        val = int(msg.data)
        with self.lock:
            self.serial_received = True
            self.serial_ok = (val == 0)
            self.serial_last_time = rospy.Time.now()

    def _serial_ready(self) -> bool:
        if not self.serial_received or not self.serial_ok:
            return False
        if self.serial_timeout_sec <= 0.0:
            return True
        if self.serial_last_time is None:
            return False
        return (rospy.Time.now() - self.serial_last_time).to_sec() <= self.serial_timeout_sec

    def _log(self, force=False, throttle=0.5):
        serial_ready = self._serial_ready()
        serial_txt = "SERIAL OK" if serial_ready else "SERIAL ERROR"
        state_txt = self.mode.lower()
        line = f"[PRE_PLANNER] | {serial_txt} |\n State = {state_txt}"

        if not serial_ready:
            rospy.logwarn_throttle(throttle, line)
            self._last_log_serial_ready = serial_ready
            self._last_log_mode = self.mode
            return

        if force or self._last_log_serial_ready != serial_ready or self._last_log_mode != self.mode:
            rospy.loginfo(line)
            self._last_log_serial_ready = serial_ready
            self._last_log_mode = self.mode

    def keyboard_loop(self):
        key_to_mode = {
            "d": "DEFAULT",
            "p": "PRE",
        }

        while not rospy.is_shutdown():
            line = sys.stdin.readline()
            if not line:
                break

            cmd = line.strip().lower()
            if cmd not in key_to_mode:
                self._log()
                continue

            new_mode = key_to_mode[cmd]
            with self.lock:
                if self.mode != new_mode:
                    if new_mode == "PRE" and not self._serial_ready():
                        self._log()
                        continue
                    self.mode = new_mode
                    self._log()


    def run(self):
        r = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            with self.lock:
                mode = self.mode  # "DEFAULT" | "PRE" | "FINAL"
                lane_steer = self.last_lane_steer if self.lane_steer_received else 0

            # ---- outputs by mode ----
            if mode == "DEFAULT":
                self._log()
                des_steer = self.default_steer
                motor_cmd = self.default_motor

            elif mode == "PRE":
                if not self._serial_ready():
                    with self.lock:
                        self.mode = "DEFAULT"
                    self._log()
                    des_steer = self.default_steer
                    motor_cmd = self.default_motor
                else:
                    self._log()
                    # pre: lane_steer ?�스?�루 + 모터 ?�수�???
                    des_steer = lane_steer
                    motor_cmd = self.pre_motor_cmd

            else:
                # safety fallback
                self._log()
                des_steer = 0
                motor_cmd = 0

            self.pub_des_steer.publish(Int16(des_steer))
            self.pub_motor.publish(Int16(motor_cmd))

            r.sleep()


if __name__ == "__main__":
    node = PreFinalPlanner()
    node.run()


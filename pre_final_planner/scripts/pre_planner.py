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

        # ---- state ----
        self.mode = "d"  # 'd'|'p'|'f'
        self.last_lane_steer = 0
        self.lane_steer_received = False
        self.lock = threading.Lock()
        self.serial_ok = False
        self.serial_received = False

        # ---- pubs/subs ----
        self.sub_lane = rospy.Subscriber("/lane_steer", Int16, self.cb_lane_steer, queue_size=1)
        self.sub_serial = rospy.Subscriber("/rosserial_check", Int16, self.cb_serial_check, queue_size=1)
        self.pub_des_steer = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.pub_motor = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)

        # ---- keyboard thread ----
        th = threading.Thread(target=self.keyboard_loop, daemon=True)
        th.start()

        self._log("started. mode=d (default)")
        self._log("Type: d (default) / p (pre)")

    def cb_lane_steer(self, msg: Int16):
        with self.lock:
            self.last_lane_steer = int(msg.data)
            self.lane_steer_received = True

    def cb_serial_check(self, msg: Int16):
        val = int(msg.data)
        with self.lock:
            self.serial_received = True
            self.serial_ok = (val == 0)

    def _log(self, msg: str, throttle=None):
        serial_txt = "Serial OK" if self.serial_ok else "Serial ERR"
        state_txt = self.mode
        line = f"[PRE_FINAL_PLANNER] | {serial_txt:<11} | State = {state_txt:<12} | {msg}"
        if throttle is None:
            rospy.loginfo(line)
        else:
            rospy.loginfo_throttle(throttle, line)

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
                self._log("invalid key inturrupt")
                continue

            new_mode = key_to_mode[cmd]
            with self.lock:
                if self.mode != new_mode:
                    if new_mode == "PRE" and not self.serial_ok:
                        self._log("SERIAL ERROR!!!!!!!!!!!!!!!!!!!")
                        continue
                    self.mode = new_mode
                    self._log(f'set to mode "{new_mode}"')


    def run(self):
        r = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            with self.lock:
                mode = self.mode  # "DEFAULT" | "PRE" | "FINAL"
                lane_steer = self.last_lane_steer if self.lane_steer_received else 0

            # ---- outputs by mode ----
            if mode == "DEFAULT":
                if not self.serial_received or not self.serial_ok:
                    self._log("serial error: keep DEFAULT. reset serial node.", throttle=0.2)
                des_steer = self.default_steer
                motor_cmd = self.default_motor

            elif mode == "PRE":
                # pre: lane_steer 패스스루 + 모터 상수로 힘
                des_steer = lane_steer
                motor_cmd = self.pre_motor_cmd

            else:
                # safety fallback
                des_steer = 0
                motor_cmd = 0

            self.pub_des_steer.publish(Int16(des_steer))
            self.pub_motor.publish(Int16(motor_cmd))

            r.sleep()


if __name__ == "__main__":
    node = PreFinalPlanner()
    node.run()

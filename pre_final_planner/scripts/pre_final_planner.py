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
        self.pre_motor_cmd = int(rospy.get_param("~pre_motor_cmd_long", 80))
        self.default_steer = int(rospy.get_param("~default_steer", 0))
        self.default_motor = int(rospy.get_param("~default_motor_cmd_long", 0))

        # ---- state ----
        self.mode = "d"  # 'd'|'p'|'f'
        self.last_lane_steer = 0
        self.lane_steer_received = False
        self.lock = threading.Lock()

        # ---- pubs/subs ----
        self.sub_lane = rospy.Subscriber("/lane_steer", Int16, self.cb_lane_steer, queue_size=1)
        self.pub_des_steer = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.pub_motor = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)

        # ---- keyboard thread ----
        th = threading.Thread(target=self.keyboard_loop, daemon=True)
        th.start()

        rospy.loginfo("[pre_final_planner] started. mode=d (default)")
        rospy.loginfo("Type: d (default) / p (pre) / f (final)")

    def cb_lane_steer(self, msg: Int16):
        with self.lock:
            self.last_lane_steer = int(msg.data)
            self.lane_steer_received = True

    def keyboard_loop(self):
        key_to_mode = {
            "d": "DEFAULT",
            "p": "PRE",
            "f": "FINAL",
        }

        while not rospy.is_shutdown():
            line = sys.stdin.readline()
            if not line:
                break

            cmd = line.strip().lower()
            if cmd not in key_to_mode:
                rospy.logwarn("invalid key inturrupt")
                continue

            new_mode = key_to_mode[cmd]
            with self.lock:
                if self.mode != new_mode:
                    self.mode = new_mode
                    rospy.loginfo(f'pre_final_planner set to mode "{new_mode}"')


    def run(self):
        r = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                mode = self.mode
                lane_steer = self.last_lane_steer if self.lane_steer_received else 0

            if mode == "d":
                des_steer = self.default_steer
                motor_cmd = self.default_motor

            elif mode == "p":
                des_steer = lane_steer      # pre: lane_steer 패스스루 (일단)
                motor_cmd = self.pre_motor_cmd

            else:  # mode == "f"
                # TODO: final 로직 나중에 구현
                des_steer = 0
                motor_cmd = 0

            self.pub_des_steer.publish(Int16(des_steer))
            self.pub_motor.publish(Int16(motor_cmd))
            r.sleep()

if __name__ == "__main__":
    node = PreFinalPlanner()
    node.run()

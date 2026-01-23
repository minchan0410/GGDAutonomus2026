#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading
from std_msgs.msg import Int16
import numpy as np
from geometry_msgs.msg import PointStamped
from collections import deque

ROI_MIN_X = 0
ROI_MAX_X = 0.5
ROI_MIN_Y = -0.4
ROI_MAX_Y = 0.4

SPEED_0    = 0
SPEED_MID  = 150
SPEED_HIGH = 255

LC_STEER      = 22.5
STEER_TIME    = 1
STRAIGHT_TIME = 1

class FinalPlanner:
    def __init__(self):
        rospy.init_node("final_planner", anonymous=False)

        self.rate_hz = rospy.get_param("~rate_hz", 20)
        self.default_motor = rospy.get_param("~default_motor", 0)
        

        rospy.Subscriber("/lane_steer", Int16, self.lane_steer_callback, queue_size=1)
        rospy.Subscriber("/ultrasonic1", Int16, self.ultrasonic1_callback, queue_size=1)
        rospy.Subscriber("/cur_lane", Int16, self.cur_lane_callback, queue_size=1)
        rospy.Subscriber("/car_projected", PointStamped, self.car_projected_callback, queue_size=1)
        
        rospy.Subscriber("/traffic", Int16, self.traffic_callback, queue_size=1)
        
    
        
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)

        self.lock = threading.Lock()
        th = threading.Thread(target=self.keyboard_listener, daemon=True)
        th.start()
        self.rate = rospy.Rate(self.rate_hz)

        # Parameters (can be overridden via YAML / rosparam)
        self.roi_min_x = rospy.get_param("~roi_min_x", ROI_MIN_X)
        self.roi_max_x = rospy.get_param("~roi_max_x", ROI_MAX_X)
        self.roi_min_y = rospy.get_param("~roi_min_y", ROI_MIN_Y)
        self.roi_max_y = rospy.get_param("~roi_max_y", ROI_MAX_Y)
        self.roi_offset_x = rospy.get_param("~roi_offset_x", 0.74)

        self.SPEED_0 = rospy.get_param("~speed_0", SPEED_0)
        self.SPEED_MID = rospy.get_param("~speed_mid", SPEED_MID)
        self.SPEED_HIGH = rospy.get_param("~speed_high", SPEED_HIGH)

        self.LC_STEER = rospy.get_param("~lc_steer", LC_STEER)
        self.steer_time = rospy.get_param("~steer_time", STEER_TIME)
        self.straight_time = rospy.get_param("~straight_time", STRAIGHT_TIME)

        self.ultrasonic_threshold = rospy.get_param("~ultrasonic_threshold", 300)
        self.queues_maxlen = rospy.get_param("~queues_maxlen", 10)
        self.yolo_count_threshold = rospy.get_param("~yolo_count_threshold", 7)
        self.ultrasonic_count_threshold = rospy.get_param("~ultrasonic_count_threshold", 7)

        self.mode = "DEFAULT"
        self.last_lane_steer = 0
        self.lane_steer_received = False
        self.state = "lane_driving"

        # Queues (bounded by configured maxlen)
        self.ultrasonic_queue = deque(maxlen=self.queues_maxlen)
        self.ultrasonic_crash = False

        self.yolo_queue = deque(maxlen=self.queues_maxlen)
        self.yolo_crash = False

        self.cur_lane = 2

        self.start_time = None
        self.wait_for_traffic = False

        self.traffic_light = 0
        self.traffic_queue = deque(maxlen=self.queues_maxlen)
        self.traffic_red_threshold = rospy.get_param("~traffic_red_threshold", 5)

        self.run()
        
        
    def ultrasonic1_callback(self, msg):
        self.ultrasonic_queue.append(msg.data)

        if len(self.ultrasonic_queue) == self.ultrasonic_queue.maxlen:
            count_over = sum(1 for v in self.ultrasonic_queue if v <= self.ultrasonic_threshold)
            self.ultrasonic_crash = count_over >= self.ultrasonic_count_threshold
            
    
    def car_projected_callback(self, msg: PointStamped):
        
        x = msg.point.x
        y = msg.point.y

        inside = (self.roi_min_x + self.roi_offset_x) <= x <= (self.roi_max_x + self.roi_offset_x) and self.roi_min_y <= y <= self.roi_max_y
        self.yolo_queue.append(inside)

        if len(self.yolo_queue) == self.yolo_queue.maxlen:
            count_over = sum(1 for v in self.yolo_queue if v)
            self.yolo_crash = count_over >= self.yolo_count_threshold
        
    
    def cur_lane_callback(self, msg): self.cur_lane = msg.data


    def lane_steer_callback(self, msg):
        with self.lock:
            self.last_lane_steer = int(msg.data)
            self.lane_steer_received = True
    
    
    def traffic_callback(self, msg):
        val = int(msg.data)
        with self.lock:
            self.traffic_light = val
            if val in (1, 3):
                self.traffic_queue.append(val)


    def keyboard_listener(self):
        while not rospy.is_shutdown():
            key = input().strip().lower()

            with self.lock:
                if key == "d":
                    self.mode = "DEFAULT"
                    rospy.loginfo("-> DEFAULT")

                elif key == "f":
                    self.mode = "FINAL"
                    rospy.loginfo("-> FINAL")

                else:
                    rospy.logwarn("invalid key")


    def run(self):
        while not rospy.is_shutdown():
            with self.lock:
                mode = self.mode
                lane_steer = self.last_lane_steer if self.lane_steer_received else 0

            if mode == "DEFAULT":
                self.drive(0, self.default_motor)

            elif mode == "FINAL":
                
                if self.state == "lane_driving":
                    
                    if self.wait_for_traffic:
                        with self.lock:
                            self.state = "traffic"
                            self.traffic_queue.clear()
                        continue
                        
                    self.drive(lane_steer, self.SPEED_HIGH)
                    
                    if self.ultrasonic_crash or self.yolo_crash:
                        
                        if self.cur_lane == 1:
                            self.state = "lane_change_to_right"
                            
                        elif self.cur_lane == 2:
                            self.state = "lane_change_to_left"
                            
                
                if self.state == "lane_change_to_right":

                    if self.start_time is None:
                        self.start_time = rospy.Time.now()
                        self.lc_step = 0

                    elapsed = (rospy.Time.now() - self.start_time).to_sec()

                    # ===== STEP 0: 우로 꺾기 =====
                    if self.lc_step == 0:
                        self.drive(-self.LC_STEER, self.SPEED_HIGH)

                        if elapsed >= self.steer_time:
                            self.lc_step = 1
                            self.start_time = rospy.Time.now()
                            
                    # ===== STEP 1: 직진 =====
                    elif self.lc_step == 1:
                        self.drive(0, self.SPEED_HIGH)

                        if elapsed >= self.straight_time:
                            self.lc_step = 2
                            self.start_time = rospy.Time.now()
                            
                    # ===== STEP 2: 좌로 꺾기 =====
                    elif self.lc_step == 2:
                        self.drive(self.LC_STEER, self.SPEED_HIGH)

                        if elapsed >= self.steer_time:
                            self.lc_step = 3
                            self.start_time = rospy.Time.now()

                    # ===== STEP 3: 직진 =====
                    elif self.lc_step == 3:
                        self.state = "lane_driving"
                        self.start_time = None
                        self.wait_for_traffic = True
                        

                if self.state == "lane_change_to_left":
                    
                    if self.start_time is None:
                        self.start_time = rospy.Time.now()
                        self.lc_step = 0

                    elapsed = (rospy.Time.now() - self.start_time).to_sec()

                    # ===== STEP 0: 좌로 꺾기 =====
                    if self.lc_step == 0:
                        self.drive(self.LC_STEER, self.SPEED_HIGH)

                        if elapsed >= self.steer_time:
                            self.lc_step = 1
                            self.start_time = rospy.Time.now()
                            
                    # ===== STEP 1: 직진 =====
                    elif self.lc_step == 1:
                        self.drive(0, self.SPEED_HIGH)

                        if elapsed >= self.straight_time:
                            self.lc_step = 2
                            self.start_time = rospy.Time.now()
                            
                    # ===== STEP 2: 우로 꺾기 =====
                    elif self.lc_step == 2:
                        self.drive(-self.LC_STEER, self.SPEED_HIGH)

                        if elapsed >= self.steer_time:
                            self.lc_step = 3
                            self.start_time = rospy.Time.now()

                    # ===== STEP 3: 직진 =====
                    elif self.lc_step == 3:
                        self.state = "lane_driving"
                        self.start_time = None
                
                
                if self.state == "traffic":
                    # If recent traffic signals contain enough red(1) entries, stop.
                    with self.lock:
                        red_count = sum(1 for v in self.traffic_queue if v == 1)

                    if red_count >= self.traffic_red_threshold:
                        self.drive(lane_steer, self.SPEED_0)
                    else:
                        self.drive(lane_steer, self.SPEED_MID)

            self.rate.sleep()

    def drive(self, des_steer, long_cmd):
        self.motor_cmd_steer_pub.publish(Int16(des_steer))
        self.motor_long_pub.publish(Int16(long_cmd))

if __name__ == "__main__":
    try:
        FinalPlanner()
    except rospy.ROSInterruptException:
        pass

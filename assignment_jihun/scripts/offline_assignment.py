#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
from std_msgs.msg import Int16
from geometry_msgs.msg import PoseArray

# ==========================================
# 신호등 상태 상수
# ==========================================
STATE_NONE   = 0
STATE_GREEN  = 1
STATE_RED    = 2
STATE_YELLOW = 3

class TrafficSignalController:
    def __init__(self):
        # anonymous=True를 주어 노드 이름 충돌 방지
        rospy.init_node("traffic_signal_controller", anonymous=True)
        
        # 파라미터 설정 (KaTeX: $v_{high}$, $v_{stop}$)
        self.speed_high = rospy.get_param("~speed_high", -70)
        self.speed_0 = rospy.get_param("~speed_0", 0)
        self.obstacle_distance_threshold = rospy.get_param("~obstacle_distance", 0.4)
        
        self.traffic_light = STATE_NONE
        self.closest_obstacle_distance = float('inf')
        
        # Publisher
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)
        
        # Subscriber
        rospy.Subscriber("/traffic", Int16, self.traffic_callback, queue_size=1)
        rospy.Subscriber("/detection_poses", PoseArray, self.detection_callback, queue_size=1)
        
        rospy.loginfo("[TrafficController] Node initialized and waiting for messages...")

    def traffic_callback(self, msg: Int16):
        self.traffic_light = int(msg.data)
        # 로깅은 콜백에서 너무 자주 발생하지 않도록 주의
    
    def detection_callback(self, msg: PoseArray):
        if not msg.poses:
            self.closest_obstacle_distance = float('inf')
            return
        
        min_distance = float('inf')
        for pose in msg.poses:
            # 거리 계산: $d = \sqrt{x^2 + y^2}$
            distance = math.sqrt(pose.position.x**2 + pose.position.y**2)
            if distance < min_distance:
                min_distance = distance
        self.closest_obstacle_distance = min_distance
            
    def drive(self, des_steer, long_cmd):
        self.motor_cmd_steer_pub.publish(Int16(int(des_steer)))
        self.motor_long_pub.publish(Int16(int(long_cmd)))
    
    def is_obstacle_detected(self):
        return self.closest_obstacle_distance < self.obstacle_distance_threshold
    
    def handle_traffic_signal(self):
        """신호 및 장애물 상태에 따른 로직 제어"""
        
        if self.traffic_light == STATE_RED:
            rospy.loginfo_throttle(2, "[STATUS] RED: STOP")
            self.drive(0, self.speed_0)

        elif self.traffic_light == STATE_GREEN:
            if self.is_obstacle_detected():
                rospy.logwarn_throttle(1, f"[STATUS] GREEN but OBSTACLE ({self.closest_obstacle_distance:.2f}m): STOP")
                self.drive(0, self.speed_0)
            else:
                rospy.loginfo_throttle(2, "[STATUS] GREEN: GO")
                self.drive(0, self.speed_high)

        elif self.traffic_light == STATE_YELLOW:
            rospy.loginfo_throttle(2, "[STATUS] YELLOW: CAUTION GO")
            self.drive(0, self.speed_high)

        else: # STATE_NONE 포함
            # "신호 안들어왔을때" 출력 (로깅으로 대체하여 터미널 도배 방지)
            rospy.loginfo_throttle(2, "[STATUS] NO SIGNAL: DEFAULT GO")
            self.drive(0, self.speed_high)
            if self.is_obstacle_detected():
                self.drive(0, self.speed_0)
    
    def run(self):
        """메인 루프를 __init__ 외부에서 실행"""
        rate = rospy.Rate(10) # $10Hz$
        while not rospy.is_shutdown():
            self.handle_traffic_signal()
            rate.sleep()

if __name__ == "__main__":
    try:
        controller = TrafficSignalController()
        controller.run() # 생성 후 루프 실행
    except rospy.ROSInterruptException:
        pass
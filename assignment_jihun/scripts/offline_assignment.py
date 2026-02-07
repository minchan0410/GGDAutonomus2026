#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
from std_msgs.msg import Int16
from geometry_msgs.msg import PoseArray
from cv_bridge import CvBridge

# ==========================================
# 신호등 상태 상수
# ==========================================
STATE_NONE   = 0
STATE_GREEN  = 1
STATE_RED    = 2
STATE_YELLOW = 3

# ==========================================
# 속도 상수
# ==========================================
SPEED_0    = 0      # 정지
SPEED_HIGH = 100    # 최대 속도

class TrafficSignalController:
    def __init__(self):
        rospy.init_node("traffic_signal_controller", anonymous=False)
        
        # 파라미터 설정
        self.speed_high = rospy.get_param("~speed_high", SPEED_HIGH)
        self.speed_0 = rospy.get_param("~speed_0", SPEED_0)
        
        # 물체 감지 거리 임계값 (단위: m)
        self.obstacle_distance_threshold = rospy.get_param("~obstacle_distance", 0.7)  # 70cm
        
        # 현재 신호 상태
        self.traffic_light = STATE_NONE
        
        # 가장 가까운 물체까지의 거리
        self.closest_obstacle_distance = float('inf')
        
        # Publisher: 모터 제어 명령
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)
        
        # Subscriber: 신호등 신호
        rospy.Subscriber("/traffic", Int16, self.traffic_callback, queue_size=1)
        
        # Subscriber: 탐지된 물체 위치 (detection.cpp 기반)
        rospy.Subscriber("/detection_poses", PoseArray, self.detection_callback, queue_size=1)
        
        rospy.loginfo("[TrafficController] Node initialized")
        
        # 메인 루프
        self.run()
    
    def traffic_callback(self, msg: Int16):
        """신호등 신호 수신"""
        self.traffic_light = int(msg.data)
        signal_name = ["NONE", "GREEN", "RED", "YELLOW"]
        rospy.loginfo(f"[TrafficController] Signal: {signal_name[self.traffic_light]}")
    
    def detection_callback(self, msg: PoseArray):
        """
        detection.cpp에서 발행하는 탐지된 물체 위치 수신
        70cm 이내 물체 감지 시 flag 설정
        """
        if not msg.poses:
            self.closest_obstacle_distance = float('inf')
            return
        
        # 가장 가까운 물체까지의 거리 계산
        min_distance = float('inf')
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            distance = math.sqrt(x*x + y*y)
            if distance < min_distance:
                min_distance = distance
        
        # 거리 업데이트
        self.closest_obstacle_distance = min_distance
            
    def drive(self, des_steer, long_cmd):
        """
        차량 제어 함수 (detection.cpp 기반)
        
        Args:
            des_steer: 조향각 명령 (-90 ~ 90)
            long_cmd: 속도 명령 (0 ~ 255)
        """
        self.motor_cmd_steer_pub.publish(Int16(int(des_steer)))
        self.motor_long_pub.publish(Int16(int(long_cmd)))
    
    def is_obstacle_detected(self):
        """
        물체 감지 판단:
        - detection_poses에서 가장 가까운 물체 < 임계값 → 물체 감지
        """
        return self.closest_obstacle_distance < self.obstacle_distance_threshold
    
    def handle_traffic_signal(self):
        """
        신호등 신호 & detection_poses에 따라 차량 제어
        
        - STATE_RED (2): 정지
        - STATE_GREEN (1): 
            * 70cm 이내 물체 감지 → 정지
            * 물체 없음 → 직진
        - STATE_YELLOW (3): 직진 
        - STATE_NONE (0): 직진
        """
        if self.traffic_light == STATE_RED:
            # 빨강색: 정지
            rospy.loginfo("[TrafficController] RED SIGNAL -> STOP")
            self.drive(0, self.speed_0)
        elif self.traffic_light == STATE_GREEN:
            # 초록색: 물체 감지 확인
            if self.is_obstacle_detected():
                rospy.logwarn(
                    "[TrafficController] GREEN SIGNAL but OBSTACLE at "
                    f"{self.closest_obstacle_distance:.3f}m -> STOP"
                )
                self.drive(0, self.speed_0)
            else:
                rospy.loginfo("[TrafficController] GREEN SIGNAL -> GO")
                self.drive(0, self.speed_high)
        else:
            # 노랑색, 신호 없음: 직진
            if self.traffic_light == STATE_YELLOW:
                rospy.loginfo("[TrafficController] YELLOW SIGNAL -> CAUTION (GO)")
            
            self.drive(0, self.speed_high)
    
    def run(self):
        """메인 루프"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            # 신호등 신호에 따라 차량 제어
            self.handle_traffic_signal()
            
            rate.sleep()

if __name__ == "__main__":
    try:
        controller = TrafficSignalController()
    except rospy.ROSInterruptException:
        pass


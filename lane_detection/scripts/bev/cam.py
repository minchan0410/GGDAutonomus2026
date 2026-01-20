#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import os
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class GridCaptureNode:
    def __init__(self):
        # 1. 노드 초기화
        rospy.init_node('grid_capture_node', anonymous=False)
        
        # 2. 설정 파라미터 (필요한 경우 토픽 이름을 변경하세요)
        self.camera_topic = "/usb_cam/image_raw"
        
        # 3. 변수 초기화
        self.bridge = CvBridge()
        self.cv_image = None      # 수신된 최신 프레임을 저장할 변수
        self.image_received = False
        
        # 4. Subscriber 설정
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        
        print(f"[{self.camera_topic}] 토픽을 기다리는 중입니다...")

    def image_callback(self, msg):
        """ROS 이미지 메시지를 받아서 OpenCV 형식으로 변환하여 저장"""
        try:
            # ROS Image -> OpenCV Image 변환 (bgr8 포맷)
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_received = True
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def run(self):
        """메인 루프: 이미지를 화면에 띄우고 키 입력을 처리"""
        
        # 저장 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, 'checkerboard.jpg')

        print("--- 정밀 사진 촬영 모드 (ROS) ---")
        print("1. 화면의 얇은 격자선을 참고하여 체커보드를 정렬하세요.")
        print("2. 's' 키를 누르면 선이 없는 깨끗한 원본 사진이 저장됩니다.")
        print("3. 'q'를 누르면 종료합니다.")
        
        rate = rospy.Rate(30) # 30Hz 루프

        while not rospy.is_shutdown():
            # 이미지가 아직 수신되지 않았으면 대기
            if not self.image_received or self.cv_image is None:
                rate.sleep()
                continue

            # [중요] 최신 프레임 복사 (Thread safety를 위해 복사해서 사용)
            frame = self.cv_image.copy()
            
            # 저장용 깨끗한 원본
            clean_frame = frame.copy()
            
            # 출력용 이미지 (가이드라인 그리기용)
            display_frame = frame.copy()
            height, width = display_frame.shape[:2]

            # --- 격자 그리기 로직 (기존 코드와 동일) ---
            line_color = (0, 255, 0) # 초록색
            thickness = 1            # 얇게

            # 1. 세로 중앙선
            center_x = width // 2
            cv2.line(display_frame, (center_x, 0), (center_x, height), line_color, thickness)

            # 2. 가로선 (4등분)
            for i in range(1, 4):
                line_y = (height // 4) * i
                cv2.line(display_frame, (0, line_y), (width, line_y), line_color, thickness)

            # 3. 보조 세로선
            for i in range(1, 4):
                line_x = (width // 4) * i
                if line_x == center_x: continue
                cv2.line(display_frame, (line_x, 0), (line_x, height), (100, 150, 0), thickness)
            # -------------------------------------------

            # 화면 출력
            cv2.imshow('Calibration Guide (ROS)', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # 저장
                cv2.imwrite(save_path, clean_frame)
                print(f"\n[성공] 사진이 '{save_path}'에 저장되었습니다.")
                rospy.signal_shutdown("Image Saved") # 노드 종료
                break
            
            elif key == ord('q'):
                print("\n촬영을 취소합니다.")
                rospy.signal_shutdown("User Quit")
                break
            
            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        node = GridCaptureNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
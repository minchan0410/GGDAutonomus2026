#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import Image  # 카메라 토픽용 메시지 타입
from cv_bridge import CvBridge, CvBridgeError # ROS 이미지 <-> OpenCV 변환
import cv2
import numpy as np
import os
import math
from sklearn.cluster import DBSCAN
import sys
import rospkg

class LaneDetector:
    def __init__(self):
        rospy.init_node('lane_detector_node', anonymous=False)
        
        # Params
        self.cam_mode = rospy.get_param("~cam_mode", False)
        self.camera_topic = rospy.get_param("~camera_topic", "/usb_cam/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "lane_deg")
        
        self.height_usage_ratio = rospy.get_param("~height_usage_ratio", 0.4)
        self.bottom_shrink_ratio = rospy.get_param("~bottom_shrink_ratio", 0.68)
        self.dbscan_eps = rospy.get_param("~dbscan_eps", 15)
        video_name = rospy.get_param("~video_file_name", "curv.mp4")
        
        self.cluster_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        self.bridge = CvBridge() # CvBridge 객체 생성

        # Publisher
        self.angle_pub = rospy.Publisher(self.output_topic, Int16, queue_size=10)

        # Init by mode
        if self.cam_mode:
            rospy.loginfo(f"Camera Mode: ON. Subscribing to {self.camera_topic}")
            self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        else:
            rospy.loginfo("Camera Mode: OFF. Using Video File.")
            # 파일 경로 설정 (기존 로직)
            try:

                script_dir = os.path.dirname(os.path.abspath(__file__))
                package_dir = os.path.dirname(script_dir)
                src_dir = os.path.dirname(package_dir)
                self.video_path = os.path.join(src_dir, "runnig_data", video_name)

                if not os.path.exists(self.video_path):
                    raise FileNotFoundError(f"경로에 파일이 없습니다: {self.video_path}")
                
                self.video_source = self.video_path
                rospy.loginfo(f"Video Path : {self.video_path}")           

            except Exception as e:
                rospy.logerr(f"[Error] Cannot Find Package or Video: {e}")
                sys.exit(1)
            
            self.cap = cv2.VideoCapture(self.video_path)

    def calculate_weighted_average_angle(self, data_list):
        if not data_list: return None
        total_weighted_angle = 0.0
        total_weight = 0.0
        for angle, weight in data_list:
            w = weight 
            total_weighted_angle += angle * w
            total_weight += w
        if total_weight == 0: return 0.0
        return total_weighted_angle / total_weight

    # --- [핵심] 카메라 콜백 함수 ---
    def image_callback(self, msg):
        try:
            # ROS Image 메시지를 OpenCV(bgr8) 포맷으로 변환
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frame(frame)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    # --- [핵심] 이미지 처리 공통 함수 (기존 run 루프 내부 로직) ---
    def process_frame(self, frame):
        # 1. 전처리 준비
        w = frame.shape[1]
        h = frame.shape[0]
        roi_h = int(h * self.height_usage_ratio)
        start_y = h - roi_h
        
        src_pts = np.float32([[0, 0], [w, 0], [w, roi_h], [0, roi_h]])
        shrink_pixel = int(w * self.bottom_shrink_ratio / 2)
        dst_pts = np.float32([[0, 0], [w, 0], [w - shrink_pixel, roi_h], [shrink_pixel, roi_h]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 2. BEV 변환 및 엣지 검출
        roi_img = frame[start_y:h, 0:w]
        bev_img = cv2.warpPerspective(roi_img, M, (w, roi_h), flags=cv2.INTER_LINEAR)
        
        gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 5)
        edges = cv2.Canny(blur, 50, 150)
        
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=20)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        bev_viz = bev_img.copy()
        
        largest_cluster_data = [] 
        
        # 3. 라인 분석 및 DBSCAN
        if lines is not None:
            data_angles = []
            valid_indices = []

            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                if y1 > y2: x1, y1, x2, y2 = x2, y2, x1, y1
                
                dx = x1 - x2
                dy = y2 - y1 
                if dy == 0: angle = 90.0 if dx > 0 else -90.0
                else: angle = math.degrees(math.atan2(dx, dy))

                if abs(angle) > 80: continue
                data_angles.append(angle)
                valid_indices.append(i)

            num_samples = len(data_angles)
            if num_samples > 0:
                angles_np = np.array(data_angles)
                angle_diff_matrix = np.abs(angles_np[:, None] - angles_np[None, :])
                
                db = DBSCAN(eps=self.dbscan_eps, min_samples=3, metric='precomputed').fit(angle_diff_matrix)
                labels = db.labels_
                
                unique_labels = set(labels)
                if -1 in unique_labels: unique_labels.remove(-1)
                
                if len(unique_labels) > 0:
                    label_counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
                    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
                    rank_map = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

                    for idx, label in enumerate(labels):
                        line_idx = valid_indices[idx]
                        x1, y1, x2, y2 = lines[line_idx][0]
                        angle = data_angles[idx]

                        if label == -1:
                            cv2.line(bev_viz, (x1, y1), (x2, y2), (100, 100, 100), 1)
                        else:
                            rank = rank_map[label]
                            color = self.cluster_colors[rank % len(self.cluster_colors)]
                            thickness = 3 if rank == 0 else 1
                            cv2.line(edges_color, (x1, y1), (x2, y2), color, 2)
                            cv2.line(bev_viz, (x1, y1), (x2, y2), color, thickness)
                            if rank == 0:
                                mid_y = (y1 + y2) / 2.0
                                largest_cluster_data.append((angle, mid_y))

        # 4. 결과 발행 및 시각화
        weighted_avg_angle = self.calculate_weighted_average_angle(largest_cluster_data)
        current_avg_angle = weighted_avg_angle if weighted_avg_angle is not None else 0.0

        angle_msg = Int16()
        angle_msg.data = int(current_avg_angle)
        self.angle_pub.publish(angle_msg)

        # 화살표 그리기
        arrow_start_pt = (w // 2, roi_h - 50)
        arrow_len = 100
        angle_rad = math.radians(current_avg_angle)
        arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
        
        cv2.arrowedLine(edges_color, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.arrowedLine(bev_viz, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(bev_viz, f"Avg Angle: {int(current_avg_angle)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        combined_result = cv2.hconcat([edges_color, bev_viz])
        cv2.imshow("Lane Detection Result", combined_result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.loginfo("Quit..")
            rospy.signal_shutdown("User pressed q")

    def run(self):
        if self.cam_mode:
            # 카메라 모드: 콜백만 기다리면 되므로 spin() 사용
            rospy.spin()
        else:
            # 파일 모드: 직접 루프를 돌며 파일 읽기
            rate = rospy.Rate(30)
            while not rospy.is_shutdown() and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                self.process_frame(frame)
                rate.sleep()
            
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        lane_detector = LaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass
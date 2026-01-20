#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import os
import sys
import math
import time
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError # ROS 이미지 <-> OpenCV 변환
from std_msgs.msg import Int16
from sensor_msgs.msg import Image  # 카메라 토픽용 메시지 타입
from sklearn.cluster import DBSCAN

class LaneDetector:
    def __init__(self):
        rospy.init_node('lane_detector', anonymous=False)
        
        # Params
        self.cam_mode = rospy.get_param("~cam_mode", False)
        self.camera_topic = rospy.get_param("~camera_topic", "/usb_cam/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "lane_steer")
        
        self.height_usage_ratio = rospy.get_param("~height_usage_ratio", 0.4)
        self.bottom_shrink_ratio = rospy.get_param("~bottom_shrink_ratio", 0.68)
        self.dbscan_eps = rospy.get_param("~dbscan_eps", 15)
        video_name = rospy.get_param("~video_file_name", "curv.mp4")
        
        self.cluster_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        self.bridge = CvBridge() # CvBridge 객체 생성

        # --- [추가] BEV Matrix (.npy) 로드 ---
        self.is_matrix_loaded = False
        self.H = None
        self.bev_size = None

        # 현재 파일의 위치(bev 폴더)를 기준으로 npy 파일 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        matrix_path = os.path.join(current_dir, 'bev/bev_matrix.npy')
        size_path = os.path.join(current_dir, 'bev/bev_size.npy')

        if self.cam_mode:
            if os.path.exists(matrix_path) and os.path.exists(size_path):
                try:
                    self.H = np.load(matrix_path)
                    self.bev_size = np.load(size_path) # [width, height]
                    self.is_matrix_loaded = True
                    rospy.loginfo(f"BEV Matrix Loaded: {matrix_path}")
                    rospy.loginfo(f"Target Size: {self.bev_size}")
                except Exception as e:
                    rospy.logwarn(f"Failed to load matrix files: {e}")
            else:
                rospy.logwarn(f"NPY files not found in {current_dir}. Using default hardcoded transform.")
        # -------------------------------------

        # Publisher
        self.angle_pub = rospy.Publisher(self.output_topic, Int16, queue_size=10)

        # Init by mode
        if self.cam_mode:
            rospy.loginfo(f"Camera Mode: ON. Subscribing to {self.camera_topic}")
            self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        else:
            rospy.loginfo("Camera Mode: OFF. Using Video File.")
            try:
                # 패키지 경로 탐색 (단순화된 경로 탐색 로직 사용 권장)
                # 여기서는 기존 로직 유지
                script_dir = os.path.dirname(os.path.abspath(__file__))
                package_dir = os.path.dirname(script_dir) # scripts
                src_dir = os.path.dirname(package_dir)    # package root
                # running_data 폴더 위치에 맞춰 수정 필요할 수 있음
                self.video_path = os.path.join(src_dir, "runnig_data", video_name)

                # 만약 위 경로 로직이 복잡하다면 절대 경로를 확인해보세요.
                if not os.path.exists(self.video_path):
                     # fallback: 현재 스크립트 기준 상위 폴더 등 탐색
                     self.video_path = os.path.join(package_dir, "runnig_data", video_name)

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

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            start_t = time.time()          
            self.process_frame(frame)
            end_t = time.time()
            diff = end_t - start_t
            rospy.loginfo(f"Process Time: {diff * 1000:.2f} ms")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def process_frame(self, frame):
        w = frame.shape[1]
        h = frame.shape[0]

        # ---------------------------------------------------------
        # 1. BEV 변환 (NPY 파일 사용 vs 기존 하드코딩 방식 분기)
        # ---------------------------------------------------------
        if self.cam_mode and self.is_matrix_loaded:
            # [방법 A] NPY 매트릭스 사용 (Full FOV)
            # 저장된 크기(new_w, new_h)로 전체 변환
            target_w, target_h = int(self.bev_size[0]), int(self.bev_size[1])
            bev_img = cv2.warpPerspective(frame, self.H, (target_w, target_h), flags=cv2.INTER_LINEAR)
            
            # 후처리를 위한 높이 정보 갱신
            roi_h = target_h 
            roi_w = target_w
        else:
            # [방법 B] 기존 하드코딩 방식 (Video 모드거나 파일 없을 때)
            roi_h = int(h * self.height_usage_ratio)
            start_y = h - roi_h
            
            src_pts = np.float32([[0, 0], [w, 0], [w, roi_h], [0, roi_h]])
            shrink_pixel = int(w * self.bottom_shrink_ratio / 2)
            dst_pts = np.float32([[0, 0], [w, 0], [w - shrink_pixel, roi_h], [shrink_pixel, roi_h]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            roi_img = frame[start_y:h, 0:w]
            bev_img = cv2.warpPerspective(roi_img, M, (w, roi_h), flags=cv2.INTER_LINEAR)
            roi_w = w

        # ---------------------------------------------------------
        # 2. 이미지 처리 (엣지 검출)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 5)
        edges = cv2.Canny(blur, 50, 150)
        
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=20)
        
        # 시각화용 이미지
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        bev_viz = bev_img.copy()
        
        largest_cluster_data = [] 
        
        # ---------------------------------------------------------
        # 3. 라인 분석 및 DBSCAN
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 4. 결과 발행 및 시각화
        # ---------------------------------------------------------
        weighted_avg_angle = self.calculate_weighted_average_angle(largest_cluster_data)
        current_avg_angle = weighted_avg_angle if weighted_avg_angle is not None else 0.0

        angle_msg = Int16()
        angle_msg.data = int(current_avg_angle)
        self.angle_pub.publish(angle_msg)

        # 화살표 그리기 (bev 높이에 맞춰 하단 중앙 위치 조정)
        arrow_start_pt = (roi_w // 2, roi_h - 50) 
        arrow_len = 100
        angle_rad = math.radians(current_avg_angle)
        arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
        
        cv2.arrowedLine(edges_color, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.arrowedLine(bev_viz, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(bev_viz, f"Avg Angle: {int(current_avg_angle)}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        combined_result = cv2.hconcat([edges_color, bev_viz])
        
        # [중요] Full FOV 이미지는 너무 클 수 있으므로 화면 출력 시 리사이즈
        if combined_result.shape[1] > 1920 or combined_result.shape[0] > 1080:
             display_img = cv2.resize(combined_result, (1280, int(1280 * combined_result.shape[0] / combined_result.shape[1])))
        else:
             display_img = combined_result

        cv2.imshow("Lane Detection Result", display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.loginfo("Quit..")
            rospy.signal_shutdown("User pressed q")

    def run(self):
        if self.cam_mode:
            rospy.spin()
        else:
            rate = rospy.Rate(30)
            while not rospy.is_shutdown() and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                start_t = time.time()          
                self.process_frame(frame)
                end_t = time.time()
                diff = end_t - start_t
                rospy.loginfo(f"Process Time: {diff * 1000:.2f} ms")

                rate.sleep()
            
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        lane_detector = LaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass
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
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int16
from sensor_msgs.msg import Image 
from sklearn.cluster import DBSCAN

class LaneDetector:
    def __init__(self):
        rospy.init_node('lane_detector', anonymous=False)
        
        # [설정] 무조건 토픽 구독 모드로 고정
        self.camera_topic = rospy.get_param("~camera_topic", "/cam1/usb_cam/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "/parking_lane_steer")
        
        # [기존 파라미터]
        self.height_usage_ratio = rospy.get_param("~height_usage_ratio", 0.4)
        self.bottom_shrink_ratio = rospy.get_param("~bottom_shrink_ratio", 0.68)
        self.dbscan_eps = rospy.get_param("~dbscan_eps", 15)
        
        # [추가] 최소 선 검출 개수 설정 (이 값보다 적으면 조향각 0)
        self.min_line_count = int(rospy.get_param("~min_line_count", 15))
        
        # [추가] ROI 설정 (BEV 이미지 기준 비율 0.0 ~ 1.0)
        self.roi_x_ratio = rospy.get_param("~roi_x_ratio", 0.0)
        self.roi_y_ratio = rospy.get_param("~roi_y_ratio", 0.0) 
        self.roi_w_ratio = rospy.get_param("~roi_w_ratio", 0.5)
        self.roi_h_ratio = rospy.get_param("~roi_h_ratio", 1)

        self.cluster_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        self.bridge = CvBridge()

        # BEV Matrix 로드
        self.is_matrix_loaded = False
        self.H = None
        self.bev_size = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        matrix_path = os.path.join(current_dir, 'bev/bev_matrix.npy')
        size_path = os.path.join(current_dir, 'bev/bev_size.npy')

        if os.path.exists(matrix_path) and os.path.exists(size_path):
            try:
                self.H = np.load(matrix_path)
                self.bev_size = np.load(size_path) 
                self.is_matrix_loaded = True
                rospy.loginfo(f"BEV Matrix Loaded: {matrix_path}")
            except Exception as e:
                rospy.logwarn(f"Failed to load matrix files: {e}")
        else:
            rospy.logwarn(f"NPY files not found. Using default hardcoded transform.")

        self.angle_pub = rospy.Publisher(self.output_topic, Int16, queue_size=10)
        rospy.loginfo(f"Subscribing to Camera Topic: {self.camera_topic}")
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback)

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
            self.process_frame(frame)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def process_frame(self, frame):
        w = frame.shape[1]
        h = frame.shape[0]

        # ---------------------------------------------------------
        # 1. BEV 변환 (전체 이미지)
        # ---------------------------------------------------------
        if self.is_matrix_loaded:
            target_w, target_h = int(self.bev_size[0]), int(self.bev_size[1])
            bev_img = cv2.warpPerspective(frame, self.H, (target_w, target_h), flags=cv2.INTER_LINEAR)
            bev_h, bev_w = target_h, target_w
        else:
            roi_h = int(h * self.height_usage_ratio)
            start_y = h - roi_h
            src_pts = np.float32([[0, 0], [w, 0], [w, roi_h], [0, roi_h]])
            shrink_pixel = int(w * self.bottom_shrink_ratio / 2)
            dst_pts = np.float32([[0, 0], [w, 0], [w - shrink_pixel, roi_h], [shrink_pixel, roi_h]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            roi_img = frame[start_y:h, 0:w]
            bev_img = cv2.warpPerspective(roi_img, M, (w, roi_h), flags=cv2.INTER_LINEAR)
            bev_h, bev_w = roi_h, w

        # 시각화용 이미지 복사
        bev_viz = bev_img.copy()

        # ---------------------------------------------------------
        # 2. ROI 계산 및 Crop
        # ---------------------------------------------------------
        roi_x = int(bev_w * self.roi_x_ratio)
        roi_y = int(bev_h * self.roi_y_ratio)
        roi_w = int(bev_w * self.roi_w_ratio)
        roi_h = int(bev_h * self.roi_h_ratio)

        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(bev_w - roi_x, roi_w)
        roi_h = min(bev_h - roi_y, roi_h)

        cv2.rectangle(bev_viz, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        cv2.putText(bev_viz, "ROI Area", (roi_x + 5, roi_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if roi_w > 0 and roi_h > 0:
            processing_img = bev_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        else:
            processing_img = bev_img

        # ---------------------------------------------------------
        # 3. 이미지 처리 (엣지 검출)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(processing_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        edges = cv2.Canny(blur, 30, 100)
        
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=70, maxLineGap=20)
        
        edges_color_roi = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_color = np.zeros_like(bev_img)
        if roi_w > 0 and roi_h > 0:
            edges_color[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = edges_color_roi

        largest_cluster_data = [] 
        
        # ---------------------------------------------------------
        # 4. 라인 분석 및 DBSCAN
        # ---------------------------------------------------------
        if lines is not None:
            data_angles = []
            valid_indices = []
            global_lines = [] 

            for i, line in enumerate(lines):
                lx1, ly1, lx2, ly2 = line[0]
                
                # 좌표 복원
                gx1, gy1 = lx1 + roi_x, ly1 + roi_y
                gx2, gy2 = lx2 + roi_x, ly2 + roi_y
                
                if gy1 > gy2: gx1, gy1, gx2, gy2 = gx2, gy2, gx1, gy1
                
                global_lines.append([gx1, gy1, gx2, gy2]) 

                dx = gx1 - gx2
                dy = gy2 - gy1 
                if dy == 0: angle = 90.0 if dx > 0 else -90.0
                else: angle = math.degrees(math.atan2(dx, dy))

                if abs(angle) > 60: continue
                
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
                        gx1, gy1, gx2, gy2 = global_lines[valid_indices[idx]]
                        angle = data_angles[idx]

                        if label == -1:
                            cv2.line(bev_viz, (gx1, gy1), (gx2, gy2), (100, 100, 100), 1)
                        else:
                            rank = rank_map[label]
                            color = self.cluster_colors[rank % len(self.cluster_colors)]
                            thickness = 3 if rank == 0 else 1
                            
                            cv2.line(edges_color, (gx1, gy1), (gx2, gy2), color, 2)
                            cv2.line(bev_viz, (gx1, gy1), (gx2, gy2), color, thickness)
                            
                            if rank == 0:
                                mid_y = (gy1 + gy2) / 2.0
                                largest_cluster_data.append((angle, mid_y))

        # ---------------------------------------------------------
        # 5. 결과 발행 및 시각화 (개수 표시 추가)
        # ---------------------------------------------------------
        detected_count = len(largest_cluster_data)
        
        # [조건 체크] 감지된 선의 개수가 기준값 미만인가?
        is_lines_enough = (detected_count >= self.min_line_count)

        if not is_lines_enough:
            # 조건 불만족 -> Steer 0
            current_avg_angle = 0.0
            rospy.logdebug(f"Not enough lines detected: {detected_count}/{self.min_line_count}")
            
            # 시각화 설정 (빨간색)
            info_color = (0, 0, 255)
            status_text = "Mode: FORCE ZERO"
            arrow_color = (150, 150, 150) # 화살표 회색 처리
        else:
            # 조건 만족 -> 정상 계산
            weighted_avg_angle = self.calculate_weighted_average_angle(largest_cluster_data)
            current_avg_angle = weighted_avg_angle if weighted_avg_angle is not None else 0.0
            
            # 시각화 설정 (초록색/노란색)
            info_color = (0, 255, 0)
            status_text = "Mode: ACTIVE"
            arrow_color = (0, 255, 255)

        # ---------------------------
        # 정보 텍스트 그리기 (좌측 상단)
        # ---------------------------
        # 1. 감지된 개수 / 최소 기준
        count_text = f"Lines: {detected_count} (Min: {self.min_line_count})"
        cv2.putText(bev_viz, count_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
        
        # 2. 현재 모드 (ACTIVE vs FORCE ZERO)
        cv2.putText(bev_viz, status_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)

        # 3. 조향각 값
        cv2.putText(bev_viz, f"Avg Angle: {int(current_avg_angle)}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, arrow_color, 2)


        # 토픽 발행
        angle_msg = Int16()
        angle_msg.data = - int(current_avg_angle * 0.7)
        self.angle_pub.publish(angle_msg)

        # 화살표 그리기
        arrow_start_pt = (bev_w // 2, bev_h - 50) 
        arrow_len = 100
        angle_rad = math.radians(current_avg_angle)
        arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
        
        cv2.arrowedLine(edges_color, arrow_start_pt, (arrow_end_x, arrow_end_y), arrow_color, 3, tipLength=0.3)
        cv2.arrowedLine(bev_viz, arrow_start_pt, (arrow_end_x, arrow_end_y), arrow_color, 3, tipLength=0.3)
        
        combined_result = cv2.hconcat([edges_color, bev_viz])
        
        if combined_result.shape[1] > 1920 or combined_result.shape[0] > 1080:
             display_img = cv2.resize(combined_result, (1280, int(1280 * combined_result.shape[0] / combined_result.shape[1])))
        else:
             display_img = combined_result

        cv2.imshow("Lane Detection Result", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            rospy.signal_shutdown("User pressed q")

    def run(self):
        rospy.spin()
    
if __name__ == "__main__":
    try:
        lane_detector = LaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass
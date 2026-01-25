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
        
        # ==========================================
        # [추가] ROI 설정 (BEV 이미지 기준 비율 0.0 ~ 1.0)
        # ==========================================
        # 예: x=0.0, y=0.0, w=1.0, h=1.0 이면 전체 화면 사용
        # 예: x=0.2, y=0.4, w=0.6, h=0.6 이면 중앙 하단부 집중
        self.roi_x_ratio = rospy.get_param("~roi_x_ratio", 0.0)
        self.roi_y_ratio = rospy.get_param("~roi_y_ratio", 0.0) 
        self.roi_w_ratio = rospy.get_param("~roi_w_ratio", 0.5)
        self.roi_h_ratio = rospy.get_param("~roi_h_ratio", 0.5)

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
        # [추가] 2. ROI 계산 및 Crop
        # ---------------------------------------------------------
        # BEV 이미지 상에서의 ROI 좌표 계산
        roi_x = int(bev_w * self.roi_x_ratio)
        roi_y = int(bev_h * self.roi_y_ratio)
        roi_w = int(bev_w * self.roi_w_ratio)
        roi_h = int(bev_h * self.roi_h_ratio)

        # 예외 처리: 이미지 범위를 벗어나지 않도록 클램핑
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(bev_w - roi_x, roi_w)
        roi_h = min(bev_h - roi_y, roi_h)

        # ROI 영역 시각화 (파란색 박스)
        cv2.rectangle(bev_viz, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        cv2.putText(bev_viz, "ROI Area", (roi_x + 5, roi_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 실제 알고리즘에 들어갈 이미지 Crop
        # (ROI 영역만 잘라내어 처리 속도를 높이고 노이즈를 줄임)
        if roi_w > 0 and roi_h > 0:
            processing_img = bev_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        else:
            processing_img = bev_img # 설정 오류 시 전체 사용

        # ---------------------------------------------------------
        # 3. 이미지 처리 (엣지 검출 - ROI 내부에서 수행)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(processing_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        edges = cv2.Canny(blur, 20, 50)
        
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=20)
        
        # 결과 표시를 위해 Crop된 엣지 이미지를 컬러로 변환
        edges_color_roi = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 전체 화면(edges_color)을 검은색으로 만들고 ROI 부분만 붙여넣음 (시각화용)
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
            
            # [중요] 검출된 라인은 Crop 이미지 기준 좌표이므로
            # 원본 BEV 좌표계로 복원(Offset)해주어야 함.
            global_lines = [] 

            for i, line in enumerate(lines):
                lx1, ly1, lx2, ly2 = line[0]
                
                # 좌표 복원: ROI 시작점(roi_x, roi_y) 더하기
                gx1, gy1 = lx1 + roi_x, ly1 + roi_y
                gx2, gy2 = lx2 + roi_x, ly2 + roi_y
                
                if gy1 > gy2: gx1, gy1, gx2, gy2 = gx2, gy2, gx1, gy1
                
                global_lines.append([gx1, gy1, gx2, gy2]) # 저장해둠

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
                        # 저장해둔 글로벌 좌표 가져오기
                        gx1, gy1, gx2, gy2 = global_lines[valid_indices[idx]]
                        angle = data_angles[idx]

                        if label == -1:
                            cv2.line(bev_viz, (gx1, gy1), (gx2, gy2), (100, 100, 100), 1)
                        else:
                            rank = rank_map[label]
                            color = self.cluster_colors[rank % len(self.cluster_colors)]
                            thickness = 3 if rank == 0 else 1
                            
                            # 시각화 (edges_color에도 그림)
                            cv2.line(edges_color, (gx1, gy1), (gx2, gy2), color, 2)
                            cv2.line(bev_viz, (gx1, gy1), (gx2, gy2), color, thickness)
                            
                            if rank == 0:
                                mid_y = (gy1 + gy2) / 2.0
                                largest_cluster_data.append((angle, mid_y))

        # ---------------------------------------------------------
        # 5. 결과 발행 및 시각화
        # ---------------------------------------------------------
        weighted_avg_angle = self.calculate_weighted_average_angle(largest_cluster_data)
        current_avg_angle = weighted_avg_angle if weighted_avg_angle is not None else 0.0

        angle_msg = Int16()
        angle_msg.data = -int(current_avg_angle)
        self.angle_pub.publish(angle_msg)

        # 화살표 그리기
        arrow_start_pt = (bev_w // 2, bev_h - 50) 
        arrow_len = 100
        angle_rad = math.radians(current_avg_angle)
        arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
        
        cv2.arrowedLine(edges_color, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.arrowedLine(bev_viz, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(bev_viz, f"Avg Angle: {int(current_avg_angle)}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
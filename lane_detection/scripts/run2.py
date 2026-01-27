#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import time
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Int32MultiArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import DBSCAN
from collections import deque

# ==========================================
# 설정 변수
# ==========================================
IMAGE_TOPIC = "/cam1/usb_cam/image_raw" 

# ==========================================
# 기존 알고리즘 함수
# ==========================================
def calculate_midpoint_score(x1, x2, w):
    hw = w/2
    midpoint = int((x1 + x2) / 2)
    return (midpoint - hw) / hw * 100

def calculate_line_score(x1, y1, x2, y2):
    if x2 - x1 == 0: return 0.0
    slope = (y2 - y1) / (x2 - x1)
    abs_slope = abs(slope)
    if abs_slope < 0.3: return 0.0
    theta_deg = math.degrees(math.atan(abs_slope))
    min_theta = math.degrees(math.atan(0.3))
    max_theta = 90.0
    if theta_deg >= max_theta: score_magnitude = 0.0
    elif theta_deg <= min_theta: score_magnitude = 100.0
    else: score_magnitude = 100 * (max_theta - theta_deg) / (max_theta - min_theta)
    return score_magnitude if slope > 0 else -score_magnitude

def average_lines(lines):
    if not lines: return None
    lines_array = np.array(lines)
    averaged_line = np.mean(lines_array, axis=0)
    return averaged_line.astype(int)

# ==========================================
# Lane Detector Class
# ==========================================
class LaneDetector:
    def __init__(self):
        rospy.init_node('canny_lane_detector', anonymous=True)
        
        output_topic_name = rospy.get_param("~output_topic", "des_steer")
        self.pub_steer = rospy.Publisher(output_topic_name, Int16, queue_size=10)
        self.pub_lines_px = rospy.Publisher("/lane_lines_px", Int32MultiArray, queue_size=10)
        self.pub_target_px = rospy.Publisher("/lane_target_px", PointStamped, queue_size=10)
        
        # [추가됨] 횡단보도 상태 발행 (0: 없음, 1: 있음)
        self.pub_cross = rospy.Publisher("/crossline", Int16, queue_size=10)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)
        
        # 조향각 스무딩용 큐
        self.window_size = 15
        self.steer_history = deque(maxlen=self.window_size)
        
        # [추가됨] 횡단보도 감지용 변수
        # 이 값(픽셀 수) 이상이면 횡단보도로 간주 (현장 상황에 맞춰 튜닝 필수!)
        self.cross_threshold = 20000  
        self.cross_queue = deque(maxlen=10) # 10개 프레임 저장

        print(f"Waiting for image topic: {IMAGE_TOPIC}...")

    def filter_by_dbscan(self, lines, img_height):
        if not lines or len(lines) < 2: return lines
        
        features = []
        valid_indices = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            if x2 - x1 == 0: continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 1e-2: continue
            angle = math.atan(slope)
            x_bottom = (img_height - y1) / slope + x1
            features.append([angle, x_bottom])
            valid_indices.append(i)
            
        if not features: return lines

        features = np.array(features)
        weight_angle = 100.0  
        weight_dist  = 0.20   
        features_scaled = np.column_stack((
            features[:, 0] * weight_angle, 
            features[:, 1] * weight_dist   
        ))

        db = DBSCAN(eps=25.0, min_samples=2).fit(features_scaled)
        
        labels = db.labels_
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)
        if not unique_labels: return [] 

        best_label = -1
        max_count = 0
        for label in unique_labels:
            count = np.sum(labels == label)
            if count > max_count:
                max_count = count
                best_label = label
        
        filtered_lines = []
        for i, label in enumerate(labels):
            if label == best_label:
                filtered_lines.append(lines[valid_indices[i]])
        return filtered_lines

    def image_callback(self, msg):
        start_time = time.time()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        h, w = frame.shape[:2]

        # 1. Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 100, 200)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
        mask_eroded = edges 

        # 2. ROI (본닛 마스킹 포함)
        roi_bottom_width = 1.0   
        roi_top_width = 0.9      
        roi_height = 0.45        

        cx = w // 2
        y_top = int(h * (1 - roi_height))
        y_mid = int(h * (1 - roi_height / 2))
        x_bottom_half = int(w * roi_bottom_width / 2)
        x_top_half = int(w * roi_top_width / 2)

        roi_verts = np.array([[
            (cx - x_bottom_half, h),
            (cx - x_top_half, y_top),
            (cx + x_top_half, y_top),
            (cx + x_bottom_half, h)
        ]], dtype=np.int32)

        roi_mask_img = np.zeros_like(mask_eroded)
        cv2.fillPoly(roi_mask_img, roi_verts, 255)

        hood_height_ratio = 0.15 
        hood_width_ratio = 0.50  
        hood_h = int(h * hood_height_ratio)
        hood_w_half = int((w * hood_width_ratio) / 2)
        hood_top_left = (cx - hood_w_half, h - hood_h)
        hood_bottom_right = (cx + hood_w_half, h)

        cv2.rectangle(roi_mask_img, hood_top_left, hood_bottom_right, 0, -1)
        
        # ROI 적용된 이미지 (흰색 픽셀 계산용)
        mask_roi_applied = cv2.bitwise_and(mask_eroded, roi_mask_img)

        # ========================================================
        # [기존 기능] 흰색 픽셀 수 계산
        # ========================================================
        white_pixel_area = cv2.countNonZero(mask_roi_applied)

        # ========================================================
        # [추가됨] 횡단보도 감지 로직 (Queue + Majority Vote)
        # ========================================================
        # 1. 임계값 비교
        is_detected_now = 1 if white_pixel_area > self.cross_threshold else 0
        
        # 2. 큐에 추가
        self.cross_queue.append(is_detected_now)
        
        # 3. 과반수 투표 (10개 중 5개 초과면 1)
        cross_vote_sum = sum(self.cross_queue)
        queue_len = len(self.cross_queue)
        
        final_cross_status = 0
        if queue_len > 0 and cross_vote_sum > (queue_len / 2):
            final_cross_status = 1
        
        # 4. 토픽 발행 (/crossline)
        self.pub_cross.publish(Int16(final_cross_status))
        # ========================================================

        # 3. Hough Transform
        lines = cv2.HoughLinesP(mask_roi_applied, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)
        
        mask_bgr = cv2.cvtColor(mask_roi_applied, cv2.COLOR_GRAY2BGR)
        cv2.polylines(mask_bgr, roi_verts, isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.line(mask_bgr, (cx, 0), (cx, h), (255, 255, 255), 1)

        # 정보 표시 (우측 상단)
        area_text = f"White Area: {white_pixel_area}"
        (text_w, text_h), _ = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(mask_bgr, area_text, (w - text_w - 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ========================================================
        # [추가됨] 횡단보도 감지 시 화면 중앙 표시
        # ========================================================
        if final_cross_status == 1:
            warning_text = "CROSSLINE DETECTED"
            (tw, th), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)
            # 화면 중앙에 빨간 글씨
            cv2.putText(mask_bgr, warning_text, (cx - tw//2, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 4. Filter & Score
        filtering_slope = 0.4 
        right_lines, left_lines = [], []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0: continue 
                slope = dy / dx
                if abs(slope) <= filtering_slope: continue

                ls = calculate_line_score(x1, y1, x2, y2)
                lm = calculate_midpoint_score(x1, y1, w)

                if ls + lm <= 0: left_lines.append([x1, y1, x2, y2])
                else: right_lines.append([x1, y1, x2, y2])
        
        left_lines = self.filter_by_dbscan(left_lines, h)
        right_lines = self.filter_by_dbscan(right_lines, h)

        left_result = average_lines(left_lines)
        right_result = average_lines(right_lines)

        left_mid_point, right_mid_point = 0, w

        # 5. Visualization & Compute
        if left_result is not None:
            lx1, ly1, lx2, ly2 = left_result
            left_mid_point = int((lx1 + lx2) / 2)
            cv2.circle(mask_bgr, (left_mid_point, y_mid), 20, (0, 0, 255), -1)
            cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)
        
        if right_result is not None:
            lx1, ly1, lx2, ly2 = right_result
            right_mid_point = int((lx1 + lx2) / 2)
            cv2.circle(mask_bgr, (right_mid_point, y_mid), 20, (255, 0, 0), -1)
            cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)
        
        final_midpoint = int((left_mid_point + right_mid_point) / 2)

        # 조향각 이동 평균
        image_center_x = w // 2
        self.steer_history.append(final_midpoint)
        
        if len(self.steer_history) > 0:
            filtered_midpoint = int(sum(self.steer_history) / len(self.steer_history))
        else:
            filtered_midpoint = final_midpoint 
        
        pubdata = int(-(filtered_midpoint - image_center_x) * 0.2 )
        msg_steer = Int16()
        msg_steer.data = pubdata
        self.pub_steer.publish(msg_steer)

        # Pub Lines
        lane_data = [-1, -1, -1, -1, -1, -1, -1, -1]
        if left_result is not None: lane_data[0:4] = left_result
        if right_result is not None: lane_data[4:8] = right_result
        lines_msg = Int32MultiArray()
        lines_msg.data = lane_data
        self.pub_lines_px.publish(lines_msg)

        # Pub Target
        target_msg = PointStamped()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.header.frame_id = "camera_frame"
        target_msg.point.x = filtered_midpoint 
        target_msg.point.y = y_mid
        target_msg.point.z = 0.0
        self.pub_target_px.publish(target_msg)

        # Visualization Extra
        cv2.rectangle(mask_bgr, hood_top_left, hood_bottom_right, (0, 0, 255), 2)
        cv2.putText(mask_bgr, "Hood Mask", (hood_top_left[0], hood_top_left[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.circle(mask_bgr, (filtered_midpoint, y_mid), 20, (255, 255, 0), -1)
        cv2.putText(mask_bgr, f"Offset: {msg_steer.data}", (filtered_midpoint - 80, y_mid - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for line in right_lines:
            cv2.line(mask_bgr, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 1)
        for line in left_lines:
            cv2.line(mask_bgr, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)

        combined = np.hstack((frame, mask_bgr))
        target_width = 1200
        scale = target_width / combined.shape[1]
        new_w = int(combined.shape[1] * scale)
        new_h = int(combined.shape[0] * scale)
        combined_small = cv2.resize(combined, (new_w, new_h))
        
        # cv2.imshow('Lane Detector', combined_small)
        # cv2.waitKey(1)

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000) 
        
        # 로그 출력에 Crosswalk 상태 추가
        print(f"Steer: {pubdata}, White: {white_pixel_area}, Cross: {final_cross_status}, Time: {elapsed_time}")

    def clean_up(self):
        cv2.destroyAllWindows()
        print("Clean up done.")

if __name__ == '__main__':
    ld = LaneDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ld.clean_up()
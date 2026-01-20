#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from std_msgs.msg import Int16 
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import DBSCAN
from collections import deque  # [추가] 큐(Queue) 자료구조 사용

# ==========================================
# 설정 변수
# ==========================================
IMAGE_TOPIC = "/usb_cam/image_raw" 

# ==========================================
# 기존 알고리즘 함수 (유지)
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
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)
        
        # [변경] Low Pass Filter 설정을 이동 평균(Moving Average) 방식으로 변경
        # window_size가 클수록 부드럽지만 반응이 느려짐 (보통 5~10 추천)
        self.window_size = 15
        self.steer_history = deque(maxlen=self.window_size)
        
        print(f"Waiting for image topic: {IMAGE_TOPIC}...")
        print(f"Publishing steering to: {output_topic_name} (Method: Moving Average)")

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
        
        # DBSCAN 파라미터
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

        # 2. ROI
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
        mask_roi_applied = cv2.bitwise_and(mask_eroded, roi_mask_img)

        # 3. Hough Transform
        lines = cv2.HoughLinesP(mask_roi_applied, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=50)
        mask_bgr = cv2.cvtColor(mask_roi_applied, cv2.COLOR_GRAY2BGR)
        cv2.polylines(mask_bgr, roi_verts, isClosed=True, color=(0, 255, 0), thickness=2)

        # 4. Filter & Score
        filtering_slope = 0.3 
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

        # --------------------------------------------------------
        # [변경됨] 단순 이동 평균(SMA) 필터 적용
        # --------------------------------------------------------
        image_center_x = w // 2
        # raw_offset = final_midpoint - image_center_x
        
        # 큐에 현재 데이터 추가 (꽉 차면 가장 오래된 것 자동 삭제)
        self.steer_history.append(final_midpoint)
        
        # 큐에 있는 값들의 평균 계산
        if len(self.steer_history) > 0:
            filtered_midpoint = int(sum(self.steer_history) / len(self.steer_history))
        else:
            filtered_midpoint = final_midpoint 
        
        msg_steer = Int16()
        msg_steer.data = filtered_midpoint - image_center_x
        self.pub_steer.publish(msg_steer)
        # --------------------------------------------------------

        cv2.circle(mask_bgr, (filtered_midpoint, y_mid), 20, (255, 255, 0), -1)
        cv2.putText(mask_bgr, f"Offset: {msg_steer.data}", (filtered_midpoint - 80, y_mid - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for line in right_lines:
            x1, y1, x2 ,y2 = line
            cv2.line(mask_bgr, (x1, y1), (x2, y2), (255, 0, 0), 1)
        for line in left_lines:
            x1, y1, x2 ,y2 = line
            cv2.line(mask_bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)

        combined = np.hstack((frame, mask_bgr))
        target_width = 1200
        scale = target_width / combined.shape[1]
        new_w = int(combined.shape[1] * scale)
        new_h = int(combined.shape[0] * scale)
        combined_small = cv2.resize(combined, (new_w, new_h))
        
        cv2.imshow('Lane Detector', combined_small)
        cv2.waitKey(1)

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
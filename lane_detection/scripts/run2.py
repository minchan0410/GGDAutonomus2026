#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import time
import threading
import queue
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
# 보조 함수
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

def average_lines_projected(lines, y_min, y_max):
    if not lines: return None
    x_tops, x_bottoms = [], []
    for line in lines:
        x1, y1, x2, y2 = line
        if x2 == x1:
            x_tops.append(x1); x_bottoms.append(x1)
            continue
        slope = (y2 - y1) / (x2 - x1)
        val_x_top = (y_min - y1) / slope + x1
        val_x_bottom = (y_max - y1) / slope + x1
        x_tops.append(val_x_top); x_bottoms.append(val_x_bottom)
    avg_x_top = int(np.mean(x_tops))
    avg_x_bottom = int(np.mean(x_bottoms))
    return [avg_x_top, y_min, avg_x_bottom, y_max]

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
        self.pub_cross = rospy.Publisher("/crossline", Int16, queue_size=10)
        
        self.bridge = CvBridge()
        # queue_size=1, buff_size를 키워서 최신 프레임만 빠르게 받도록 설정
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        self.window_size = 15
        self.steer_history = deque(maxlen=self.window_size)
        
        self.cross_threshold = 20000  
        self.cross_queue = deque(maxlen=10) 

        # [Thread] 시각화 전용 스레드 설정
        self.vis_queue = queue.Queue(maxsize=1) 
        self.is_running = True
        self.vis_thread = threading.Thread(target=self.display_worker)
        self.vis_thread.daemon = True 
        self.vis_thread.start()

        print(f"Waiting for image topic: {IMAGE_TOPIC}...")

    def display_worker(self):
        """ 시각화(imshow)만 담당하는 스레드 함수 """
        while self.is_running and not rospy.is_shutdown():
            try:
                # 0.1초 대기 후 없으면 다시 루프 (블로킹 방지)
                img = self.vis_queue.get(timeout=0.1)
                cv2.imshow('Lane Detector', img)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except Exception as e:
                rospy.logwarn(f"Display Error: {e}")

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
        features_scaled = np.column_stack((features[:, 0] * weight_angle, features[:, 1] * weight_dist))

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

    def filter_by_position(self, lines, side, img_w, img_h):
        """
        DBSCAN 이후, 반대편 상단 영역에 잘못 검출된 노이즈 선을 제거합니다.
        :param lines: [[x1, y1, x2, y2], ...] 형태의 리스트
        :param side: 'left' or 'right'
        :param img_w: 이미지 너비
        :param img_h: 이미지 높이
        """
        if not lines: return []

        filtered_lines = []
        
        # 1. 제외할 영역(Forbidden Zone) 설정
        # x좌표 기준: 화면의 중앙 (중앙을 넘어가면 반대편 차선일 확률 높음)
        # y좌표 기준: 차선의 위쪽 영역 (아래쪽은 겹칠 수 있으나 윗부분이 반대로 가는 건 노이즈)
        
        threshold_x = img_w // 2  # 화면 중앙
        threshold_y = img_h * 0.5 # 하단 20%를 제외한 위쪽 영역
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # 선의 중점(Midpoint) 계산
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            
            is_noise = False
            
            if side == 'left':
                # 왼쪽 차선 그룹인데, 위치가 '오른쪽 위'에 있는 경우 제거
                # 조건: x가 중앙보다 크고(Right), y가 임계값보다 작음(Top)
                if mx > threshold_x and my < threshold_y:
                    is_noise = True
                    
            elif side == 'right':
                # 오른쪽 차선 그룹인데, 위치가 '왼쪽 위'에 있는 경우 제거
                # 조건: x가 중앙보다 작고(Left), y가 임계값보다 작음(Top)
                if mx < threshold_x and my < threshold_y:
                    is_noise = True
            
            if not is_noise:
                filtered_lines.append(line)
                
        return filtered_lines

    def image_callback(self, msg):
        start_time = time.time() 
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        h, w = frame.shape[:2]
        cx = w // 2
        
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
        roi_height = 0.45        
        y_top = int(h * (1 - roi_height))
        y_mid = int(h * (1 - roi_height / 2))
        
        # ROI Polygon
        roi_verts = np.array([[
            (cx - int(w * 0.5), h),
            (cx - int(w * 0.45), y_top),
            (cx + int(w * 0.45), y_top),
            (cx + int(w * 0.5), h)
        ]], dtype=np.int32)

        roi_mask_img = np.zeros_like(mask_eroded)
        cv2.fillPoly(roi_mask_img, roi_verts, 255)

        # Hood Mask
        hood_h = int(h * 0.1)
        hood_w_half = int((w * 0.50) / 2)
        hood_top_left = (cx - hood_w_half, h - hood_h)
        hood_bottom_right = (cx + hood_w_half, h)
        cv2.rectangle(roi_mask_img, hood_top_left, hood_bottom_right, 0, -1)
        
        mask_roi_applied = cv2.bitwise_and(mask_eroded, roi_mask_img)

        # 3. Crosswalk Logic
        white_pixel_area = cv2.countNonZero(mask_roi_applied)
        is_detected_now = 1 if white_pixel_area > self.cross_threshold else 0
        self.cross_queue.append(is_detected_now)
        final_cross_status = 1 if sum(self.cross_queue) > (len(self.cross_queue) / 2) else 0
        self.pub_cross.publish(Int16(final_cross_status))

        # 4. Hough Transform
        lines = cv2.HoughLinesP(mask_roi_applied, rho=1, theta=np.pi/180, threshold=50, minLineLength=90, maxLineGap=100)
        
        # 시각화를 위한 BGR 이미지는 여기서 생성 (그리기 작업을 위해 필요)
        mask_bgr = cv2.cvtColor(mask_roi_applied, cv2.COLOR_GRAY2BGR)

        # 5. Filter & Lane Compute
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
        
        # (1) DBSCAN Clustering
        left_lines = self.filter_by_dbscan(left_lines, h)
        right_lines = self.filter_by_dbscan(right_lines, h)

        # (2) [NEW] Position Filtering (잘못된 영역의 선 제거)
        # 왼쪽 라인: 중앙보다 오른쪽 위(Right-Upper) 영역 제거
        left_lines = self.filter_by_position(left_lines, 'left', w, h)
        # 오른쪽 라인: 중앙보다 왼쪽 위(Left-Upper) 영역 제거
        right_lines = self.filter_by_position(right_lines, 'right', w, h)

        left_result = average_lines_projected(left_lines, y_top, h)
        right_result = average_lines_projected(right_lines, y_top, h)

        # 6. Steering Compute & Line Drawing
        # 기본값 설정
        left_mid_point = 0
        right_mid_point = w

        # (1) 왼쪽 차선 좌표 계산 및 시각화
        if left_result:
            lx1, ly1, lx2, ly2 = left_result
            if (ly2 - ly1) != 0:
                left_mid_point = int((y_mid - ly1) * (lx2 - lx1) / (ly2 - ly1) + lx1)
            else:
                left_mid_point = lx1
            cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)

        # (2) 오른쪽 차선 좌표 계산 및 시각화
        if right_result:
            lx1, ly1, lx2, ly2 = right_result
            if (ly2 - ly1) != 0:
                right_mid_point = int((y_mid - ly1) * (lx2 - lx1) / (ly2 - ly1) + lx1)
            else:
                right_mid_point = lx1
            cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)
        
        # 차선 역전(Crossing) 방지 및 최소 간격 유지 로직
        if left_result and right_result:
            min_lane_width = 200  
            if left_mid_point > (right_mid_point - min_lane_width):
                left_mid_point = right_mid_point - min_lane_width
            if right_mid_point < (left_mid_point + min_lane_width):
                right_mid_point = left_mid_point + min_lane_width
        
        final_midpoint = int((left_mid_point + right_mid_point) / 2)
        image_center_x = w // 2
        self.steer_history.append(final_midpoint)
        
        filtered_midpoint = int(sum(self.steer_history) / len(self.steer_history)) if self.steer_history else final_midpoint
        
        pubdata = int(-(filtered_midpoint - image_center_x) * 0.2 )
        self.pub_steer.publish(Int16(pubdata))

        # 7. Publish Topics
        lane_data = [-1] * 8
        if left_result: lane_data[0:4] = left_result
        if right_result: lane_data[4:8] = right_result
        self.pub_lines_px.publish(Int32MultiArray(data=lane_data))

        target_msg = PointStamped()
        target_msg.header.stamp = rospy.Time.now()
        target_msg.header.frame_id = "camera_frame"
        target_msg.point.x = filtered_midpoint 
        target_msg.point.y = y_mid
        self.pub_target_px.publish(target_msg)

        # 8. Visualization Overlay 
        # (1) ROI Polygon
        cv2.polylines(mask_bgr, roi_verts, isClosed=True, color=(0, 255, 0), thickness=2)
        
        # (2) Center Line
        cv2.line(mask_bgr, (cx, 0), (cx, h), (255, 255, 255), 1)

        # (3) White Pixel Count
        area_text = f"White Area: {white_pixel_area}"
        (text_w, text_h), _ = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(mask_bgr, area_text, (w - text_w - 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # (4) Hood Mask Rect & Text
        cv2.rectangle(mask_bgr, hood_top_left, hood_bottom_right, (0, 0, 255), 2)
        cv2.putText(mask_bgr, "Hood Mask", (hood_top_left[0], hood_top_left[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # (5) Target & Offset
        cv2.circle(mask_bgr, (filtered_midpoint, y_mid), 20, (255, 255, 0), -1)
        cv2.putText(mask_bgr, f"Offset: {pubdata}", (filtered_midpoint - 80, y_mid - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.circle(mask_bgr, (left_mid_point, y_mid), 20, (0, 0, 255), -1)
        cv2.circle(mask_bgr, (right_mid_point, y_mid), 20, (255, 0, 0), -1)

        # (6) Crossline Warning
        if final_cross_status == 1:
            warning_text = "CROSSLINE DETECTED"
            (tw, th), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)
            cv2.putText(mask_bgr, warning_text, (cx - tw//2, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # ----------------------------------------------------------------
        # [Thread Queue] 완성된 이미지를 디스플레이 스레드로 전달
        # ----------------------------------------------------------------
        if not self.vis_queue.full():
            self.vis_queue.put_nowait(mask_bgr)

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000) 
        
        # 15ms 이상 소요될 때만 경고 출력
        if elapsed_time > 15: 
             print(f"[Lag Warning] Loop Time: {elapsed_time}ms | White: {white_pixel_area}")

    def clean_up(self):
        self.is_running = False
        self.vis_thread.join()
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
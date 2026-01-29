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
from collections import deque

# ==========================================
# 설정 변수
# ==========================================
IMAGE_TOPIC = "/cam1/usb_cam/image_raw" 

# ==========================================
# 보조 함수 (CPU 연산 유지)
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

# [NEW] 중간 영역 필터링 함수 추가
def filter_middle_33_percent(lines):
    """
    직선들의 중점 Y좌표를 기준으로 정렬 후,
    상위 33% (화면 위쪽), 하위 33% (화면 아래쪽)을 버리고
    중간 34% 영역의 직선만 남깁니다.
    """
    if not lines or len(lines) < 3: 
        # 라인이 너무 적으면 필터링 없이 반환 (최소 3개는 있어야 자를 수 있음)
        return lines

    # 1. (line, midpoint_y) 튜플 리스트 생성
    # y좌표가 작을수록 화면 위, 클수록 화면 아래
    lines_with_mid = []
    for line in lines:
        x1, y1, x2, y2 = line
        mid_x = (x1 + x2) / 2.0
        lines_with_mid.append((line, mid_x))

    # 2. x 중점 기준으로 정렬
    lines_with_mid.sort(key=lambda x: x[1])

    # 3. 자를 인덱스 계산
    total_count = len(lines)
    start_idx = int(total_count * 0.33)       # 상위 33% 제외
    end_idx = int(total_count * (1 - 0.33))   # 하위 33% 제외

    # 4. 슬라이싱 (안전장치 포함)
    if start_idx >= end_idx:
        return lines
        
    filtered_data = lines_with_mid[start_idx:end_idx]
    
    # 원래 포맷([x1, y1, x2, y2])으로 복원하여 반환
    return [item[0] for item in filtered_data]

# ==========================================
# Lane Detector Class with CUDA
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
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        self.window_size = 15
        self.steer_history = deque(maxlen=self.window_size)
        
        self.cross_threshold = 20000  
        self.cross_queue = deque(maxlen=10) 

        # [CUDA Init]
        try:
            self.gpu_src = cv2.cuda_GpuMat()
            self.gpu_roi_mask = cv2.cuda_GpuMat()
            self.cuda_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (7, 7), 0)
            self.cuda_canny = cv2.cuda.createCannyEdgeDetector(100, 200)
            
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.cuda_erode = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, kernel_erode, iterations=1)
            
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.cuda_dilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel_dilate, iterations=3)

            # Hough Parameter
            self.cuda_hough = cv2.cuda.createHoughSegmentDetector(
                1.0, 
                np.pi / 180.0,
                50, # 최소 선 길이
                1 # 최대 허용 간격
            )
            print("CUDA Accelerated OpenCV Initialized Successfully.")
        except AttributeError:
            print("[ERROR] No CUDA Support")
            rospy.signal_shutdown("No CUDA Support")

        self.vis_queue = queue.Queue(maxsize=1) 
        self.is_running = True
        self.vis_thread = threading.Thread(target=self.display_worker)
        self.vis_thread.daemon = True 
        self.vis_thread.start()

    def display_worker(self):
        while self.is_running and not rospy.is_shutdown():
            try:
                img = self.vis_queue.get(timeout=0.1)
                cv2.imshow('Lane Detector (CUDA)', img)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except Exception as e:
                rospy.logwarn(f"Display Error: {e}")

    def filter_by_position(self, lines, side, img_w, img_h):
        if not lines: return []
        filtered_lines = []
        threshold_x = img_w // 2 
        threshold_y = img_h * 0.8 
        for line in lines:
            x1, y1, x2, y2 = line
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            is_noise = False
            if side == 'left':
                if mx > threshold_x and my < threshold_y: is_noise = True
            elif side == 'right':
                if mx < threshold_x and my < threshold_y: is_noise = True
            if not is_noise: filtered_lines.append(line)
        return filtered_lines

    def image_callback(self, msg):
        start_time = time.time() 
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            return

        h, w = frame.shape[:2]
        cx = w // 2
        
        # 1. Pre-processing (CUDA)
        self.gpu_src.upload(frame)
        gpu_gray = cv2.cuda.cvtColor(self.gpu_src, cv2.COLOR_BGR2GRAY)
        gpu_blur = self.cuda_gaussian.apply(gpu_gray)
        gpu_edges = self.cuda_canny.detect(gpu_blur)
        _, gpu_thresh = cv2.cuda.threshold(gpu_gray, 1, 255, cv2.THRESH_BINARY)
        gpu_mask = self.cuda_erode.apply(gpu_thresh)
        gpu_edges = cv2.cuda.bitwise_and(gpu_edges, gpu_edges, mask=gpu_mask)
        gpu_edges = self.cuda_dilate.apply(gpu_edges)

        # 2. ROI Logic
        roi_height = 0.45        
        y_top = int(h * (1 - roi_height))
        y_mid = int(h * (1 - roi_height / 2))
        
        roi_verts = np.array([[
            (cx - int(w * 0.5), h),
            (cx - int(w * 0.45), y_top),
            (cx + int(w * 0.45), y_top),
            (cx + int(w * 0.5), h)
        ]], dtype=np.int32)

        roi_mask_cpu = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_cpu, roi_verts, 255)
        hood_h = int(h * 0.1)
        hood_w_half = int((w * 0.50) / 2)
        hood_top_left = (cx - hood_w_half, h - hood_h)
        hood_bottom_right = (cx + hood_w_half, h)
        cv2.rectangle(roi_mask_cpu, hood_top_left, hood_bottom_right, 0, -1)
        
        self.gpu_roi_mask.upload(roi_mask_cpu)
        gpu_roi_applied = cv2.cuda.bitwise_and(gpu_edges, self.gpu_roi_mask)

        # 3. CPU Logic Requirement
        mask_roi_applied = gpu_roi_applied.download()
        white_pixel_area = cv2.countNonZero(mask_roi_applied)
        is_detected_now = 1 if white_pixel_area > self.cross_threshold else 0
        self.cross_queue.append(is_detected_now)
        final_cross_status = 1 if sum(self.cross_queue) > (len(self.cross_queue) / 2) else 0
        self.pub_cross.publish(Int16(final_cross_status))

        # 4. Hough Transform
        d_lines = self.cuda_hough.detect(gpu_roi_applied)
        if d_lines is not None and not d_lines.empty():
            lines = d_lines.download()
            lines = lines.reshape(-1, 1, 4)
        else:
            lines = None

        mask_bgr = cv2.cvtColor(mask_roi_applied, cv2.COLOR_GRAY2BGR)

        # 5. Filter & Lane Compute (CPU)
        filtering_slope = 0.5 
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

        # Position Filtering
        left_lines = self.filter_by_position(left_lines, 'left', w, h)
        right_lines = self.filter_by_position(right_lines, 'right', w, h)
        
        # -------------------------------------------------------------
        # [NEW] 상위 33%, 하위 33% 제거 필터링 적용 (중간 34%만 남김)
        # -------------------------------------------------------------
        left_lines = filter_middle_33_percent(left_lines)
        right_lines = filter_middle_33_percent(right_lines)
        # -------------------------------------------------------------
        
        # Visualization Lines (Raw)
        if left_lines:
            for lx1, ly1, lx2, ly2 in left_lines:
                cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (0, 0, 255), 1)
        if right_lines:
            for lx1, ly1, lx2, ly2 in right_lines:
                cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (255, 0, 0), 1)

        left_result = average_lines_projected(left_lines, y_top, h)
        right_result = average_lines_projected(right_lines, y_top, h)

        # 6. Steering Compute
        left_mid_point = 0
        right_mid_point = w

        if left_result:
            lx1, ly1, lx2, ly2 = left_result
            if (ly2 - ly1) != 0:
                left_mid_point = int((y_mid - ly1) * (lx2 - lx1) / (ly2 - ly1) + lx1)
            else:
                left_mid_point = lx1
            cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (0, 0, 255), 3)

        if right_result:
            lx1, ly1, lx2, ly2 = right_result
            if (ly2 - ly1) != 0:
                right_mid_point = int((y_mid - ly1) * (lx2 - lx1) / (ly2 - ly1) + lx1)
            else:
                right_mid_point = lx1
            cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)
        
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

        # 7. Publish & Vis
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

        cv2.polylines(mask_bgr, roi_verts, isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.line(mask_bgr, (cx, 0), (cx, h), (255, 255, 255), 1)

        area_text = f"White Area: {white_pixel_area}"
        (text_w, text_h), _ = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(mask_bgr, area_text, (w - text_w - 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.rectangle(mask_bgr, hood_top_left, hood_bottom_right, (0, 0, 255), 2)
        cv2.putText(mask_bgr, "Hood Mask", (hood_top_left[0], hood_top_left[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.circle(mask_bgr, (filtered_midpoint, y_mid), 20, (255, 255, 0), -1)
        cv2.putText(mask_bgr, f"Offset: {pubdata}", (filtered_midpoint - 80, y_mid - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.circle(mask_bgr, (left_mid_point, y_mid), 20, (0, 0, 255), -1)
        cv2.circle(mask_bgr, (right_mid_point, y_mid), 20, (255, 0, 0), -1)

        if final_cross_status == 1:
            warning_text = "CROSSLINE DETECTED"
            (tw, th), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 1)
            cv2.putText(mask_bgr, warning_text, (cx - tw//2, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if not self.vis_queue.full():
            self.vis_queue.put_nowait(mask_bgr)

        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000) 
        if elapsed_time > 10: 
             print(f"[Lag Warning] Loop Time: {elapsed_time}ms")

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
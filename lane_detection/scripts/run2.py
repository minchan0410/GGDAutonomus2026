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

def extend_line_to_roi(x1, y1, x2, y2, y_min, y_max):
    """
    직선을 ROI의 상단(y_min)과 하단(y_max)까지 연장
    공식: x = (y - y1) / slope + x1
    """
    if x2 == x1:  # 수직선 예외 처리
        return [x1, y_min, x1, y_max]
    
    slope = (y2 - y1) / (x2 - x1)
    if slope == 0: 
        return [x1, y_min, x2, y_max]

    val_x_top = (y_min - y1) / slope + x1
    val_x_bottom = (y_max - y1) / slope + x1
    
    return [int(val_x_top), int(y_min), int(val_x_bottom), int(y_max)]

def average_lines_projected(lines):
    if not lines: return None
    
    x_tops = [line[0] for line in lines]
    x_bottoms = [line[2] for line in lines]
    
    y_min = lines[0][1]
    y_max = lines[0][3]

    avg_x_top = int(np.mean(x_tops))
    avg_x_bottom = int(np.mean(x_bottoms))
    
    return [avg_x_top, y_min, avg_x_bottom, y_max]

def filter_innermost_lines(lines, lr):
    if not lines or len(lines) < 3:
        return lines

    lines_with_mid = []
    for line in lines:
        x1, y1, x2, y2 = line
        mid_x = (x1 + x2) / 2.0
        lines_with_mid.append((line, mid_x))

    if lr == 'left':
        # 왼쪽 차선: X가 클수록 안쪽(중앙) -> 내림차순
        lines_with_mid.sort(key=lambda x: x[1], reverse=True)
    else:
        # 오른쪽 차선: X가 작을수록 안쪽(중앙) -> 오름차순
        lines_with_mid.sort(key=lambda x: x[1], reverse=False)
    
    total_count = len(lines)
    start_idx = int(total_count * 0.0)      
    end_idx = int(total_count * 0.30)

    if end_idx <= start_idx: end_idx = start_idx + 1
    if end_idx > total_count: end_idx = total_count

    filtered_data = lines_with_mid[start_idx:end_idx]
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
        
        self.cross_threshold = 14000  
        self.cross_queue = deque(maxlen=10) 
        
        # [Performance Optimization] 마스크 캐싱용 변수
        self.roi_mask_cpu = None
        self.cached_roi_verts = None

        # [CUDA Init]
        try:
            self.gpu_src = cv2.cuda_GpuMat()
            self.gpu_roi_mask = cv2.cuda_GpuMat()
            self.cuda_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (7, 7), 0)
            self.cuda_canny = cv2.cuda.createCannyEdgeDetector(100, 200)
            
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.cuda_erode = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, kernel_erode, iterations=1)
            
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.cuda_dilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel_dilate, iterations=2)

            self.cuda_hough = cv2.cuda.createHoughSegmentDetector(
                2.0, np.pi / 180.0, 50, 1
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
        
        for line in lines:
            x1, y1, x2, y2 = line
            mx = (x1 + x2) / 2
            
            is_noise = False
            if side == 'left':
                if mx > threshold_x + 50: is_noise = True
            elif side == 'right':
                if mx < threshold_x - 50: is_noise = True
            
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
        _, gpu_thresh = cv2.cuda.threshold(gpu_gray, 20, 255, cv2.THRESH_BINARY)
        gpu_mask = self.cuda_erode.apply(gpu_thresh)
        gpu_edges = cv2.cuda.bitwise_and(gpu_edges, gpu_edges, mask=gpu_mask)
        gpu_edges = self.cuda_dilate.apply(gpu_edges)

        # 2. ROI Logic [Optimized]
        roi_height = 0.45        
        y_top = int(h * (1 - roi_height)) 
        y_mid = int(h * (1 - roi_height / 2))
        y_bottom = h 

        # 마스크를 매번 생성하지 않고, 처음이거나 해상도가 바뀔 때만 생성
        if self.roi_mask_cpu is None or self.roi_mask_cpu.shape != (h, w):
            roi_verts = np.array([[
                (cx - int(w * 0.5), h),
                (cx - int(w * 0.45), y_top),
                (cx + int(w * 0.45), y_top),
                (cx + int(w * 0.5), h)
            ]], dtype=np.int32)
            
            self.cached_roi_verts = roi_verts # 시각화를 위해 저장
            
            self.roi_mask_cpu = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(self.roi_mask_cpu, roi_verts, 255)
            
            hood_h = int(h * 0.1)
            hood_w_half = int((w * 0.50) / 2)
            hood_top_left = (cx - hood_w_half, h - hood_h)
            hood_bottom_right = (cx + hood_w_half, h)
            cv2.rectangle(self.roi_mask_cpu, hood_top_left, hood_bottom_right, 0, -1)
            
            # GPU로 업로드 (마스크가 바뀔 때만 수행)
            self.gpu_roi_mask.upload(self.roi_mask_cpu)

        # 이미 업로드된 GPU 마스크 사용하여 AND 연산
        gpu_roi_applied = cv2.cuda.bitwise_and(gpu_edges, self.gpu_roi_mask)

        # 3. CPU Logic Requirement (Crosswalk)
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
                
                projected_line = extend_line_to_roi(x1, y1, x2, y2, y_top, y_bottom)

                if ls + lm <= 0: 
                    left_lines.append(projected_line)
                else: 
                    right_lines.append(projected_line)

        left_lines = self.filter_by_position(left_lines, 'left', w, h)
        right_lines = self.filter_by_position(right_lines, 'right', w, h)

        if left_lines:
            for lx1, ly1, lx2, ly2 in left_lines:
                cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (0, 0, 100), 1)
        if right_lines:
            for lx1, ly1, lx2, ly2 in right_lines:
                cv2.line(mask_bgr, (lx1, ly1), (lx2, ly2), (100, 0, 0), 1)
        
        left_lines = filter_innermost_lines(left_lines, 'left')
        right_lines = filter_innermost_lines(right_lines, 'right')

        left_result = average_lines_projected(left_lines)
        right_result = average_lines_projected(right_lines)

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
        
        pubdata = int(-(filtered_midpoint - image_center_x) * 0.18 )
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
        
        # [Visual Update] ROI 마스크 외곽선
        if self.cached_roi_verts is not None:
            cv2.polylines(mask_bgr, self.cached_roi_verts, isClosed=True, color=(0, 255, 0), thickness=2)
            
        cv2.line(mask_bgr, (cx, 0), (cx, h), (255, 255, 255), 1)

        # ========================================================
        # [NEW] ROI 내 대칭 세로선 그리기 (Guide Lines)
        # ========================================================
        guide_offset_top = 145    # 상단 중심 거리 (멀리 있는 도로 폭, 좁게 설정)
        guide_offset_bottom = 275 # 하단 중심 거리 (가까이 있는 도로 폭, 넓게 설정)
        
        # 왼쪽 가이드라인: (Top Left) -> (Bottom Left)
        # 식: (cx - top_offset, y_top) -> (cx - bottom_offset, h)
        cv2.line(mask_bgr, 
                (cx - guide_offset_top, y_top), 
                (cx - guide_offset_bottom, h), 
                (0, 255, 255), 3)
        
        # 오른쪽 가이드라인: (Top Right) -> (Bottom Right)
        # 식: (cx + top_offset, y_top) -> (cx + bottom_offset, h)
        cv2.line(mask_bgr,
                (cx + guide_offset_top, y_top), 
                (cx + guide_offset_bottom, h), 
                (0, 255, 255), 3)
        # ========================================================

        area_text = f"White Area: {white_pixel_area}"
        (text_w, text_h), _ = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(mask_bgr, area_text, (w - text_w - 20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Hood Mask는 이제 self.roi_mask_cpu에 통합되어 있지만 시각화를 위해 사각형은 따로 그림
        hood_h = int(h * 0.1)
        hood_w_half = int((w * 0.50) / 2)
        hood_top_left = (cx - hood_w_half, h - hood_h)
        hood_bottom_right = (cx + hood_w_half, h)
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
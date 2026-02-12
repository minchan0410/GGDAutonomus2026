#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from collections import deque

from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge

def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

class CurLaneDetectorCudaViz:
    def __init__(self):
        rospy.init_node("cur_lane_detector", anonymous=False)

        # ---- 파라미터 로드 ----
        self.image_topic = rospy.get_param("~image_topic", "/cam1/usb_cam/image_raw")
        self.cur_lane_topic = rospy.get_param("~cur_lane_topic", "/cur_lane")

        self.bottom_ratio = float(rospy.get_param("~roi/bottom_ratio", 0.30))
        self.split_ratio = float(rospy.get_param("~roi/split_ratio", 0.50))

        # [업데이트] 초록색 판단 문턱값 (우측 영역에서 초록색이 이 비율 이상이면 2차선)
        self.green_threshold = float(rospy.get_param("~decision/green_threshold", 0.15)) 
        self.min_valid_pixels = int(rospy.get_param("~decision/min_valid_pixels", 1000))

        self.vote_window = int(rospy.get_param("~vote/window", 10))
        self.gray_lo = [int(x) for x in rospy.get_param("~hsv/gray/lo", [0, 0, 50])]
        self.gray_hi = [int(x) for x in rospy.get_param("~hsv/gray/hi", [179, 60, 200])]
        self.green_lo = [int(x) for x in rospy.get_param("~hsv/green/lo", [35, 40, 40])]
        self.green_hi = [int(x) for x in rospy.get_param("~hsv/green/hi", [95, 255, 255])]
        self.white_lo = [int(x) for x in rospy.get_param("~hsv/white/lo", [0, 0, 200])]
        self.white_hi = [int(x) for x in rospy.get_param("~hsv/white/hi", [179, 50, 255])]

        self.cuda_device = int(rospy.get_param("~cuda/device", 0))
        self.debug_enable = bool(rospy.get_param("~debug/enable", True))

        self.bridge = CvBridge()
        self.vote_q = deque(maxlen=max(1, self.vote_window))
        self.last_lane = 1

        self.pub_lane = rospy.Publisher(self.cur_lane_topic, Int16, queue_size=10)
        self.pub_overlay = rospy.Publisher("/cur_lane/debug_overlay", Image, queue_size=1)
        self.pub_class = rospy.Publisher("/cur_lane/debug_class", Image, queue_size=1)

        self._init_cuda()
        self.sub = rospy.Subscriber(self.image_topic, Image, self.cb_img, queue_size=1, buff_size=2**24)

    def _init_cuda(self):
        cv2.cuda.setDevice(self.cuda_device)
        self.use_cuda = True
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.morph_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, self.morph_kernel)

    def _cuda_in_range(self, hsv_gpu, lo, hi):
        channels = cv2.cuda.split(hsv_gpu)
        masks = []
        for i in range(3):
            _, m1 = cv2.cuda.threshold(channels[i], lo[i], 255, cv2.THRESH_BINARY)
            _, m2 = cv2.cuda.threshold(channels[i], hi[i], 255, cv2.THRESH_BINARY_INV)
            masks.append(cv2.cuda.bitwise_and(m1, m2))
        res = cv2.cuda.bitwise_and(masks[0], masks[1])
        return cv2.cuda.bitwise_and(res, masks[2])

    def _decide_lane_cuda(self, roi_bgr):
        gpu = cv2.cuda_GpuMat()
        gpu.upload(roi_bgr)
        hsv = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2HSV)

        # 1. 색상 분류
        m_gray = self._cuda_in_range(hsv, self.gray_lo, self.gray_hi)
        m_green = self._cuda_in_range(hsv, self.green_lo, self.green_hi)
        m_white = self._cuda_in_range(hsv, self.white_lo, self.white_hi)

        # 2. 노이즈 제거
        m_gray = self.morph_filter.apply(m_gray)
        m_green = self.morph_filter.apply(m_green)

        # 3. 우측 영역 집중 분석
        h, w = roi_bgr.shape[:2]
        split_x = int(w * self.split_ratio)
        
        # 우측 절반 ROI 추출
        R_gray = m_gray.colRange(split_x, w)
        R_green = m_green.colRange(split_x, w)
        R_white = m_white.colRange(split_x, w)

        def count_nz(m): return int(cv2.cuda.countNonZero(m))

        Rg = count_nz(R_gray)
        Rgr = count_nz(R_green)
        Rw = count_nz(R_white)

        total_right_valid = Rg + Rgr + Rw

        # [알고리즘 업데이트]
        # 우측 영역에서 초록색이 차지하는 비율 계산
        if total_right_valid < 100: # 데이터 부족 시
            lane_raw = self.last_lane
        else:
            green_ratio_right = float(Rgr) / float(total_right_valid)
            
            # 우측에 초록색(잔디)이 많으면 2차선, 아니면(회색 도로면) 1차선
            if green_ratio_right > self.green_threshold:
                lane_raw = 2
            else:
                lane_raw = 1

        stats = (Rg, Rgr, Rw, total_right_valid, green_ratio_right)
        return lane_raw, (m_gray, m_green, m_white), stats, split_x

    def cb_img(self, msg):
        try:
            full_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        H, W = full_bgr.shape[:2]
        y0 = clamp_int(int(H * (1.0 - self.bottom_ratio)), 0, H - 1)
        roi_bgr = full_bgr[y0:H, :, :]

        lane_raw, masks, stats, split_x = self._decide_lane_cuda(roi_bgr)
        
        # 투표 (안정성 확보)
        self.vote_q.append(lane_raw)
        counts = [self.vote_q.count(1), self.vote_q.count(2)]
        lane_voted = 1 if counts[0] >= counts[1] else 2
        
        self.last_lane = lane_voted
        self.pub_lane.publish(Int16(data=lane_voted))

        if self.debug_enable:
            m_gray, m_green, m_white = masks
            gray, green, white = m_gray.download(), m_green.download(), m_white.download()
            
            # 시각화용 이미지 생성
            class_img = np.zeros_like(roi_bgr)
            class_img[white > 0] = (255, 255, 255)
            class_img[green > 0] = (0, 255, 0)
            class_img[gray > 0] = (128, 128, 128)
            
            overlay = full_bgr.copy()
            cv2.line(overlay, (split_x, y0), (split_x, H-1), (0, 0, 255), 2)
            
            Rg, Rgr, Rw, tot, g_ratio = stats
            txt = f"Lane:{lane_voted} | R_Green:{g_ratio:.2f}"
            cv2.putText(overlay, txt, (10, 30), 1, 1.5, (0, 255, 0) if lane_voted==2 else (255, 255, 255), 2)
            
            self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(overlay, "bgr8"))
            self.pub_class.publish(self.bridge.cv2_to_imgmsg(class_img, "bgr8"))

if __name__ == "__main__":
    CurLaneDetectorCudaViz()
    rospy.spin()
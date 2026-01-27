#!/home/vic/yoloenv/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import deque, Counter  # [추가] 큐와 최빈값 계산을 위한 모듈

# ==========================================
# 출력 상태 상수
# ==========================================
STATE_NONE   = 0
STATE_GREEN  = 1
STATE_RED    = 2
STATE_YELLOW = 3

class TrafficDetectionNode:
    def __init__(self):
        rospy.init_node("traffic_detection", anonymous=False)
        self.bridge = CvBridge()

        # 1. 통신 파라미터
        self.image_topic = rospy.get_param("~image_topic", "/cam2/usb_cam/image_raw")
        self.state_topic = rospy.get_param("~state_topic", "/traffic")
        self.use_overlay = bool(rospy.get_param("~overlay_enable", True))
        self.overlay_topic = rospy.get_param("~overlay_topic", "/traffic_overlay/image")

        # [추가] 결과 스무딩을 위한 큐 설정
        # 큐의 길이가 길수록 안정적이지만 반응 속도가 느려집니다. (예: 30fps 기준 10~15 추천)
        self.queue_len = int(rospy.get_param("~queue_len", 10))
        self.result_queue = deque(maxlen=self.queue_len)

        # 2. YOLO 파라미터 & 클래스 ID
        self.weights = rospy.get_param("~weights", "")
        self.conf_th = float(rospy.get_param("~conf_th", 0.40))
        
        # 기본값: Green(0), Red(1), Yellow(2)
        self.cls_id_green  = int(rospy.get_param("~class_id_green", 0))
        self.cls_id_red    = int(rospy.get_param("~class_id_red", 1))
        self.cls_id_yellow = int(rospy.get_param("~class_id_yellow", 2))

        # 3. ROI (관심 영역) 파라미터 [0.0 ~ 1.0 단위]
        self.roi_x_pct = float(rospy.get_param("~roi_x", 0.0))
        self.roi_y_pct = float(rospy.get_param("~roi_y", 0.0))
        self.roi_w_pct = float(rospy.get_param("~roi_w", 0.0))
        self.roi_h_pct = float(rospy.get_param("~roi_h", 0.0))

        # 모델 로드
        if not self.weights:
            rospy.logerr("Weights path is empty!")
            raise RuntimeError("Weight param missing")

        rospy.loginfo(f"[Traffic] Load YOLO: {self.weights}")
        self.model = YOLO(self.weights)

        # GPU 가속 확인
        if torch.cuda.is_available():
            self.device = 'cuda'
            rospy.loginfo(f"[Traffic] Inference Device: {self.device}")
        else:
            rospy.logfatal("[Traffic] GPU (CUDA) not found! This node requires a GPU to run.")
            raise RuntimeError("CUDA not available")

        # Pub / Sub
        self.pub_state = rospy.Publisher(self.state_topic, Int16, queue_size=1)
        if self.use_overlay:
            self.pub_overlay = rospy.Publisher(self.overlay_topic, Image, queue_size=1)

        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1, buff_size=2**24)

    def get_roi_pixel_rect(self, img_w, img_h):
        """ 퍼센트 설정을 픽셀로 변환 """
        if self.roi_w_pct <= 0.0 or self.roi_h_pct <= 0.0:
            return 0, 0, img_w, img_h

        x = int(self.roi_x_pct * img_w)
        y = int(self.roi_y_pct * img_h)
        w = int(self.roi_w_pct * img_w)
        h = int(self.roi_h_pct * img_h)

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(0, min(w, img_w - x))
        h = max(0, min(h, img_h - y))

        return x, y, w, h

    def is_in_roi(self, box, roi_rect):
        """
        박스 중심이 ROI 픽셀 영역 안에 있는지 확인
        """
        rx, ry, rw, rh = roi_rect
        if rw <= 0 or rh <= 0: 
            return True

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh):
            return True
        return False

    def cb_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"Image decode error: {e}")
            return

        img_h, img_w = frame.shape[:2]
        current_roi_rect = self.get_roi_pixel_rect(img_w, img_h)

        # YOLO 추론
        results = self.model.predict(frame, conf=self.conf_th, verbose=False, half=True)
        
        # [현재 프레임]에서의 투표 결과 (박스가 여러 개일 수 있으므로)
        frame_vote_counts = {STATE_RED: 0, STATE_YELLOW: 0, STATE_GREEN: 0}
        detected_boxes = []

        if results:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls  = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                if not self.is_in_roi(xyxy, current_roi_rect):
                    continue

                target_state = STATE_NONE
                if cls == self.cls_id_red:      target_state = STATE_RED
                elif cls == self.cls_id_yellow: target_state = STATE_YELLOW
                elif cls == self.cls_id_green:  target_state = STATE_GREEN
                
                if target_state != STATE_NONE:
                    frame_vote_counts[target_state] += 1
                    detected_boxes.append((xyxy, target_state, conf))

        # 1. 현재 프레임의 대표 상태 결정
        frame_result = STATE_NONE
        max_count = 0
        for state in [STATE_RED, STATE_YELLOW, STATE_GREEN]:
            if frame_vote_counts[state] > max_count:
                max_count = frame_vote_counts[state]
                frame_result = state
        
        # [수정됨] 2. 결과 큐(History)에 현재 프레임 결과 추가
        # STATE_NONE이라도 큐에 추가하여 신호가 사라진 상태를 반영해야 함
        self.result_queue.append(frame_result)

        # [수정됨] 3. 큐 전체에서 최빈값(Majority Vote) 결정
        # Counter.most_common(1)은 [(값, 빈도수)] 형태 리스트 반환
        if len(self.result_queue) > 0:
            most_common = Counter(self.result_queue).most_common(1)
            final_smoothed_state = most_common[0][0]
        else:
            final_smoothed_state = STATE_NONE

        # 4. 발행 (스무딩된 결과 발행)
        self.pub_state.publish(Int16(final_smoothed_state))

        # 5. 오버레이 (박스는 현재 프레임 기준, 결과 텍스트는 스무딩된 기준)
        if self.use_overlay:
            self.publish_overlay(frame, detected_boxes, final_smoothed_state, current_roi_rect, msg.header)

    def publish_overlay(self, frame, boxes, final_state, roi_rect, header):
        display = frame.copy()
        rx, ry, rw, rh = roi_rect
        img_h, img_w = display.shape[:2]

        if rw < img_w or rh < img_h:
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
            cv2.putText(display, "ROI Filter", (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        colors = {
            STATE_RED:    (0, 0, 255),
            STATE_GREEN:  (0, 255, 0),
            STATE_YELLOW: (0, 255, 255)
        }
        label_map = {STATE_RED:"RED", STATE_GREEN:"GREEN", STATE_YELLOW:"YELLOW"}

        for (xyxy, state, conf) in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            color = colors.get(state, (200, 200, 200))
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{label_map.get(state)} {conf:.2f}"
            cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        status_text = ["NONE", "GREEN", "RED", "YELLOW"]
        result_str = f"RESULT: {status_text[final_state]}"
        
        # 큐 상태 표시 (디버깅용 - 선택 사항)
        # 예: Q[N N R R R R] 형태
        # debug_q_str = f"Q Len:{len(self.result_queue)}"
        # cv2.putText(display, debug_q_str, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display, result_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(display, result_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        img_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        img_msg.header = header
        self.pub_overlay.publish(img_msg)

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = TrafficDetectionNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
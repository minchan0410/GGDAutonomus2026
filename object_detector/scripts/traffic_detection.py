#!/home/vic/yoloenv/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
from ultralytics import YOLO

# ==========================================
# 출력 상태 상수
# ==========================================
STATE_NONE   = 0
STATE_GREEN  = 1
STATE_YELLOW = 2
STATE_RED    = 3

class TrafficDetectionNode:
    def __init__(self):
        rospy.init_node("traffic_detection", anonymous=False)
        self.bridge = CvBridge()

        # 1. 통신 파라미터
        self.image_topic = rospy.get_param("~image_topic", "/head_camera/image_raw")
        self.state_topic = rospy.get_param("~state_topic", "/traffic_light/state")
        self.use_overlay = bool(rospy.get_param("~overlay_enable", True))
        self.overlay_topic = rospy.get_param("~overlay_topic", "/traffic_overlay/image")

        # 2. YOLO 파라미터
        self.weights = rospy.get_param("~weights", "")
        self.conf_th = float(rospy.get_param("~conf_th", 0.40))
        
        self.cls_id_red    = int(rospy.get_param("~class_id_red", 1))
        self.cls_id_yellow = int(rospy.get_param("~class_id_yellow", 2))
        self.cls_id_green  = int(rospy.get_param("~class_id_green", 0))

        # =========================================================
        # 3. ROI (관심 영역) 파라미터 [0.0 ~ 1.0 단위]
        # 예: x=0.25, w=0.5 이면 가로 중앙 50% 영역 사용
        # w나 h가 0.0이면 전체 화면 사용
        # =========================================================
        self.roi_x_pct = float(rospy.get_param("~roi_x", 0.0))
        self.roi_y_pct = float(rospy.get_param("~roi_y", 0.0))
        self.roi_w_pct = float(rospy.get_param("~roi_w", 0.0))
        self.roi_h_pct = float(rospy.get_param("~roi_h", 0.0))

        # 모델 로드
        if not self.weights:
            rospy.logerr("Weigths path is empty!")
            raise RuntimeError("Weight param missing")

        rospy.loginfo(f"[Traffic] Load YOLO: {self.weights}")
        self.model = YOLO(self.weights)

        # Pub / Sub
        self.pub_state = rospy.Publisher(self.state_topic, Int16, queue_size=1)
        if self.use_overlay:
            self.pub_overlay = rospy.Publisher(self.overlay_topic, Image, queue_size=1)

        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)

    def get_roi_pixel_rect(self, img_w, img_h):
        """
        퍼센트(0.0~1.0) 설정을 현재 이미지 크기에 맞춰 픽셀(px)로 변환
        """
        # ROI 크기가 설정되지 않았으면(0.0) 전체 화면 반환
        if self.roi_w_pct <= 0.0 or self.roi_h_pct <= 0.0:
            return 0, 0, img_w, img_h

        # 픽셀 계산
        x = int(self.roi_x_pct * img_w)
        y = int(self.roi_y_pct * img_h)
        w = int(self.roi_w_pct * img_w)
        h = int(self.roi_h_pct * img_h)

        # 경계 체크 (이미지 밖으로 나가지 않게)
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(0, min(w, img_w - x))
        h = max(0, min(h, img_h - y))

        return x, y, w, h

    def is_in_roi(self, box, roi_rect):
        """
        박스 중심이 계산된 ROI 픽셀 영역 안에 있는지 확인
        box: [x1, y1, x2, y2]
        roi_rect: (rx, ry, rw, rh)
        """
        rx, ry, rw, rh = roi_rect
        
        # 전체 화면 모드라면 무조건 True
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
            return

        # 이미지 크기에 맞춰 ROI 픽셀 좌표 계산
        img_h, img_w = frame.shape[:2]
        current_roi = self.get_roi_pixel_rect(img_w, img_h)

        # 1. YOLO 추론
        results = self.model.predict(frame, conf=self.conf_th, verbose=False)
        
        vote_counts = {STATE_RED: 0, STATE_YELLOW: 0, STATE_GREEN: 0}
        detected_boxes = []

        if results:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls  = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                # 2. ROI 필터링 (계산된 픽셀 좌표 사용)
                if not self.is_in_roi(xyxy, current_roi):
                    continue

                target_state = STATE_NONE
                if cls == self.cls_id_red:    target_state = STATE_RED
                elif cls == self.cls_id_yellow: target_state = STATE_YELLOW
                elif cls == self.cls_id_green:  target_state = STATE_GREEN
                
                if target_state != STATE_NONE:
                    vote_counts[target_state] += 1
                    detected_boxes.append((xyxy, target_state, conf))

        # 3. 최빈값 결정
        final_state = STATE_NONE
        max_count = 0
        for state in [STATE_RED, STATE_YELLOW, STATE_GREEN]:
            if vote_counts[state] > max_count:
                max_count = vote_counts[state]
                final_state = state

        # 4. 발행
        self.pub_state.publish(Int16(final_state))

        # 5. 오버레이
        if self.use_overlay:
            self.publish_overlay(frame, detected_boxes, final_state, current_roi, msg.header)

    def publish_overlay(self, frame, boxes, final_state, roi_rect, header):
        display = frame.copy()
        rx, ry, rw, rh = roi_rect
        img_h, img_w = display.shape[:2]

        # ROI 그리기 (설정된 경우만, 전체화면일 땐 안 그림)
        # 전체 화면이 아닐 때만 사각형 표시 (전체 width와 다를 때)
        if rw < img_w or rh < img_h:
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
            cv2.putText(display, "ROI Area", (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 박스 및 결과 그리기
        colors = {STATE_RED: (0, 0, 255), STATE_GREEN: (0, 255, 0), STATE_YELLOW: (0, 255, 255)}
        label_map = {STATE_RED:"RED", STATE_YELLOW:"YELLOW", STATE_GREEN:"GREEN"}

        for (xyxy, state, conf) in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            color = colors.get(state, (200, 200, 200))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, f"{label_map.get(state)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 결과 텍스트
        status_text = ["NONE", "GREEN", "YELLOW", "RED"]
        result_str = f"RESULT: {status_text[final_state]}"
        cv2.putText(display, result_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(display, result_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        img_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
        img_msg.header = header
        self.pub_overlay.publish(img_msg)

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    node = TrafficDetectionNode()
    node.spin()
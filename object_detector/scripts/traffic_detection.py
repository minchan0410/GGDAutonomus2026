#!/home/vic/yoloenv/bin/python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Header
from cv_bridge import CvBridge

from ultralytics import YOLO

# 0: unknown/none, 1: red, 2: yellow, 3: green
UNKNOWN = 0
RED = 1
YELLOW = 2
GREEN = 3


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def hsv_color_vote(bgr_roi):
    if bgr_roi is None or bgr_roi.size == 0:
        return UNKNOWN

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    # red (wrap-around)
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    green  = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))

    k = np.ones((3, 3), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k, iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k, iterations=1)

    r = int(np.sum(red > 0))
    y = int(np.sum(yellow > 0))
    g = int(np.sum(green > 0))

    total = bgr_roi.shape[0] * bgr_roi.shape[1]
    if total <= 0:
        return UNKNOWN

    min_ratio = 0.02
    best = max(r, y, g)
    if best / float(total) < min_ratio:
        return UNKNOWN

    if best == r:
        return RED
    elif best == y:
        return YELLOW
    else:
        return GREEN


class TrafficDetectionNode:
    def __init__(self):
        rospy.init_node("traffic_detection", anonymous=False)
        self.bridge = CvBridge()

        # =========================
        # params
        # =========================
        self.image_topic = rospy.get_param("~image_topic", "/head_camera/image_raw")
        self.state_topic = rospy.get_param("~state_topic", "/traffic_light/state")

        self.weights = rospy.get_param("~weights", "")
        self.conf_th = float(rospy.get_param("~conf_th", 0.25))
        self.iou_th  = float(rospy.get_param("~iou_th", 0.45))

        self.traffic_light_class_id = int(
            rospy.get_param("~traffic_light_class_id", -1)
        )

        self.min_box_area = int(rospy.get_param("~min_box_area", 12 * 12))

        # overlay params
        self.ov_enable = bool(rospy.get_param("~overlay/enable", True))
        self.ov_topic = rospy.get_param("~overlay/topic", "/traffic_overlay/image")
        self.ov_labels = bool(rospy.get_param("~overlay/draw_labels", True))
        self.ov_thick = int(rospy.get_param("~overlay/thickness", 2))
        self.ov_font = float(rospy.get_param("~overlay/font_scale", 0.7))
        self.ov_show_conf = bool(rospy.get_param("~overlay/show_conf", True))

        # 상태별 BGR 색
        self.ov_color_red = tuple(int(x) for x in rospy.get_param("~overlay/red_bgr", [0, 0, 255]))
        self.ov_color_yellow = tuple(int(x) for x in rospy.get_param("~overlay/yellow_bgr", [0, 255, 255]))
        self.ov_color_green = tuple(int(x) for x in rospy.get_param("~overlay/green_bgr", [0, 255, 0]))
        self.ov_color_unknown = tuple(int(x) for x in rospy.get_param("~overlay/unknown_bgr", [200, 200, 200]))

        if not self.weights:
            rospy.logerr("traffic_detection: ~weights is empty")
            raise RuntimeError("YOLO weights not set")

        rospy.loginfo(f"[traffic_detection] Loading YOLO: {self.weights}")
        self.model = YOLO(self.weights)

        # =========================
        # pub / sub
        # =========================
        self.pub_state = rospy.Publisher(
            self.state_topic, Int16, queue_size=1
        )
        self.pub_overlay = rospy.Publisher(self.ov_topic, Image, queue_size=1) if self.ov_enable else None
        self.sub_img = rospy.Subscriber(
            self.image_topic, Image, self.cb_image,
            queue_size=1, buff_size=2**24
        )

        self.last_pub_state = UNKNOWN

    def _state_to_label_color(self, state: int):
        if state == RED:
            return "RED", self.ov_color_red
        if state == YELLOW:
            return "YELLOW", self.ov_color_yellow
        if state == GREEN:
            return "GREEN", self.ov_color_green
        return "UNKNOWN", self.ov_color_unknown

    def _draw_box(self, img, x1, y1, x2, y2, label_text, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.ov_thick)
        if not self.ov_labels:
            return
        y = max(0, y1 - 6)
        cv2.putText(img, label_text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, self.ov_font, color, 2, cv2.LINE_AA)

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            rospy.logwarn(f"[traffic_detection] cv_bridge error: {e}")
            return

        results = self.model.predict(
            frame, conf=self.conf_th, iou=self.iou_th, verbose=False
        )

        state = UNKNOWN
        best_box = None
        best_conf = None

        if results and len(results) > 0:
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)

            best_score = -1.0

            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = (
                    boxes.conf.cpu().numpy()
                    if boxes.conf is not None
                    else np.ones((xyxy.shape[0],), dtype=np.float32)
                )
                cls = (
                    boxes.cls.cpu().numpy().astype(int)
                    if boxes.cls is not None
                    else np.full((xyxy.shape[0],), -1, dtype=np.int32)
                )

                h, w = frame.shape[:2]

                for i in range(xyxy.shape[0]):
                    if (
                        self.traffic_light_class_id >= 0
                        and cls[i] != self.traffic_light_class_id
                    ):
                        continue

                    x1, y1, x2, y2 = xyxy[i]
                    x1 = clamp(int(x1), 0, w - 1)
                    y1 = clamp(int(y1), 0, h - 1)
                    x2 = clamp(int(x2), 0, w - 1)
                    y2 = clamp(int(y2), 0, h - 1)

                    area = (x2 - x1) * (y2 - y1)
                    if area < self.min_box_area:
                        continue

                    score = float(conf[i])
                    if score > best_score:
                        best_score = score
                        best_box = (x1, y1, x2, y2)
                        best_conf = score

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                roi = frame[y1:y2, x1:x2]
                state = hsv_color_vote(roi)

        # =========================
        # publish (only on change)
        # =========================
        if state != self.last_pub_state:
            self.pub_state.publish(Int16(state))
            self.last_pub_state = state

        # overlay publish (구독자가 있을 때만)
        if self.ov_enable and self.pub_overlay is not None:
            overlay = frame.copy()

            label, color = self._state_to_label_color(state)

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                if self.ov_show_conf and best_conf is not None:
                    txt = f"{label} conf:{best_conf:.2f}"
                else:
                    txt = f"{label}"
                self._draw_box(overlay, x1, y1, x2, y2, txt, color)
            else:
                # 박스가 없을 때도 상태를 화면에 띄우고 싶으면 좌상단 표시
                txt = f"{label} (no bbox)"
                cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, self.ov_font, color, 2, cv2.LINE_AA)

            out_img = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            out_img.header = msg.header
            self.pub_overlay.publish(out_img)

    def spin(self):
        rospy.loginfo("[traffic_detection] started.")
        rospy.spin()


if __name__ == "__main__":
    TrafficDetectionNode().spin()

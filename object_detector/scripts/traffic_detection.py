#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import Int16
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

    # 빨강은 Hue wrap 때문에 2구간
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

    # 최소 비율(너무 약하면 UNKNOWN)
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

        # params
        self.image_topic = rospy.get_param("~image_topic", "/head_camera/image_raw")
        self.state_topic = rospy.get_param("~state_topic", "/traffic_light/state")

        self.weights = rospy.get_param("~weights", "")
        self.conf_th = float(rospy.get_param("~conf_th", 0.25))
        self.iou_th  = float(rospy.get_param("~iou_th", 0.45))

        # traffic light class id: 모르면 -1로 두고 "가장 높은 conf" 박스 1개만 사용
        self.traffic_light_class_id = int(rospy.get_param("~traffic_light_class_id", -1))

        # 박스 면적 필터 (너무 작은 점 검출 방지)
        self.min_box_area = int(rospy.get_param("~min_box_area", 12 * 12))

        if not self.weights:
            rospy.logerr("traffic_detection: ~weights is empty. Set YOLO weights path.")
            raise RuntimeError("weights not set")

        rospy.loginfo(f"[traffic_detection] Loading YOLO: {self.weights}")
        self.model = YOLO(self.weights)

        self.pub_state = rospy.Publisher(self.state_topic, Int16, queue_size=1)
        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1, buff_size=2**24)

        self.last_state = UNKNOWN

    def cb_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"[traffic_detection] cv_bridge error: {e}")
            return

        results = self.model.predict(frame, conf=self.conf_th, iou=self.iou_th, verbose=False)

        state = UNKNOWN

        if results and len(results) > 0:
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)

            best_box = None
            best_score = -1.0

            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
                cls  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.full((xyxy.shape[0],), -1, dtype=np.int32)

                h, w = frame.shape[:2]

                for i in range(xyxy.shape[0]):
                    # class filter
                    if self.traffic_light_class_id >= 0 and cls[i] != self.traffic_light_class_id:
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

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                roi = frame[y1:y2, x1:x2]
                state = hsv_color_vote(roi)

        # 변경시에만 publish(스팸 줄임)
        if state != self.last_state:
            self.pub_state.publish(Int16(state))
            self.last_state = state

    def spin(self):
        rospy.loginfo("[traffic_detection] started.")
        rospy.spin()


if __name__ == "__main__":
    TrafficDetectionNode().spin()

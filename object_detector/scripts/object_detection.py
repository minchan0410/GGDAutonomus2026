#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import threading

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header


class ObjectDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic")
        self.model_path = rospy.get_param("~model_path")
        self.conf_thres = float(rospy.get_param("~conf_thres"))
        self.device = rospy.get_param("~device")

        # ---- only car class ----
        self.car_name = rospy.get_param("~car_class_name")
        self.car_topic = rospy.get_param("~car_topic")

        # 반드시 20Hz로 publish
        self.pub_rate = float(rospy.get_param("~pub_rate", 20.0))
        self.frame_id_fallback = rospy.get_param("~frame_id", "")

        self.ov_enable = bool(rospy.get_param("~overlay/enable", True))
        self.ov_topic = rospy.get_param("~overlay/topic", "/yolo_overlay/image")
        self.ov_labels = bool(rospy.get_param("~overlay/draw_labels", True))
        self.ov_thick = int(rospy.get_param("~overlay/thickness", 2))
        self.ov_font = float(rospy.get_param("~overlay/font_scale", 0.6))
        self.ov_show_conf = bool(rospy.get_param("~overlay/show_conf", True))

        car_color = rospy.get_param("~overlay/car_color_bgr", [0, 255, 0])
        self.ov_car_color = tuple(int(x) for x in car_color)

        self.model = YOLO(self.model_path)

        names = self.model.names
        if not isinstance(names, dict):
            names = {i: n for i, n in enumerate(names)}

        self.car_ids = [i for i, n in names.items() if n == self.car_name]
        self.filter_ids = sorted(list(set(self.car_ids)))

        self.pub_car = rospy.Publisher(self.car_topic, Detection2DArray, queue_size=1)
        self.pub_overlay = rospy.Publisher(self.ov_topic, Image, queue_size=1) if self.ov_enable else None

        # 최신 프레임만 덮어쓰기 (캐시/큐 안 쌓이게)
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_header = None

        self.sub = rospy.Subscriber(self.image_topic, Image, self.cb_img, queue_size=1, buff_size=2**24)

        # 20Hz 고정 루프
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self.on_timer)

        rospy.loginfo(f"[object_detection] model={self.model_path} conf={self.conf_thres} device={self.device}")
        rospy.loginfo(f"[object_detection] car_ids={self.car_ids} filter_ids={self.filter_ids}")
        rospy.loginfo(f"[object_detection] pub_rate={self.pub_rate}Hz")
        rospy.loginfo(f"[object_detection] pub car={self.car_topic} overlay={self.ov_topic if self.ov_enable else 'disabled'}")

    def cb_img(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[object_detection] cv_bridge failed: {e}")
            return

        with self._lock:
            self._latest_frame = frame  # 최신만 유지
            self._latest_header = msg.header

    def make_det(self, header, x1, y1, x2, y2, cls_id, score):
        det = Detection2D()
        det.header = header

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1)
        h = (y2 - y1)

        det.bbox.center.x = float(cx)
        det.bbox.center.y = float(cy)
        det.bbox.center.theta = 0.0
        det.bbox.size_x = float(w)
        det.bbox.size_y = float(h)

        hyp = ObjectHypothesisWithPose()
        hyp.id = int(cls_id)
        hyp.score = float(score)
        det.results.append(hyp)
        return det

    def draw_box(self, img, x1, y1, x2, y2, label, score, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.ov_thick)
        if not self.ov_labels:
            return
        txt = f"{label} {score:.2f}" if self.ov_show_conf else f"{label}"
        y = max(0, y1 - 6)
        cv2.putText(img, txt, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, self.ov_font, color, 1, cv2.LINE_AA)

    def _fallback_header(self):
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.frame_id_fallback
        return h

    def on_timer(self, _evt):
        # 20Hz마다 무조건 publish (객체 없으면 빈 배열)
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            header = self._latest_header if self._latest_header is not None else self._fallback_header()

        car_arr = Detection2DArray()
        car_arr.header = header

        need_overlay = (
            self.ov_enable and
            (self.pub_overlay is not None) and
            (self.pub_overlay.get_num_connections() > 0) and
            (frame is not None)
        )
        overlay = frame.copy() if need_overlay else None

        if frame is None:
            # 아직 이미지 못 받았어도 20Hz로 빈 토픽 계속 송신
            self.pub_car.publish(car_arr)
            return

        H, W = frame.shape[:2]

        try:
            pred = self.model.predict(
                source=frame,
                conf=self.conf_thres,
                device=self.device,
                classes=self.filter_ids if len(self.filter_ids) > 0 else None,
                verbose=False
            )[0]
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[object_detection] YOLO predict failed: {e}")
            # 실패해도 20Hz 유지: 빈 배열 publish
            self.pub_car.publish(car_arr)
            return

        boxes = pred.boxes
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), p, c in zip(xyxy, conf, cls):
                x1 = int(np.clip(x1, 0, W - 1))
                x2 = int(np.clip(x2, 0, W - 1))
                y1 = int(np.clip(y1, 0, H - 1))
                y2 = int(np.clip(y2, 0, H - 1))

                det = self.make_det(header, x1, y1, x2, y2, c, p)

                if c in self.car_ids:
                    car_arr.detections.append(det)
                    if need_overlay:
                        self.draw_box(overlay, x1, y1, x2, y2, self.car_name, p, self.ov_car_color)

        # 핵심: 매 틱마다 무조건 publish
        self.pub_car.publish(car_arr)

        if need_overlay:
            out_img = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            out_img.header = header
            self.pub_overlay.publish(out_img)


if __name__ == "__main__":
    rospy.init_node("object_detection")
    ObjectDetectionNode()
    rospy.spin()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading

import rospy
from cv_bridge import CvBridge

import cv2

from sensor_msgs.msg import Image, CompressedImage


class ImageClickSaver:
    def __init__(self):
        rospy.init_node("image_click_saver", anonymous=False)

        # ---- params ----
        self.image_topic = rospy.get_param("~image_topic", "/cam1/usb_cam/image_raw")
        self.compressed = bool(rospy.get_param("~compressed", False))  # True면 CompressedImage 구독
        self.save_dir = rospy.get_param("~save_dir", "/media/vic/ESD-USB/image_cap")             # 저장 폴더 (기본: 현재폴더)
        self.prefix = rospy.get_param("~prefix", "cap")               # 파일 prefix
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95)) # 0~100
        self.window_name = rospy.get_param("~window_name", "ImageClickSaver")

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_bgr = None
        self.counter = 0
        self.last_save_time = 0.0

        # ---- subscribers ----
        if self.compressed:
            self.sub = rospy.Subscriber(self.image_topic, CompressedImage, self.cb_compressed, queue_size=1)
            rospy.loginfo(f"[image_click_saver] Subscribing CompressedImage: {self.image_topic}")
        else:
            self.sub = rospy.Subscriber(self.image_topic, Image, self.cb_raw, queue_size=1)
            rospy.loginfo(f"[image_click_saver] Subscribing Image: {self.image_topic}")

        # ---- OpenCV window ----
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        rospy.loginfo(f"[image_click_saver] save_dir={os.path.abspath(self.save_dir)} prefix={self.prefix}")

    def cb_raw(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest_bgr = bgr
        except Exception as e:
            rospy.logwarn(f"[image_click_saver] raw convert failed: {e}")

    def cb_compressed(self, msg: CompressedImage):
        try:
            # msg.data: uint8[]
            import numpy as np
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imdecode returned None")
            with self.lock:
                self.latest_bgr = bgr
        except Exception as e:
            rospy.logwarn(f"[image_click_saver] compressed decode failed: {e}")

    def unique_path(self):
        # 나노초까지 넣어서 거의 절대 안 겹치게 + counter로 완전 보장
        ts = time.strftime("%Y%m%d_%H%M%S")
        ns = int(time.time_ns() % 1_000_000_000)
        self.counter += 1
        fname = f"{self.prefix}_{ts}_{ns:09d}_{self.counter:04d}.jpg"
        return os.path.join(self.save_dir, fname)

    def save_latest(self):
        with self.lock:
            if self.latest_bgr is None:
                rospy.logwarn("[image_click_saver] No image received yet.")
                return
            img = self.latest_bgr.copy()

        out_path = self.unique_path()
        ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])

        if ok:
            rospy.loginfo(f"[image_click_saver] Saved: {out_path}")
        else:
            rospy.logerr(f"[image_click_saver] Failed to save: {out_path}")

    def on_mouse(self, event, x, y, flags, param):
        # 좌클릭 시 저장 (너무 연속 클릭 방지용 0.15초 쿨다운)
        if event == cv2.EVENT_LBUTTONDOWN:
            now = time.time()
            if now - self.last_save_time < 0.15:
                return
            self.last_save_time = now
            self.save_latest()

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                img = None if self.latest_bgr is None else self.latest_bgr.copy()

            if img is not None:
                cv2.imshow(self.window_name, img)

            # 키 입력 처리 (q 또는 ESC로 종료)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                rospy.signal_shutdown("User quit")

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        node = ImageClickSaver()
        node.spin()
    except rospy.ROSInterruptException:
        pass

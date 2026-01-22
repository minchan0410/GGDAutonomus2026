#!/usr/bin/env python3
import rospy
import math
import threading

from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header


class ObjectProjectionNode:
    def __init__(self):
        # topics
        self.sub_car_topic = rospy.get_param("~sub_car_topic")
        self.pub_pt_topic = rospy.get_param("~pub_car_projected_topic")
        self.pub_mk_topic = rospy.get_param("~pub_markers_topic")

        # behavior
        self.frame_id = rospy.get_param("~frame_id", "")
        self.select_nearest = bool(rospy.get_param("~select_nearest", True))
        self.pub_rate = float(rospy.get_param("~pub_rate", 20.0))  # 반드시 20Hz

        # camera intrinsics/extrinsics (simple ground projection)
        self.fx = float(rospy.get_param("~camera/fx"))
        self.fy = float(rospy.get_param("~camera/fy"))
        self.cx = float(rospy.get_param("~camera/cx"))
        self.cy = float(rospy.get_param("~camera/cy"))
        self.h = float(rospy.get_param("~camera/height"))

        # pitch convention: degrees in yaml, up=+, down=-
        # backward compatible: if pitch_deg not set, use ~camera/pitch (radians)
        if rospy.has_param("~camera/pitch_deg"):
            self.pitch_rad = math.radians(float(rospy.get_param("~camera/pitch_deg")))
        else:
            self.pitch_rad = float(rospy.get_param("~camera/pitch", 0.0))

        # marker config (RViz visualization)
        self.mk_enable = bool(rospy.get_param("~marker/enable", True))
        self.mk_ns = rospy.get_param("~marker/ns", "car_projected")
        self.mk_id = int(rospy.get_param("~marker/id", 0))
        self.mk_type = str(rospy.get_param("~marker/type", "sphere")).lower()
        self.mk_scale = float(rospy.get_param("~marker/scale", 0.4))
        self.mk_life = float(rospy.get_param("~marker/lifetime", 0.0))  # 0=forever (DELETE로 지움)
        rgba = rospy.get_param("~marker/color_rgba", [0.0, 1.0, 0.0, 1.0])
        self.mk_color = tuple(float(x) for x in rgba)
        self.mk_z = float(rospy.get_param("~marker/z", 0.0))

        # ROS I/O
        self.sub = rospy.Subscriber(self.sub_car_topic, Detection2DArray, self.cb, queue_size=1)
        self.pub_pt = rospy.Publisher(self.pub_pt_topic, PointStamped, queue_size=1)
        self.pub_mk = rospy.Publisher(self.pub_mk_topic, Marker, queue_size=1) if self.mk_enable else None

        # keep latest message only
        self._lock = threading.Lock()
        self._latest_msg = None

        # 20Hz loop
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self.on_timer)

        rospy.loginfo(
            f"[object_projection] sub={self.sub_car_topic} pub_pt={self.pub_pt_topic} "
            f"pub_mk={self.pub_mk_topic if self.mk_enable else 'disabled'} pub_rate={self.pub_rate}Hz "
            f"pitch_rad={self.pitch_rad:.4f} (yaml pitch_deg: up=+, down=-)"
        )

    def cb(self, msg: Detection2DArray):
        with self._lock:
            self._latest_msg = msg

    def project(self, u, v):
        # u,v : pixel coordinates
        beta = math.atan((u - self.cx) / self.fx)

        # image-based downward angle (v grows downward)
        alpha_img = math.atan((v - self.cy) / self.fy)

        # pitch convention: up=+, down=-  => downward angle decreases when pitch is positive
        alpha = alpha_img - self.pitch_rad

        if alpha <= 1e-6:
            return None

        X = self.h / math.tan(alpha)
        if X <= 0:
            return None

        Y_right = X * math.tan(beta)
        Y_left = -Y_right
        return X, Y_left

    def _fallback_header(self):
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.frame_id
        return h

    def _marker_type(self):
        if self.mk_type == "cube":
            return Marker.CUBE
        if self.mk_type == "arrow":
            return Marker.ARROW
        return Marker.SPHERE

    def _base_marker(self, header):
        m = Marker()
        m.header = header
        m.header.frame_id = self.frame_id
        m.ns = self.mk_ns
        m.id = self.mk_id
        m.type = self._marker_type()
        m.pose.orientation.w = 1.0
        m.scale.x = self.mk_scale
        m.scale.y = self.mk_scale
        m.scale.z = self.mk_scale
        m.color.r = self.mk_color[0]
        m.color.g = self.mk_color[1]
        m.color.b = self.mk_color[2]
        m.color.a = self.mk_color[3]
        m.lifetime = rospy.Duration(self.mk_life)
        return m

    def on_timer(self, _evt):
        with self._lock:
            msg = self._latest_msg

        # 항상 20Hz로 publish해야 하니 header도 항상 생성/갱신
        header = msg.header if msg is not None else self._fallback_header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id  # 프레임 통일

        best = None  # (X, Y)

        if msg is not None:
            for det in msg.detections:
                u = det.bbox.center.x
                v = det.bbox.center.y + det.bbox.size_y * 0.5  # bbox 하단(대략 바닥점)
                xy = self.project(u, v)
                if xy is None:
                    continue
                if best is None:
                    best = xy
                elif self.select_nearest and xy[0] < best[0]:
                    best = xy

        # --- PointStamped: 항상 publish (없으면 NaN) ---
        pt = PointStamped()
        pt.header = header
        if best is None:
            nan = float("nan")
            pt.point.x = nan
            pt.point.y = nan
            pt.point.z = nan
        else:
            pt.point.x = float(best[0])
            pt.point.y = float(best[1])
            pt.point.z = 0.0
        self.pub_pt.publish(pt)

        # --- Marker: 시각화용 (없으면 DELETE로 RViz에서 사라지게) ---
        if self.mk_enable and (self.pub_mk is not None):
            m = self._base_marker(header)
            if best is None:
                m.action = Marker.DELETE
                self.pub_mk.publish(m)
            else:
                m.action = Marker.ADD
                m.pose.position.x = pt.point.x
                m.pose.position.y = pt.point.y
                m.pose.position.z = float(self.mk_z)
                self.pub_mk.publish(m)


if __name__ == "__main__":
    rospy.init_node("object_projection")
    ObjectProjectionNode()
    rospy.spin()

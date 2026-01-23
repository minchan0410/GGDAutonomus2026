#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String, Bool, Int16
from visualization_msgs.msg import Marker, MarkerArray
from jsk_rviz_plugins.msg import OverlayText

# NOTE: ColorRGBA is inside std_msgs, but OverlayText uses it as a field type
from std_msgs.msg import ColorRGBA


def _rgba_msg(r, g, b, a):
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


class FinalPlannerRviz:
    """
    RViz publishers:
      1) MarkerArray:
         - ROI thin box (green -> orange on yolo crash, orange latched during lane_change)
         - Ultrasonic threshold bar (green -> orange on sonic crash, orange latched during lane_change)
         - TEXT_VIEW_FACING (legacy; can be disabled)
      2) JSK OverlayText (HUD):
         - fixed on RViz screen (top-left), shows state/reason/crash flags
    """

    def __init__(self):
        rospy.init_node("final_planner_rviz", anonymous=False)

        # ---- params ----
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.marker_topic = rospy.get_param("~marker_topic", "/final_planner/markers")

        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ultrasonic_frame = rospy.get_param("~ultrasonic_frame", "ultrasonic_1_link")

        # ROI params (shared with planner)
        self.roi_min_x = float(rospy.get_param("planner_common/roi/min_x", 0.0))
        self.roi_max_x = float(rospy.get_param("planner_common/roi/max_x", 0.5))
        self.roi_min_y = float(rospy.get_param("planner_common/roi/min_y", -0.4))
        self.roi_max_y = float(rospy.get_param("planner_common/roi/max_y", 0.4))
        self.roi_offset_x = float(rospy.get_param("planner_common/roi/offset_x", 0.74))
        self.roi_z = float(rospy.get_param("~roi/z", 0.05))
        self.roi_thickness = float(rospy.get_param("~roi/thickness", 0.02))
        self.roi_alpha = float(rospy.get_param("~roi/alpha", 0.25))

        # Ultrasonic bar params (shared with planner)
        self.us_threshold = float(rospy.get_param("planner_common/ultrasonic/threshold", 300.0))
        self.us_unit_scale = float(rospy.get_param("planner_common/ultrasonic/unit_scale", 0.001))  # 300(mm)->0.3(m)
        self.us_bar_y = float(rospy.get_param("~ultrasonic/bar_y", 0.03))
        self.us_bar_z = float(rospy.get_param("~ultrasonic/bar_z", 0.03))
        self.us_z_offset = float(rospy.get_param("~ultrasonic/z_offset", 0.0))
        self.us_alpha = float(rospy.get_param("~ultrasonic/alpha", 0.8))

        # Colors
        green = rospy.get_param("~colors/green", [0.0, 1.0, 0.0])
        orange = rospy.get_param("~colors/orange", [1.0, 0.5, 0.0])
        self.col_green = (float(green[0]), float(green[1]), float(green[2]))
        self.col_orange = (float(orange[0]), float(orange[1]), float(orange[2]))

        # ---- legacy 3D text marker (optional) ----
        self.enable_3d_text = bool(rospy.get_param("~text3d/enable", False))
        self.text_x = float(rospy.get_param("~text3d/x", 0.0))
        self.text_y = float(rospy.get_param("~text3d/y", 0.0))
        self.text_z = float(rospy.get_param("~text3d/z", 1.2))
        self.text_scale = float(rospy.get_param("~text3d/scale", 0.18))
        textc = rospy.get_param("~text3d/color", [1.0, 1.0, 1.0])
        self.col_text3d = (float(textc[0]), float(textc[1]), float(textc[2]))

        # ---- HUD (OverlayText) params ----
        self.hud_enable = bool(rospy.get_param("~hud/enable", True))
        self.hud_topic = rospy.get_param("~hud/topic", "/final_planner/hud")

        # Overlay position: top-left fixed (pixels)
        self.hud_left = int(rospy.get_param("~hud/left", 10))
        self.hud_top = int(rospy.get_param("~hud/top", 10))
        # You can set width/height if you want; keep 0 to auto-ish (depends on plugin)
        self.hud_width = int(rospy.get_param("~hud/width", 450))
        self.hud_height = int(rospy.get_param("~hud/height", 120))

        # Text style
        self.hud_text_size = int(rospy.get_param("~hud/text_size", 18))
        self.hud_line_width = int(rospy.get_param("~hud/line_width", 2))

        # Foreground / background color
        hud_fg = rospy.get_param("~hud/fg", [1.0, 1.0, 1.0, 1.0])        # white
        hud_bg = rospy.get_param("~hud/bg", [0.0, 0.0, 0.0, 0.55])       # black, semi-transparent
        self.hud_fg = _rgba_msg(*hud_fg)
        self.hud_bg = _rgba_msg(*hud_bg)

        # ---- state inputs ----
        self.state = "lane_driving"
        self.yolo_crash = False
        self.sonic_crash = False
        self.reason = "none"

        # lane-change latch for orange 유지
        self.prev_state = None
        self.latched_yolo = False
        self.latched_sonic = False

        # optional raw ultrasonic display
        self.ultrasonic_raw = None

        # ---- ros io ----
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        if self.hud_enable:
            self.hud_pub = rospy.Publisher(self.hud_topic, OverlayText, queue_size=1)
        else:
            self.hud_pub = None

        rospy.Subscriber("/final_planner/state", String, self.cb_state, queue_size=1)
        rospy.Subscriber("/final_planner/yolo_crash", Bool, self.cb_yolo, queue_size=1)
        rospy.Subscriber("/final_planner/sonic_crash", Bool, self.cb_sonic, queue_size=1)
        rospy.Subscriber("/final_planner/lane_change_reason", String, self.cb_reason, queue_size=1)
        rospy.Subscriber("/ultrasonic1", Int16, self.cb_ultrasonic_raw, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.on_timer)
        rospy.loginfo("[final_planner_rviz] rate=%.1f marker=%s hud=%s base=%s us_frame=%s",
                      self.publish_rate, self.marker_topic, self.hud_topic if self.hud_enable else "(disabled)",
                      self.base_frame, self.ultrasonic_frame)

    # ---- callbacks ----
    def cb_state(self, msg: String):
        self.state = msg.data.strip()

    def cb_yolo(self, msg: Bool):
        self.yolo_crash = bool(msg.data)

    def cb_sonic(self, msg: Bool):
        self.sonic_crash = bool(msg.data)

    def cb_reason(self, msg: String):
        self.reason = msg.data.strip()

    def cb_ultrasonic_raw(self, msg: Int16):
        self.ultrasonic_raw = int(msg.data)

    # ---- internal ----
    @staticmethod
    def _is_lane_change_state(s: str) -> bool:
        return s in ("lane_change_to_left", "lane_change_to_right")

    def _update_latch(self):
        cur = self.state
        prev = self.prev_state
        lane_change_now = self._is_lane_change_state(cur)
        lane_change_prev = self._is_lane_change_state(prev) if prev is not None else False

        # enter lane change: latch current flags
        if lane_change_now and not lane_change_prev:
            self.latched_yolo = bool(self.yolo_crash)
            self.latched_sonic = bool(self.sonic_crash)

        # during lane change: keep OR-ing
        if lane_change_now:
            self.latched_yolo = self.latched_yolo or bool(self.yolo_crash)
            self.latched_sonic = self.latched_sonic or bool(self.sonic_crash)

        # exit lane change -> lane_driving: reset latch
        if (not lane_change_now) and lane_change_prev:
            self.latched_yolo = False
            self.latched_sonic = False

        self.prev_state = cur

    def _make_marker(self, frame_id: str, ns: str, mid: int, mtype: int) -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = int(mid)
        m.type = int(mtype)
        m.action = Marker.ADD
        m.lifetime = rospy.Duration(0)
        m.pose.orientation.w = 1.0
        return m

    def _roi_marker(self, orange: bool) -> Marker:
        m = self._make_marker(self.base_frame, "final_planner", 0, Marker.CUBE)

        cx = self.roi_offset_x + (self.roi_min_x + self.roi_max_x) * 0.5
        cy = (self.roi_min_y + self.roi_max_y) * 0.5
        cz = self.roi_z

        m.pose.position.x = cx
        m.pose.position.y = cy
        m.pose.position.z = cz

        m.scale.x = (self.roi_max_x - self.roi_min_x)
        m.scale.y = (self.roi_max_y - self.roi_min_y)
        m.scale.z = self.roi_thickness

        r, g, b = self.col_orange if orange else self.col_green
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(self.roi_alpha)
        return m

    def _ultrasonic_bar_marker(self, orange: bool) -> Marker:
        m = self._make_marker(self.ultrasonic_frame, "final_planner", 1, Marker.CUBE)

        length_m = max(0.0, self.us_threshold * self.us_unit_scale)

        # bar center at +x length/2 in ultrasonic frame
        m.pose.position.x = length_m * 0.5
        m.pose.position.y = 0.0
        m.pose.position.z = self.us_z_offset

        m.scale.x = length_m
        m.scale.y = self.us_bar_y
        m.scale.z = self.us_bar_z

        r, g, b = self.col_orange if orange else self.col_green
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(self.us_alpha)
        return m

    def _text3d_marker(self) -> Marker:
        m = self._make_marker(self.base_frame, "final_planner", 2, Marker.TEXT_VIEW_FACING)
        m.pose.position.x = self.text_x
        m.pose.position.y = self.text_y
        m.pose.position.z = self.text_z
        m.scale.z = self.text_scale

        r, g, b = self.col_text3d
        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = 1.0

        lane_change = self._is_lane_change_state(self.state)
        y_for_text = self.latched_yolo if lane_change else self.yolo_crash
        s_for_text = self.latched_sonic if lane_change else self.sonic_crash
        reason = self.reason if lane_change else "none"

        us_raw_str = ""
        if self.ultrasonic_raw is not None:
            us_raw_str = f"\nultrasonic1: {self.ultrasonic_raw}"

        m.text = (
            f"state: {self.state}"
            f"\nLC_reason: {reason}"
            f"\nyolo_crash: {bool(y_for_text)}"
            f"\nsonic_crash: {bool(s_for_text)}"
            f"{us_raw_str}"
        )
        return m

    def _hud_msg(self) -> OverlayText:
        hud = OverlayText()
        hud.width = self.hud_width
        hud.height = self.hud_height
        hud.left = self.hud_left
        hud.top = self.hud_top
        hud.text_size = self.hud_text_size
        hud.line_width = self.hud_line_width

        # Colors
        hud.fg_color = self.hud_fg
        hud.bg_color = self.hud_bg

        lane_change = self._is_lane_change_state(self.state)
        y_show = self.latched_yolo if lane_change else self.yolo_crash
        s_show = self.latched_sonic if lane_change else self.sonic_crash
        reason = self.reason if lane_change else "none"

        us_raw_str = ""
        if self.ultrasonic_raw is not None:
            us_raw_str = f"\nultrasonic1: {self.ultrasonic_raw}"

        hud.text = (
            f"state: {self.state}\n"
            f"LC_reason: {reason}\n"
            f"yolo_crash: {bool(y_show)}\n"
            f"sonic_crash: {bool(s_show)}"
            f"{us_raw_str}"
        )
        return hud

    def on_timer(self, _evt):
        self._update_latch()

        lane_change = self._is_lane_change_state(self.state)

        # 색 규칙:
        # - lane_driving: 현재 crash True면 주황
        # - lane_change_*: lane change 동안 latch된 값이 True면 주황 유지
        roi_orange = (self.latched_yolo if lane_change else self.yolo_crash)
        us_orange = (self.latched_sonic if lane_change else self.sonic_crash)

        arr = MarkerArray()
        arr.markers.append(self._roi_marker(orange=roi_orange))
        arr.markers.append(self._ultrasonic_bar_marker(orange=us_orange))

        if self.enable_3d_text:
            arr.markers.append(self._text3d_marker())

        self.marker_pub.publish(arr)

        if self.hud_enable and self.hud_pub is not None:
            self.hud_pub.publish(self._hud_msg())


if __name__ == "__main__":
    try:
        FinalPlannerRviz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

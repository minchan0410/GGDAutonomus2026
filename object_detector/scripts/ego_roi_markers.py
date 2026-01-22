#!/usr/bin/env python3
import rospy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class EgoAndRoiMarkers:
    def __init__(self):
        # topics / behavior
        self.pub_topic = rospy.get_param("~pub_markers_topic", "/viz/ego_roi_markers")
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.pub_rate = float(rospy.get_param("~pub_rate", 20.0))

        # --- car box params (meters) ---
        self.car_length = float(rospy.get_param("~car/length", 1.60))
        self.car_width  = float(rospy.get_param("~car/width", 0.90))
        self.car_height = float(rospy.get_param("~car/height", 0.55))
        # car pose in frame_id (usually base_link origin). If your base_link is at rear axle center,
        # you may want to shift x forward by length/2.
        self.car_x = float(rospy.get_param("~car/x", 0.0))
        self.car_y = float(rospy.get_param("~car/y", 0.0))
        self.car_z = float(rospy.get_param("~car/z", self.car_height * 0.5))

        # --- ROI rectangle params (meters) ---
        # Define ROI in vehicle frame: x in [x_min, x_max], y in [y_min, y_max] (y left +)
        self.roi_x_min = float(rospy.get_param("~roi/x_min", 0.0))
        self.roi_x_max = float(rospy.get_param("~roi/x_max", 8.0))
        self.roi_y_min = float(rospy.get_param("~roi/y_min", -1.5))
        self.roi_y_max = float(rospy.get_param("~roi/y_max",  1.5))

        # ROI “height=0” requested -> make it very thin so RViz can render it
        self.roi_thickness = float(rospy.get_param("~roi/thickness", 0.01))
        self.roi_z = float(rospy.get_param("~roi/z", self.roi_thickness * 0.5))

        # Colors (RGBA)
        self.car_rgba = tuple(float(x) for x in rospy.get_param("~car/color_rgba", [0.2, 0.7, 1.0, 0.6]))
        self.roi_rgba = tuple(float(x) for x in rospy.get_param("~roi/color_rgba", [1.0, 1.0, 0.0, 0.25]))

        # Optional: axes marker
        self.axes_enable = bool(rospy.get_param("~axes/enable", True))
        self.axes_scale = float(rospy.get_param("~axes/scale", 1.0))

        self.pub = rospy.Publisher(self.pub_topic, MarkerArray, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self.on_timer)

        rospy.loginfo(f"[ego_roi_markers] pub={self.pub_topic} frame_id={self.frame_id} rate={self.pub_rate}Hz")

    def _header(self):
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.frame_id
        return h

    def _base_marker(self, header, ns, mid, mtype):
        m = Marker()
        m.header = header
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = rospy.Duration(0.0)  # forever
        return m

    def _set_color(self, m, rgba):
        m.color.r, m.color.g, m.color.b, m.color.a = rgba

    def _car_marker(self, header):
        m = self._base_marker(header, "ego", 0, Marker.CUBE)
        m.pose.position.x = self.car_x
        m.pose.position.y = self.car_y
        m.pose.position.z = self.car_z
        m.scale.x = self.car_length
        m.scale.y = self.car_width
        m.scale.z = self.car_height
        self._set_color(m, self.car_rgba)
        return m

    def _roi_marker(self, header):
        m = self._base_marker(header, "roi", 0, Marker.CUBE)

        roi_len = max(1e-6, self.roi_x_max - self.roi_x_min)
        roi_wid = max(1e-6, self.roi_y_max - self.roi_y_min)

        m.pose.position.x = 0.5 * (self.roi_x_min + self.roi_x_max)
        m.pose.position.y = 0.5 * (self.roi_y_min + self.roi_y_max)
        m.pose.position.z = self.roi_z

        m.scale.x = roi_len
        m.scale.y = roi_wid
        m.scale.z = max(1e-6, self.roi_thickness)  # thin “floor”
        self._set_color(m, self.roi_rgba)
        return m

    def _axes_marker(self, header):
        # RViz has Marker.ARROW but 3-axis is easier using LINE_LIST
        m = self._base_marker(header, "axes", 0, Marker.LINE_LIST)
        m.scale.x = 0.05  # line width
        m.color.a = 1.0

        s = self.axes_scale
        # Points are paired: (start,end) for each axis
        # X axis (red)
        m.points.append(self._pt(0, 0, 0))
        m.points.append(self._pt(s, 0, 0))
        m.colors.append(self._col(1, 0, 0, 1))
        m.colors.append(self._col(1, 0, 0, 1))
        # Y axis (green) - left
        m.points.append(self._pt(0, 0, 0))
        m.points.append(self._pt(0, s, 0))
        m.colors.append(self._col(0, 1, 0, 1))
        m.colors.append(self._col(0, 1, 0, 1))
        # Z axis (blue)
        m.points.append(self._pt(0, 0, 0))
        m.points.append(self._pt(0, 0, s))
        m.colors.append(self._col(0, 0, 1, 1))
        m.colors.append(self._col(0, 0, 1, 1))
        return m

    def _pt(self, x, y, z):
        from geometry_msgs.msg import Point
        p = Point()
        p.x, p.y, p.z = float(x), float(y), float(z)
        return p

    def _col(self, r, g, b, a):
        from std_msgs.msg import ColorRGBA
        c = ColorRGBA()
        c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
        return c

    def on_timer(self, _evt):
        header = self._header()
        arr = MarkerArray()

        arr.markers.append(self._car_marker(header))
        arr.markers.append(self._roi_marker(header))

        if self.axes_enable:
            arr.markers.append(self._axes_marker(header))

        self.pub.publish(arr)


if __name__ == "__main__":
    rospy.init_node("ego_roi_markers")
    EgoAndRoiMarkers()
    rospy.spin()

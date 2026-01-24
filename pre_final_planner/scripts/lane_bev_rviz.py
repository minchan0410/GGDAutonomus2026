#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy

from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid


def rot_x(roll):
    c, s = math.cos(roll), math.sin(roll)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float64)


def rot_y(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    # base_link: x forward, y left, z up
    # positive pitch -> forward 방향이 -z로 내려감(=카메라가 아래를 본다)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c],
    ], dtype=np.float64)


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float64)


class LaneBEVRviz:
    """
    Subscribe:
      - /lane_lines_px  (Int32MultiArray) [lx1,ly1,lx2,ly2, rx1,ry1,rx2,ry2]   (pixels)
      - /lane_target_px (PointStamped) point.x=u, point.y=v                    (pixels, optional)

    Publish:
      - /lane_bev/grid  (OccupancyGrid) in frame_id (default base_link)

    Assumptions:
      - base_link axes: x forward, y left, z up
      - camera optical axes: x right, y down, z forward (OpenCV/ROS optical convention)
      - ground plane: z = 0 in base_link
      - camera pose given relative to base_link by (cam_x, cam_y, cam_z) and roll/pitch/yaw (deg)
        * pitch_deg: positive => camera looks downward (toward -z), per rot_y definition above.
    """

    def __init__(self):
        rospy.init_node("lane_bev_rviz", anonymous=False)

        # -------- topics --------
        self.lines_topic = rospy.get_param("~lines_topic", "/lane_lines_px")
        self.target_topic = rospy.get_param("~target_topic", "/lane_target_px")
        self.pub_topic = rospy.get_param("~grid_topic", "/lane_bev/grid")

        # -------- camera intrinsics --------
        self.fx = float(rospy.get_param("~fx"))
        self.fy = float(rospy.get_param("~fy"))
        self.cx = float(rospy.get_param("~cx"))
        self.cy = float(rospy.get_param("~cy"))

        # -------- camera extrinsics (base_link) --------
        self.cam_x = float(rospy.get_param("~cam_x_m", 0.0))
        self.cam_y = float(rospy.get_param("~cam_y_m", 0.0))
        self.cam_z = float(rospy.get_param("~cam_height_m"))  # height above ground (z=0)

        self.roll = math.radians(float(rospy.get_param("~cam_roll_deg", 0.0)))
        self.pitch = math.radians(float(rospy.get_param("~cam_pitch_deg")))
        self.yaw = math.radians(float(rospy.get_param("~cam_yaw_deg", 0.0)))

        # -------- occupancy grid params --------
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.res = float(rospy.get_param("~resolution", 0.05))
        self.grid_w = int(rospy.get_param("~grid_w", 400))
        self.grid_h = int(rospy.get_param("~grid_h", 400))
        self.origin_x = float(rospy.get_param("~origin_x", 0.0))
        self.origin_y = float(rospy.get_param("~origin_y", -(self.grid_w * self.res) / 2.0))

        # -------- drawing / sampling --------
        self.sample_n = int(rospy.get_param("~sample_n", 30))
        self.target_radius_cells = int(rospy.get_param("~target_radius_cells", 2))
        self.max_range_x = float(rospy.get_param("~max_range_x", self.grid_h * self.res))  # forward limit
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))

        # -------- state --------
        self.last_lines = None  # 8 ints
        self.last_target = None  # (u, v)

        # -------- precompute matrices --------
        # Optical (x right, y down, z forward) -> camera_link-like (x forward, y left, z up)
        # x_fwd = z_opt
        # y_left = -x_opt
        # z_up = -y_opt
        self.M_opt_to_cam = np.array([
            [0,  0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=np.float64)

        # Rotation from camera-aligned frame to base_link
        self.R_cam_to_base = rot_z(self.yaw) @ rot_y(self.pitch) @ rot_x(self.roll)

        self.t_base = np.array([self.cam_x, self.cam_y, self.cam_z], dtype=np.float64)

        # -------- ROS I/O --------
        self.sub_lines = rospy.Subscriber(self.lines_topic, Int32MultiArray, self.cb_lines, queue_size=1)
        self.sub_target = rospy.Subscriber(self.target_topic, PointStamped, self.cb_target, queue_size=1)

        self.pub_grid = rospy.Publisher(self.pub_topic, OccupancyGrid, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / max(1e-3, self.publish_rate)), self.on_timer)

        rospy.loginfo("[lane_bev_rviz] sub_lines=%s sub_target=%s pub=%s frame=%s",
                      self.lines_topic, self.target_topic, self.pub_topic, self.frame_id)
        rospy.loginfo("[lane_bev_rviz] intrinsics fx=%.3f fy=%.3f cx=%.3f cy=%.3f", self.fx, self.fy, self.cx, self.cy)
        rospy.loginfo("[lane_bev_rviz] extrinsics cam=(%.2f,%.2f,%.2f)m rpy=(%.1f,%.1f,%.1f)deg",
                      self.cam_x, self.cam_y, self.cam_z,
                      math.degrees(self.roll), math.degrees(self.pitch), math.degrees(self.yaw))
        rospy.loginfo("[lane_bev_rviz] grid %dx%d res=%.3f origin=(%.2f,%.2f)",
                      self.grid_w, self.grid_h, self.res, self.origin_x, self.origin_y)

    def cb_lines(self, msg: Int32MultiArray):
        if len(msg.data) >= 8:
            self.last_lines = list(msg.data[:8])

    def cb_target(self, msg: PointStamped):
        # point.x=u, point.y=v (pixels)
        self.last_target = (float(msg.point.x), float(msg.point.y))

    def pixel_to_ground_xy(self, u: float, v: float):
        """
        Pixel (u,v) -> ground intersection (X,Y) in base_link.
        Returns None if ray doesn't intersect ground in front.
        """
        # ray in optical camera coords
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        ray_opt = np.array([x, y, 1.0], dtype=np.float64)

        # optical -> camera-aligned
        ray_cam = self.M_opt_to_cam @ ray_opt  # (x fwd, y left, z up) in camera-aligned frame

        # camera-aligned -> base_link
        dir_base = self.R_cam_to_base @ ray_cam

        dz = dir_base[2]
        if abs(dz) < 1e-9:
            return None

        s = -self.t_base[2] / dz  # intersect z=0
        if s <= 0:
            return None

        p = self.t_base + s * dir_base  # (X,Y,0)

        # forward range gating (optional)
        if p[0] < 0 or p[0] > self.max_range_x:
            return None

        return (float(p[0]), float(p[1]))

    def xy_to_cell(self, X: float, Y: float):
        ix = int((X - self.origin_x) / self.res)
        iy = int((Y - self.origin_y) / self.res)
        if 0 <= ix < self.grid_w and 0 <= iy < self.grid_h:
            return ix, iy
        return None

    def stamp_disk(self, grid: np.ndarray, ix: int, iy: int, r: int, val: int = 100):
        # grid shape: (H, W) with indexing [iy, ix]
        for dy in range(-r, r + 1):
            yy = iy + dy
            if yy < 0 or yy >= self.grid_h:
                continue
            for dx in range(-r, r + 1):
                xx = ix + dx
                if xx < 0 or xx >= self.grid_w:
                    continue
                if dx*dx + dy*dy <= r*r:
                    grid[yy, xx] = val

    def draw_segment(self, grid: np.ndarray, x1, y1, x2, y2):
        # if invalid
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            return

        # sample along pixel segment
        n = max(2, self.sample_n)
        for i in range(n):
            t = i / float(n - 1)
            u = x1 * (1 - t) + x2 * t
            v = y1 * (1 - t) + y2 * t
            xy = self.pixel_to_ground_xy(u, v)
            if xy is None:
                continue
            cell = self.xy_to_cell(xy[0], xy[1])
            if cell is None:
                continue
            grid[cell[1], cell[0]] = 100

    def on_timer(self, _evt):
        if self.last_lines is None and self.last_target is None:
            return

        # build occupancy grid array (int8: 0..100, -1 unknown)
        occ = np.zeros((self.grid_h, self.grid_w), dtype=np.int8)

        if self.last_lines is not None:
            lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2 = self.last_lines
            self.draw_segment(occ, lx1, ly1, lx2, ly2)
            self.draw_segment(occ, rx1, ry1, rx2, ry2)

        if self.last_target is not None:
            u, v = self.last_target
            xy = self.pixel_to_ground_xy(u, v)
            if xy is not None:
                cell = self.xy_to_cell(xy[0], xy[1])
                if cell is not None:
                    self.stamp_disk(occ, cell[0], cell[1], self.target_radius_cells, 100)

        # publish OccupancyGrid
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = self.frame_id

        grid_msg.info.resolution = self.res
        grid_msg.info.width = self.grid_w
        grid_msg.info.height = self.grid_h
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # flatten row-major
        grid_msg.data = occ.flatten(order="C").tolist()

        self.pub_grid.publish(grid_msg)


if __name__ == "__main__":
    LaneBEVRviz()
    rospy.spin()

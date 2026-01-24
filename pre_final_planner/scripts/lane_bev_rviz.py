#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy

from std_msgs.msg import Int16, Int32MultiArray
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker


def rot_x(roll):
    # X축(전방) 기준 회전 행렬
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
    # Y축(좌측) 기준 회전 행렬
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c],
    ], dtype=np.float64)


def rot_z(yaw):
    # Z축(상향) 기준 회전 행렬
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
        # 차선/타깃 입력과 BEV 그리드 출력 토픽
        self.lines_topic = rospy.get_param("~lines_topic", "/lane_lines_px")
        self.target_topic = rospy.get_param("~target_topic", "/lane_target_px")
        self.lane_steer_topic = rospy.get_param("~lane_steer_topic", "/lane_steer")
        self.pub_topic = rospy.get_param("~grid_topic", "/lane_bev/grid")
        self.marker_topic = rospy.get_param("~marker_topic", "/lane_bev/markers")

        # -------- camera intrinsics --------
        # 카메라 내부 파라미터(픽셀 단위)
        self.fx = float(rospy.get_param("~fx"))
        self.fy = float(rospy.get_param("~fy"))
        self.cx = float(rospy.get_param("~cx"))
        self.cy = float(rospy.get_param("~cy"))

        # -------- camera extrinsics (base_link) --------
        # base_link 기준 카메라 위치 (m)와 자세 (deg)
        self.cam_x = float(rospy.get_param("~cam_x_m", 0.0))
        self.cam_y = float(rospy.get_param("~cam_y_m", 0.0))
        self.cam_z = float(rospy.get_param("~cam_height_m"))  # height above ground (z=0)

        self.roll = math.radians(float(rospy.get_param("~cam_roll_deg", 0.0)))
        self.pitch = math.radians(float(rospy.get_param("~cam_pitch_deg")))
        self.yaw = math.radians(float(rospy.get_param("~cam_yaw_deg", 0.0)))

        # -------- occupancy grid params --------
        # BEV 그리드 해상도/크기/원점 설정
        self.frame_id = rospy.get_param("~frame_id", "base_link")
        self.res = float(rospy.get_param("~resolution", 0.05))
        self.grid_w = int(rospy.get_param("~grid_w", 400))
        self.grid_h = int(rospy.get_param("~grid_h", 400))
        self.origin_x = float(rospy.get_param("~origin_x", 0.0))
        self.origin_y = float(rospy.get_param("~origin_y", -(self.grid_w * self.res) / 2.0))

        # -------- drawing / sampling --------
        # 샘플링 간격과 표시 범위(전방 거리)
        self.sample_n = int(rospy.get_param("~sample_n", 30))
        self.target_radius_cells = int(rospy.get_param("~target_radius_cells", 2))
        self.line_thickness_cells = int(rospy.get_param("~line_thickness_cells", 0))
        self.max_range_x = float(rospy.get_param("~max_range_x", self.grid_h * self.res))  # forward limit
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))

        # -------- state --------
        # 최신 수신 데이터를 저장해 주기적으로 그리드로 변환
        self.last_lines = None  # 8 ints
        self.last_target = None  # (u, v)
        self.last_lane_steer = None  # Int16

        # -------- marker style --------
        # 타깃 포인트와 조향 텍스트 색/크기 설정
        self.target_color = rospy.get_param("~target_color_rgba", [1.0, 0.3, 0.0, 1.0])
        self.steer_text_color = rospy.get_param("~steer_text_color_rgba", [1.0, 1.0, 1.0, 1.0])
        self.steer_text_size = float(rospy.get_param("~steer_text_size", 0.3))
        self.steer_text_z = float(rospy.get_param("~steer_text_z", 0.4))

        # -------- precompute matrices --------
        # Optical (x right, y down, z forward) -> camera_link-like (x forward, y left, z up)
        # x_fwd = z_opt
        # y_left = -x_opt
        # z_up = -y_opt
        # OpenCV 광학 좌표를 차량 좌표계 유사 기준으로 변환하는 행렬
        self.M_opt_to_cam = np.array([
            [0,  0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ], dtype=np.float64)

        # Rotation from camera-aligned frame to base_link
        # 카메라 정렬 좌표 -> base_link 회전
        self.R_cam_to_base = rot_z(self.yaw) @ rot_y(self.pitch) @ rot_x(self.roll)

        # base_link 기준 카메라 위치 벡터
        self.t_base = np.array([self.cam_x, self.cam_y, self.cam_z], dtype=np.float64)

        # -------- ROS I/O --------
        # 입력 구독, 출력 퍼블리시 설정
        self.sub_lines = rospy.Subscriber(self.lines_topic, Int32MultiArray, self.cb_lines, queue_size=1)
        self.sub_target = rospy.Subscriber(self.target_topic, PointStamped, self.cb_target, queue_size=1)
        self.sub_lane_steer = rospy.Subscriber(self.lane_steer_topic, Int16, self.cb_lane_steer, queue_size=1)

        self.pub_grid = rospy.Publisher(self.pub_topic, OccupancyGrid, queue_size=1)
        self.pub_marker = rospy.Publisher(self.marker_topic, Marker, queue_size=2)

        # 지정 주기로 BEV 그리드를 계산하여 발행
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
        # 차선 픽셀 좌표 8개를 저장 (좌/우 2개 세그먼트)
        if len(msg.data) >= 8:
            self.last_lines = list(msg.data[:8])

    def cb_target(self, msg: PointStamped):
        # point.x=u, point.y=v (pixels)
        # 타깃 픽셀 좌표 저장
        self.last_target = (float(msg.point.x), float(msg.point.y))

    def cb_lane_steer(self, msg: Int16):
        # 조향 오프셋 값 저장 (Int16)
        self.last_lane_steer = int(msg.data)

    def pixel_to_ground_xy(self, u: float, v: float):
        """
        Pixel (u,v) -> ground intersection (X,Y) in base_link.
        Returns None if ray doesn't intersect ground in front.
        """
        # ray in optical camera coords
        # 픽셀을 정규화한 광학 좌표계 방향 벡터
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        ray_opt = np.array([x, y, 1.0], dtype=np.float64)

        # optical -> camera-aligned
        # 광학 좌표 -> 카메라 정렬 좌표
        ray_cam = self.M_opt_to_cam @ ray_opt  # (x fwd, y left, z up) in camera-aligned frame

        # camera-aligned -> base_link
        # 카메라 정렬 좌표 -> base_link 방향 벡터
        dir_base = self.R_cam_to_base @ ray_cam

        dz = dir_base[2]
        if abs(dz) < 1e-9:
            return None

        # 지면(z=0)과의 교점 계산
        s = -self.t_base[2] / dz  # intersect z=0
        if s <= 0:
            return None

        p = self.t_base + s * dir_base  # (X,Y,0)

        # forward range gating (optional)
        # 전방 범위 밖은 제외
        if p[0] < 0 or p[0] > self.max_range_x:
            return None

        return (float(p[0]), float(p[1]))

    def xy_to_cell(self, X: float, Y: float):
        # 월드 좌표(X,Y)를 그리드 셀 인덱스로 변환
        ix = int((X - self.origin_x) / self.res)
        iy = int((Y - self.origin_y) / self.res)
        if 0 <= ix < self.grid_w and 0 <= iy < self.grid_h:
            return ix, iy
        return None

    def stamp_disk(self, grid: np.ndarray, ix: int, iy: int, r: int, val: int = 100):
        # grid shape: (H, W) with indexing [iy, ix]
        # 중심 (ix,iy)을 반지름 r의 원형으로 채움
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
        # 픽셀 좌표가 유효하지 않으면 스킵
        if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
            return

        # sample along pixel segment
        # 픽셀 선분을 일정 개수로 샘플링하여 지면에 투영
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
            if self.line_thickness_cells > 0:
                self.stamp_disk(grid, cell[0], cell[1], self.line_thickness_cells, val=0)
            else:
                grid[cell[1], cell[0]] = 0

    def _build_marker(self, marker_id: int, marker_type: int, x: float, y: float, z: float):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        m.ns = "lane_bev_rviz"
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        return m

    def on_timer(self, _evt):
        # 입력이 없으면 아무 것도 그리지 않음
        if self.last_lines is None and self.last_target is None:
            return

        # build occupancy grid array (int8: 0..100, -1 unknown)
        # 비어있는 그리드 생성(0=비어있음)
        occ = np.full((self.grid_h, self.grid_w), -1, dtype=np.int8)

        if self.last_lines is not None:
            # 좌/우 차선 선분을 BEV 그리드에 표시
            lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2 = self.last_lines
            self.draw_segment(occ, lx1, ly1, lx2, ly2)
            self.draw_segment(occ, rx1, ry1, rx2, ry2)

        if self.last_target is not None:
            # 타깃 포인트를 원 형태로 표시
            u, v = self.last_target
            xy = self.pixel_to_ground_xy(u, v)
            if xy is not None:
                cell = self.xy_to_cell(xy[0], xy[1])
                if cell is not None:
                    self.stamp_disk(occ, cell[0], cell[1], self.target_radius_cells, val=0)

        # publish OccupancyGrid
        # ROS OccupancyGrid 메시지로 변환해 발행
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

        # 타깃 마커와 조향 텍스트 마커 발행
        if self.last_target is not None:
            xy = self.pixel_to_ground_xy(self.last_target[0], self.last_target[1])
            if xy is not None:
                radius_m = max(0.0, self.target_radius_cells * self.res)
                target_m = self._build_marker(0, Marker.CYLINDER, xy[0], xy[1], 0.0)
                target_m.scale.x = max(1e-3, 2.0 * radius_m)
                target_m.scale.y = max(1e-3, 2.0 * radius_m)
                target_m.scale.z = max(1e-3, 0.05)
                target_m.color.r = float(self.target_color[0])
                target_m.color.g = float(self.target_color[1])
                target_m.color.b = float(self.target_color[2])
                target_m.color.a = float(self.target_color[3])
                self.pub_marker.publish(target_m)

                if self.last_lane_steer is not None:
                    text_m = self._build_marker(1, Marker.TEXT_VIEW_FACING,
                                                xy[0], xy[1], self.steer_text_z)
                    text_m.text = f"steer: {self.last_lane_steer}"
                    text_m.scale.z = max(1e-3, self.steer_text_size)
                    text_m.color.r = float(self.steer_text_color[0])
                    text_m.color.g = float(self.steer_text_color[1])
                    text_m.color.b = float(self.steer_text_color[2])
                    text_m.color.a = float(self.steer_text_color[3])
                    self.pub_marker.publish(text_m)


if __name__ == "__main__":
    LaneBEVRviz()
    rospy.spin()

#!/usr/bin/env python3

import rospy, math
import numpy as np
from std_msgs.msg import Int16, ColorRGBA
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from jsk_rviz_plugins.msg import OverlayText
from tf.transformations import quaternion_from_euler
import threading


class Parking:
    def __init__(self):
        rospy.init_node("parking")
        self.load_params()
        
        rospy.Subscriber("/ultrasonic1", Int16, self.ultrasonic1_callback, queue_size=1)
        rospy.Subscriber("/ultrasonic3", Int16, self.ultrasonic3_callback, queue_size=1)
        rospy.Subscriber("/ultrasonic4", Int16, self.ultrasonic4_callback, queue_size=1)
        rospy.Subscriber("/ultrasonic5", Int16, self.ultrasonic5_callback, queue_size=1)
        rospy.Subscriber("/detection_poses",PoseArray,self.detection_poses_callback,queue_size=1)
        rospy.Subscriber("/parking_lane_steer", Int16, self.lane_steer_callback, queue_size=1)
        rospy.Subscriber("/parking_stanley_steer", Int16, self.stanley_steer_callback, queue_size=1)
        
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)
        self.stanley_path_pub = rospy.Publisher("/stanley_path",Path,queue_size=1)
        self.dest_marker_pub = rospy.Publisher("/parking_destination_marker",Marker,queue_size=1)
        self.filtered_points_marker_pub = rospy.Publisher("/filtered_points_marker",Marker,queue_size=1)
        self.roi_marker_pub = rospy.Publisher("/roi_marker",Marker,queue_size=1)
        self.debug_text_pub = rospy.Publisher("/debug_overlay_text", OverlayText, queue_size=1, latch=True)
        
        self.ultrasonics = [-1, 20000, -1, 20000, 20000, 20000]
        self.rate = rospy.Rate(20)
        
        # ========================================
        # --------------------streak--------------
        self.parking_lot_streak = 0
        self.first_car_streak   = 0
        self.second_car_streak  = 0
        self.finish_streak      = 0
        self.parked_streak      = 0
        self.pulled_streak      = 0
        # ========================================
        
        
        # ========================================
        # --------------------steer---------------
        self.stanley_steer = 0
        self.lane_steer    = 0
        # ========================================


        # ========================================
        # --------------------flag----------------
        self.first_car_detected  = False
        self.second_car_detected = False
        self.can_start_parking   = False
        self.parked              = False
        self.pulled_out          = False
        self.to_finish           = False
        
        self.both_updated = False
        self.lost_second  = False
        self.lost_first   = False
        # ========================================
        
        
        # ========================================
        # --------------------state---------------
        self.mode           = "DEFAULT"
        self.steer_source   = "NONE"  # LANE / STANLEY / CONST
        self.state          = None
        self.prev_state     = None
        self.can_park_TH    = None
        # ========================================
        
        
        # ========================================
        # -------------------variable-------------
        self.start_time      = None
        self.first_car       = np.full((1, 2), np.nan)
        self.second_car      = np.full((1, 2), np.nan)
        self.filtered_points = np.vstack([self.first_car, self.second_car])
        # ========================================


        # ========================================
        # -------------------keyboard-------------
        self.lock = threading.Lock()
        th        = threading.Thread(target=self.keyboard_listener, daemon=True)
        th.start()
        # ========================================
        
        
        # ====================================================================================================
        # ------------------------------------------main loop-------------------------------------------------
        self.run()
        # ====================================================================================================
        
    def run(self):
        
        while not rospy.is_shutdown():

            if self.mode == "DEFAULT":
                self.drive(0,0)

            elif self.mode == "FINAL":
                if self.state == "lane_driving":             # 미션 시작해서 일직선으로 주행하는 단계
                    if not self.first_car_detected:
                        self.drive(self.lane_steer, self.FORWARD_SPEED)
                    else:
                        self.state = "full_left_steer"
                        continue

                if self.state == "full_left_steer":              # 주차 공간 발견 후 왼쪽으로 살짝 꺾어서 각 만드는 단계
                    if not self.can_start_parking:
                        self.drive(self.FULL_LEFT_STEER, self.FORWARD_SPEED)
                    else:
                        self.state = "pause_after_left"
                        self.start_time = rospy.Time.now()
                        continue

                if self.state == "pause_after_left":
                    elapsed = (rospy.Time.now() - self.start_time).to_sec()
                    self.drive(0, 0)
                    if elapsed >= self.PAUSE_TIME:
                        self.start_time = None
                        self.state = "stanley"
                        continue
                    
                if self.state == "stanley":                 # 후진해서 주차하는 단계
                    if not self.parked:
                        self.drive(self.stanley_steer, self.BACKWARD_SPEED)
                        self.sonic_check()
                    else:
                        self.state = "stop"
                        self.start_time = rospy.Time.now()
                        continue
            
                if self.state == "stop":
                    elapsed = (rospy.Time.now() - self.start_time).to_sec()
                    self.drive(0, 0)
                    if elapsed >= self.PARKING_PAUSE_TIME:
                        self.start_time = None
                        self.state = "pull_out"
                        continue
                    
                if self.state == "pull_out":
                    rospy.loginfo_throttle(0.5, f"pulled out: {self.pulled_out}")
                    if not self.pulled_out:
                        self.drive(0.0, self.FORWARD_SPEED)
                        self.sonic_check()
                    else:
                        if self.start_time is None:
                            self.start_time = rospy.Time.now()

                        elapsed = (rospy.Time.now() - self.start_time).to_sec()
                        if elapsed < self.GOING_RIGHT_TIME:
                            self.drive(self.FULL_RIGHT_STEER, 225) # TODO
                        else:
                            self.start_time = None
                            self.state = "finishing"
                            continue
                        
                if self.state == "finishing":
                    self.drive(self.lane_steer, 225)


            self.rate.sleep()
    
    def load_params(self):
        # ---------------- vehicle ----------------
        self.FULL_LEFT_STEER  = rospy.get_param("~vehicle/full_left_steer", 22.5)
        self.FULL_RIGHT_STEER = rospy.get_param("~vehicle/full_right_steer", -22.5)
        self.FORWARD_SPEED    = rospy.get_param("~vehicle/forward_speed", 125)
        self.BACKWARD_SPEED   = rospy.get_param("~vehicle/backward_speed", -100)

        # ---------------- timing ----------------
        self.PAUSE_TIME          = rospy.get_param("~timing/pause", 0.5)
        self.PARKING_PAUSE_TIME  = rospy.get_param("~timing/parking_pause", 2.0)
        self.GOING_RIGHT_TIME    = rospy.get_param("~timing/going_right", 2.5)

        # ---------------- threshold ----------------
        self.CAN_PARK_TH = rospy.get_param("~threshold/can_park", -2.0)
        self.ULTRASONIC_THRESHOLD = rospy.get_param("~threshold/ultrasonic", 500)
        self.TRACK_MAX_DIST = rospy.get_param("~threshold/track_max_dist", 0.5)

        # ---------------- roi ----------------
        self.ROI_LANE = rospy.get_param("~roi/lane", {
            "x_min": -1.0, "x_max": 1.0, "y_min": -3.0, "y_max": -1.0
        })
        self.ROI_FULL_LEFT = rospy.get_param("~roi/full_left", {
            "r_min": 0.3, "r_max": 3.0, "angle_deg": 90
        })
        
        self.ROI_LOST = rospy.get_param("~roi/lost", {
            "r_min": 1.5, "r_max": 2.5, "angle_deg": 180
        })

        # ---------------- stanley path ----------------
        self.SP = rospy.get_param("~stanley_path", {
            "back_len": 2.0, "front_len": 5.0, "step": 0.1, "frame_id": "laser"
        })

    def keyboard_listener(self):
        """
        d : DEFAULT (무조건 정지, state 저장)
        f : FINAL (FSM 실행 / 이전 state 복귀)
        """

        while not rospy.is_shutdown():
            key = input().strip().lower()

            with self.lock:
                # =====================
                # DEFAULT 모드
                # =====================
                if key == "d":
                    if self.mode != "DEFAULT":
                        self.mode = "DEFAULT"

                        # FSM 진행 중이었다면 state 저장
                        if self.state is not None:
                            self.prev_state = self.state
                            rospy.loginfo(
                                f"-> DEFAULT (saved state: {self.prev_state})"
                            )
                        else:
                            rospy.loginfo("-> DEFAULT")

                    else:
                        rospy.loginfo("already in DEFAULT")

                # =====================
                # FINAL 모드
                # =====================
                elif key == "f":
                    if self.mode != "FINAL":
                        self.mode = "FINAL"

                        # DEFAULT에서 복귀하는 경우
                        if self.prev_state is not None:
                            self.state = self.prev_state
                            rospy.loginfo(
                                f"-> FINAL (resume state: {self.state})"
                            )
                        else:
                            # 최초 FINAL 진입
                            self.state = "lane_driving"
                            rospy.loginfo(
                                "-> FINAL (start lane_driving)"
                            )

                    else:
                        rospy.loginfo("already in FINAL")

                else:
                    rospy.logwarn("invalid key (use 'd' or 'f')")
                      
    def lane_steer_callback(self, msg): self.lane_steer = msg.data
    
    def stanley_steer_callback(self, msg): self.stanley_steer = msg.data
        
    def ultrasonic1_callback(self, msg):
        if msg.data == -1:
            self.ultrasonics[1] = 50000
        else:
            self.ultrasonics[1] = msg.data

    def ultrasonic3_callback(self, msg):
        
        if msg.data == -1:
            self.ultrasonics[3] = 50000
        else:
            self.ultrasonics[3] = msg.data
    
    def ultrasonic4_callback(self, msg):
        
        if msg.data == -1:
            self.ultrasonics[4] = 50000
        else:
            self.ultrasonics[4] = msg.data
    
    def ultrasonic5_callback(self, msg):
        
        if msg.data == -1:
            self.ultrasonics[5] = 50000
        else:
            self.ultrasonics[5] = msg.data

    def detection_poses_callback(self, msg):
            
        points = np.array([[pose.position.x, pose.position.y] for pose in msg.poses],dtype=float)

        self.first_updated = False
        self.second_updated = False
            
        if not self.first_car_detected:
            self.detect_first_car(points)
        else:
            self.first_updated  = self.track_first_car(points)
            
        if self.first_car_detected and not self.second_car_detected:
            self.detect_second_car(points)
        elif self.second_car_detected:
            self.second_updated = self.track_second_car(points)
            

        if self.first_car_detected and self.second_car_detected:
            self.both_updated = self.first_updated and self.second_updated

            if self.both_updated:
                self.filtered_points = np.vstack(
                    [self.first_car, self.second_car]
                )
        else:
            self.both_updated = False


        if self.state == "full_left_steer":
            if self.both_updated:
                self.determine_can_parking(self.filtered_points)

        elif self.state in ["pause_after_left", "stanley"]:

            self.stanley_path(self.filtered_points)

        self.publish_debug_text()
            
    def detect_first_car(self, point):
        
        if self.lost_first:
            point = self.roi_filter(point, "full_left_steer")
            center = self.second_car.reshape(2)
            self.publish_full_left_roi_filled(center)
        else:
            point = self.roi_filter(point, "lane_driving")
            self.publish_lane_roi_filled()
        
        if point.shape[0] != 1:
            self.first_car_streak = max(0, self.first_car_streak-1)
            return
 
        self.first_car_streak += 1

        if self.first_car_streak >= 5:
            self.first_car_detected = True
            self.clear_roi_markers()
            self.first_car = point
            rospy.loginfo_once("First car detected")
    
    def detect_second_car(self, point):
        
        point = self.roi_filter(point, "full_left_steer")
        center = self.first_car.reshape(2)
        self.publish_full_left_roi_filled(center)
        
        if point.shape[0] != 1:
            self.second_car_streak = max(0, self.second_car_streak-1)
            return
        
        self.second_car_streak += 1
        
        if self.second_car_streak >= 5:
            self.second_car_detected = True
            self.clear_roi_markers()
            self.second_car = point
            rospy.loginfo("Second car detected")
            
    def track_first_car(self, point):
        """
        point: np.ndarray (N, 2) - 현재 프레임의 후보 점들
        max_dist: float - 추적 허용 반경 (m)
        """
        max_dist = self.TRACK_MAX_DIST
        if point is None or len(point) == 0:
            return False

        prev = self.first_car.reshape(2)
        dists = np.linalg.norm(point - prev, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist <= max_dist:
            self.first_car = point[min_idx].reshape(1, 2)
            if self.lost_first:
                rospy.logwarn(f"!!!! first car updated again !!! dist: {min_dist:.2f}")
                self.lost_first = False
            else:
                rospy.loginfo_throttle(0.5,f"first car updated, dist: {min_dist:.2f}")
            return True
        else:
            rospy.logerr_once(f"!!!! first car update false !!! dist: {min_dist:.2f}")
            self.first_car_detected = False
            self.lost_first = True
            return False
 
    def track_second_car(self, point):
        """
        point: np.ndarray (N, 2) - 현재 프레임의 후보 점들
        max_dist: float - 추적 허용 반경 (m)
        """
        max_dist = self.TRACK_MAX_DIST
        if point is None or len(point) == 0:
            return False

        prev = self.second_car.reshape(2)
        dists = np.linalg.norm(point - prev, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist <= max_dist:
            self.second_car = point[min_idx].reshape(1, 2)
            if self.lost_second:
                rospy.logwarn(f"!!!! second car updated again !!! dist: {min_dist:.2f}")
                self.lost_second = False
            else:
                rospy.loginfo_throttle(0.5,f"second car updated, dist: {min_dist:.2f}")
            return True
        else:
            rospy.logerr_once(f"!!!! second car update false !!! dist: {min_dist:.2f}")
            self.second_car_detected = False
            self.lost_second = True
            return False
    
    def determine_can_parking(self, points):
        self.can_park_TH = points[0, 1] + points[1, 1]
        if self.can_park_TH > self.CAN_PARK_TH:
            self.can_start_parking = True
            rospy.loginfo_once("Parking lot detected")
    
    def roi_filter(self, point, state):
        
        if point is None or len(point) == 0:
            return point

        if state == "lane_driving":
            cfg = self.ROI_LANE
            mask = (
                (point[:, 0] >= cfg["x_min"]) & (point[:, 0] <= cfg["x_max"]) &
                (point[:, 1] >= cfg["y_min"]) & (point[:, 1] <= cfg["y_max"])
            )
            return point[mask]

        elif state == "full_left_steer":

            if self.lost_second:
                cfg = self.ROI_LOST
                r_min = float(cfg["r_min"])
                r_max = float(cfg["r_max"])
                ang = math.radians(float(cfg["angle_deg"]))

                center = self.first_car.reshape(2)
                dx = point[:, 0] - center[0]
                dy = point[:, 1] - center[1]

                dist2 = dx**2 + dy**2
                dist_mask = (dist2 >= r_min**2) & (dist2 <= r_max**2)

                angles = np.arctan2(dy, dx)
                angle_mask = (angles >= -ang) & (angles <= ang)

                return point[dist_mask & angle_mask]
            
            if self.lost_first:
                cfg = self.ROI_LOST
                r_min = float(cfg["r_min"])
                r_max = float(cfg["r_max"])
                ang = math.radians(float(cfg["angle_deg"]))

                center = self.second_car.reshape(2)
                dx = point[:, 0] - center[0]
                dy = point[:, 1] - center[1]

                dist2 = dx**2 + dy**2
                dist_mask = (dist2 >= r_min**2) & (dist2 <= r_max**2)

                angles = np.arctan2(dy, dx)
                angle_mask = (angles >= -ang) & (angles <= ang)

                return point[dist_mask & angle_mask]
            
            cfg = self.ROI_FULL_LEFT
            r_min = float(cfg["r_min"])
            r_max = float(cfg["r_max"])
            ang = math.radians(float(cfg["angle_deg"]))

            center = self.first_car.reshape(2)
            dx = point[:, 0] - center[0]
            dy = point[:, 1] - center[1]

            dist2 = dx**2 + dy**2
            dist_mask = (dist2 >= r_min**2) & (dist2 <= r_max**2)

            angles = np.arctan2(dy, dx)
            angle_mask = (angles >= -ang) & (angles <= ang)

            return point[dist_mask & angle_mask]
            
    def drive(self, des_steer, long_cmd):

        # steer source 판별
        if des_steer == self.lane_steer:
            self.steer_source = "LANE"
        elif des_steer == self.stanley_steer:
            self.steer_source = "STANLEY"
        else:
            self.steer_source = "CONST"

        self.motor_cmd_steer_pub.publish(Int16(int(des_steer)))
        self.motor_long_pub.publish(Int16(int(long_cmd)))
    
    def sonic_check(self):

        sonics_to_use1 = [self.ultrasonics[1], self.ultrasonics[3], self.ultrasonics[4], self.ultrasonics[5]]
        sonics_to_use2 = [self.ultrasonics[1], self.ultrasonics[3]]

        if self.state == "stanley":
            
            if all(x < self.ULTRASONIC_THRESHOLD for x in sonics_to_use1):
                self.parked_streak += 1
                if self.parked_streak >= 20:
                    self.parked = True
            else:
                self.parked_streak = max(0, self.parked_streak - 1)
                
        elif self.state == "pull_out":
            
            if all(x > self.ULTRASONIC_THRESHOLD for x in sonics_to_use2):
                self.pulled_streak += 1
                if self.pulled_streak >= 5:
                    self.pulled_out = True
            else:
                self.pulled_streak = max(0, self.pulled_streak - 1)
                

    def stanley_path(self, filtered_points):
        
        if not self.both_updated:
            empty_path = Path()
            empty_path.header.stamp = rospy.Time.now()
            empty_path.header.frame_id = self.SP.get("frame_id", "laser")
            self.stanley_path_pub.publish(empty_path)
            self.clear_stanley_markers()
            rospy.logwarn_throttle(1.0, "stanley path invalid (not both updated)")
            return

        dest = np.mean(filtered_points, axis=0)
        v = filtered_points[1] - filtered_points[0]

        n = np.array([-v[1], v[0]])
        n_hat = n / np.linalg.norm(n)

        if np.dot(n_hat, -dest) < 0:
            n_hat = -n_hat

        back_len  = float(self.SP.get("back_len", 2.0))
        front_len = float(self.SP.get("front_len", 5.0))
        step      = float(self.SP.get("step", 0.1))

        s_vals = np.arange(-back_len, front_len + step, step)
        path_xy = np.array([dest + s * n_hat for s in s_vals])

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = self.SP.get("frame_id", "laser")
            
        # 차량이 따라가야 할 진행 방향 (차량 → 목표)
        heading_dir = -n_hat
        yaw = math.atan2(heading_dir[1], heading_dir[0])
        _, _, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        for x, y in path_xy:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)


        self.stanley_path_pub.publish(path_msg)
        rospy.loginfo_throttle(1.0, "publishing stanley path")
        self.publish_destination_marker(dest)
        self.publish_filtered_points_marker(filtered_points)
        self.publish_filtered_points_line(filtered_points)
        
    # ====================================================================================================
    # ------------------------------------------markers---------------------------------------------------
    def publish_destination_marker(self, dest):
        """
        dest: np.ndarray (2,)
        """

        now = rospy.Time.now()

        # ---------------- sphere (destination) ----------------
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = now

        marker.ns = "parking_destination"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(dest[0])
        marker.pose.position.y = float(dest[1])
        marker.pose.position.z = 0.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.dest_marker_pub.publish(marker)

        # ---------------- text (local coordinate) ----------------
        text = Marker()
        text.header.frame_id = "laser"
        text.header.stamp = now

        text.ns = "parking_destination"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD

        # 텍스트 위치 (살짝 위)
        text.pose.position.x = float(dest[0])
        text.pose.position.y = float(dest[1])
        text.pose.position.z = 0.4

        text.scale.z = 0.15   # 글자 크기 (작게)

        # 흰색 텍스트
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0

        text.text = f"({dest[0]:+.2f}, {dest[1]:+.2f})"

        self.dest_marker_pub.publish(text)
    # 빨간 구(SPHERE): 두 차량(클러스터) 중심의 평균점 dest = “주차 목표점/중앙점”
    # 흰색 텍스트(TEXT_VIEW_FACING): 그 목표점의 좌표 (x, y)를 글자로 표시
    
    def publish_filtered_points_marker(self, points):
        
        """
        points: np.ndarray shape (2, 2)
        두 점을 각각 속이 빈 원(Line Strip)으로 시각화
        """

        if points is None or points.shape != (2, 2):
            return

        radius = 0.2        # 원 반지름 (m)
        num_segments = 40   # 원 해상도

        for i, (cx, cy) in enumerate(points):
            marker = Marker()
            marker.header.frame_id = "laser"
            marker.header.stamp = rospy.Time.now()

            marker.ns = "filtered_points"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.scale.x = 0.05   # 선 두께

            # 색상 (파란색)
            marker.color.r = 0.0
            marker.color.g = 0.4
            marker.color.b = 1.0
            marker.color.a = 1.0

            # 원 껍질 생성
            for k in range(num_segments + 1):
                theta = 2.0 * math.pi * k / num_segments
                p = Point()
                p.x = cx + radius * math.cos(theta)
                p.y = cy + radius * math.sin(theta)
                p.z = 0.0
                marker.points.append(p)

            self.filtered_points_marker_pub.publish(marker)
    # 파란 원 테두리 2개(LINE_STRIP): points[0], points[1] 각각(첫 차/둘째 차로 추적 중인 포인트)을 원 형태로 강조 표시
    
    def publish_filtered_points_line(self, points):
        """
        points: np.ndarray shape (2, 2)
        두 점을 잇는 선분 시각화
        """

        if points is None or points.shape != (2, 2):
            return

        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "filtered_points_line"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.06   # 선 두께

        # 색상 (노란색)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        p0 = Point()
        p0.x = float(points[0, 0])
        p0.y = float(points[0, 1])
        p0.z = 0.0

        p1 = Point()
        p1.x = float(points[1, 0])
        p1.y = float(points[1, 1])
        p1.z = 0.0

        marker.points.append(p0)
        marker.points.append(p1)

        self.filtered_points_marker_pub.publish(marker)
    # 노란 선(LINE_LIST): points[0] ↔ points[1] 를 잇는 선분 = 두 차량 사이 방향/간격을 시각화 (이 벡터로 법선 잡아서 stanley path 만들지)
        
    def publish_lane_roi_filled(self):
        cfg = self.ROI_LANE  # YAML에서 로드된 dict: x_min/x_max/y_min/y_max

        x_min = float(cfg["x_min"])
        x_max = float(cfg["x_max"])
        y_min = float(cfg["y_min"])
        y_max = float(cfg["y_max"])

        # CUBE는 중심/크기 형태로 넣어야 하니까 YAML 범위를 center/scale로 변환
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        sx = (x_max - x_min)
        sy = (y_max - y_min)

        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "roi_fill"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = 0.0

        marker.scale.x = sx
        marker.scale.y = sy
        marker.scale.z = 0.01

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.25

        self.roi_marker_pub.publish(marker)
    # 초록 반투명 박스(CUBE): lane_driving 상태에서 쓰는 ROI (x_min~x_max, y_min~y_max) 영역 = “첫 번째 차를 찾는 검색 구역”

    def publish_full_left_roi_filled(self, center):
        
        now = rospy.Time.now()
        kill = Marker()
        kill.header.frame_id = "laser"
        kill.header.stamp = now
        kill.ns = "roi_fill"
        kill.id = 0
        kill.action = Marker.DELETE
        self.roi_marker_pub.publish(kill)
        
        if self.lost_second or self.lost_first:
            cfg = self.ROI_LOST  # YAML: r_min/r_max/angle_deg
        else:
            cfg = self.ROI_FULL_LEFT  # YAML: r_min/r_max/angle_deg

        r_min = float(cfg["r_min"])
        r_max = float(cfg["r_max"])
        ang = math.radians(float(cfg["angle_deg"]))  # -ang ~ +ang

        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "roi_fill"
        marker.id = 1
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.45   # 0.25 -> 0.45~0.6 추천

        # 각도 범위: -ang ~ +ang (roi()와 동일)
        ang_min = -ang
        ang_max =  ang

        num = 40
        angles = np.linspace(ang_min, ang_max, num)

        for i in range(len(angles) - 1):
            th1 = angles[i]
            th2 = angles[i + 1]

            # inner arc
            i1 = Point(center[0] + r_min * math.cos(th1),
                    center[1] + r_min * math.sin(th1),
                    0.0)
            i2 = Point(center[0] + r_min * math.cos(th2),
                    center[1] + r_min * math.sin(th2),
                    0.0)

            # outer arc
            o1 = Point(center[0] + r_max * math.cos(th1),
                    center[1] + r_max * math.sin(th1),
                    0.0)
            o2 = Point(center[0] + r_max * math.cos(th2),
                    center[1] + r_max * math.sin(th2),
                    0.0)

            marker.points += [o1, i1, i2]
            marker.points += [o1, i2, o2]

        self.roi_marker_pub.publish(marker)
        
    # 파란 반투명 부채꼴 도넛(TRIANGLE_LIST): full_left_steer 상태에서 first_car를 중심으로 하는 ROI (r_min~r_max + ±angle) = “두 번째 차를 찾는 검색 구역”

    def clear_stanley_markers(self):
        now = rospy.Time.now()

        def delete_marker(ns, mid):
            m = Marker()
            m.header.frame_id = "laser"
            m.header.stamp = now
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            return m

        # destination
        self.dest_marker_pub.publish(delete_marker("parking_destination", 0))
        self.dest_marker_pub.publish(delete_marker("parking_destination", 1))

        # filtered points (2개)
        self.filtered_points_marker_pub.publish(delete_marker("filtered_points", 0))
        self.filtered_points_marker_pub.publish(delete_marker("filtered_points", 1))

        # line
        self.filtered_points_marker_pub.publish(delete_marker("filtered_points_line", 0))
    
    def clear_roi_markers(self):
        now = rospy.Time.now()

        # (추가) 같은 토픽의 모든 마커 제거
        m = Marker()
        m.header.frame_id = "laser"
        m.header.stamp = now
        m.action = Marker.DELETEALL
        self.roi_marker_pub.publish(m)

        # 기존처럼 id별 삭제도 유지
        for mid in [0, 1]:
            m = Marker()
            m.header.frame_id = "laser"
            m.header.stamp = now
            m.ns = "roi_fill"
            m.id = mid
            m.action = Marker.DELETE
            self.roi_marker_pub.publish(m)
    # 위 마커들을 ns/id 기준으로 DELETE해서 RViz에서 지움 (stanley 관련 마커 / roi_fill 마커 정리)
    
    def publish_debug_text(self):

        def fmt(p):
            if np.any(np.isnan(p)):
                return "(NaN, NaN)"
            return f"({p[0]:+.2f}, {p[1]:+.2f})"
        text = (
            f"[ PARKING DEBUG ]\n\n"
            f"STATE : {self.state}\n\n"
            f"FIRST CAR\n"
            f"   detected : {self.first_car_detected}\n"
            f"   pos      : {fmt(self.first_car.reshape(2))}\n\n"
            f"SECOND CAR\n"
            f"   detected : {self.second_car_detected}\n"
            f"   pos      : {fmt(self.second_car.reshape(2))}\n\n"
            f"STEER SOURCE   : {self.steer_source}\n"
            f"CUR STANLEY TH (up to > {self.CAN_PARK_TH}): {self.can_park_TH}\n\n"
            f"SONICS\n"
            f"left front   : {self.ultrasonics[1]}\n"
            f"right front   : {self.ultrasonics[3]}\n"
            f"left rear   : {self.ultrasonics[4]}\n"
            f"right rear   : {self.ultrasonics[5]}\n"
            f"threshold   : {self.ULTRASONIC_THRESHOLD}\n"
            f"steak   : {self.parked_streak}\n"

        )

        msg = OverlayText()
        msg.text = text

        # ---------- 화면 고정 위치 ----------
        msg.left   = 10     # 좌측 상단
        msg.top    = 10
        msg.width  = 420
        msg.height = 380
        msg.text_size = 10

        # ---------- 색상 ----------
        if self.first_updated and self.second_updated:
            msg.fg_color = ColorRGBA(0.2, 0.6, 1.0, 1.0)  # 파랑
        else:
            msg.fg_color = ColorRGBA(1.0, 0.2, 0.2, 1.0)  # 빨강
        msg.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.0)

        self.debug_text_pub.publish(msg)


    # 화면 고정 텍스트(TEXT_VIEW_FACING): 현재 STATE, 각 차 인지/업데이트 여부, 좌표, steer source, can_park_TH 같은 디버그 상태판

    # ====================================================================================================

if __name__ == "__main__":
    try:
        Parking()
    except rospy.ROSInterruptException:
        pass
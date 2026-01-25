#!/usr/bin/env python3

import rospy, math
import numpy as np
from std_msgs.msg import Int16
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler
import threading

L = 0.8 # 휠베이스
K = 20
TH_MAX               = 1
CAN_PARK_TH          = -2.5
FORWARD_SPEED        = 50
BACKWARD_SPEED       = -100
PAUSE_TIME           = 0.5
PARKING_PAUSE_TIME   = 2
GOING_LEFT_TIME      = 1.5
GOING_RIGHT_TIME     = 2.5
GOING_STRAIGHT_TIME  = 3
ULTRASONIC_THRESHOLD = 500  # 50cm
FULL_LEFT_STEER      = 22.5
FULL_RIGHT_STEER     = -22.5

class Parking:
    def __init__(self):
        rospy.init_node("parking")

        # rospy.Subscriber("/ultrasonic2", Int16, self.ultrasonic2_callback, queue_size=1)
        # rospy.Subscriber("/ultrasonic3", Int16, self.ultrasonic3_callback, queue_size=1)
        # rospy.Subscriber("/ultrasonic4", Int16, self.ultrasonic4_callback, queue_size=1)
        # rospy.Subscriber("/ultrasonic5", Int16, self.ultrasonic5_callback, queue_size=1)
        rospy.Subscriber("/detection_poses",PoseArray,self.detection_poses_callback,queue_size=1)
        rospy.Subscriber("/parking_lane_steer", Int16, self.lane_steer_callback, queue_size=1)
        rospy.Subscriber("/parking_stanley_steer", Int16, self.stanley_steer_callback, queue_size=1)
        
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)
        self.stanley_path_pub = rospy.Publisher("/stanley_path",Path,queue_size=1)
        self.dest_marker_pub = rospy.Publisher("/parking_destination_marker",Marker,queue_size=1)
        self.filtered_points_marker_pub = rospy.Publisher("/filtered_points_marker",Marker,queue_size=1)
        self.roi_marker_pub = rospy.Publisher("/roi_marker",Marker,queue_size=1)
        
        self.ultrasonics = [10000, 10000, 10000, 10000]
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
        # ========================================
        
        
        # ========================================
        # --------------------state---------------
        self.mode           = "DEFAULT"
        self.steer_source   = "NONE"  # LANE / STANLEY / CONST
        self.state          = None
        self.prev_state     = None
        self.prev_roi_state = None
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
        
        
        # ========================================
        # ------------------debugging-------------
        # self.debugging    = True   # TODO
        self.debugging = False   # TODO
        # self.debug_mode   = "step1"
        self.debug_mode   = "step2"
        # self.debug_mode   = "step3"
        # ========================================
        
        # ====================================================================================================
        # ------------------------------------------debugging-------------------------------------------------
        if self.debugging:
            
            if self.debug_mode == "step1":
                
                self.state = "lane_driving"
                rospy.logwarn_once("STANLEY DEBUG MODE ENABLED")
                rospy.loginfo_throttle(0.5,f"state: {self.state}",
                      f"parking points: {self.filtered_points}",
                      f"can_start_parking: {self.can_start_parking}"
                      )
                
                while not rospy.is_shutdown():
                    if not self.first_car_detected:
                        self.drive(self.lane_steer, FORWARD_SPEED)
                        
                    else:
                        self.state = "full_left_steer"
                        continue

                    if self.state == "full_left_steer":              # 주차 공간 발견 후 왼쪽으로 살짝 꺾어서 각 만드는 단계
                        if not self.can_start_parking:
                            self.drive(FULL_LEFT_STEER, FORWARD_SPEED)
                            
                        else:
                            self.state = "pause_after_left"
                            self.start_time = rospy.Time.now()
                            continue

                    if self.state == "pause_after_left":
                        elapsed = (rospy.Time.now() - self.start_time).to_sec()
                        self.drive(0, 0)
                        
                        if elapsed >= PAUSE_TIME:
                            self.start_time = None
                            self.state = "stanley"
                            continue
                        
                    self.rate.sleep()
                    
                    
            if self.debug_mode == "step2":
                
                self.state = "stanley"
                rospy.logwarn_once("STANLEY DEBUG MODE ENABLED")
                print(f"state: {self.state}",
                      f"parking points: {self.filtered_points}",
                      f"parked: {self.parked}",
                      )
                
                while not rospy.is_shutdown():
                    
                    if not self.parked:
                        self.drive(self.stanley_steer, BACKWARD_SPEED)
                        self.sonic_check()
                    else:
                        self.state = "stop"
                        self.start_time = rospy.Time.now()
                        continue
            
                    if self.state == "stop":
                        elapsed = (rospy.Time.now() - self.start_time).to_sec()
                        self.drive(0, 0)
                        if elapsed >= PARKING_PAUSE_TIME:
                            self.start_time = None
                            self.state = "pull_out"
                            continue
                    
                    self.rate.sleep()
            
            
            if self.debug_mode == "step3":
                
                self.state = "pull_out"
                rospy.logwarn_once("PULL OUT DEBUG MODE ENABLED")
                print(f"state: {self.state}",
                      f"parking points: {self.filtered_points}",
                      f"pulled_out: {self.pulled_out}",
                      )
                
                while not rospy.is_shutdown():
                    
                    if not self.pulled_out:
                        self.drive(0.0, FORWARD_SPEED)
                        self.sonic_check()
                    else:
                        if self.start_time is None:
                            self.start_time = rospy.Time.now()

                        elapsed = (rospy.Time.now() - self.start_time).to_sec()
                        if elapsed < GOING_RIGHT_TIME:
                            self.drive(FULL_RIGHT_STEER, FORWARD_SPEED)
                        else:
                            self.start_time = None
                            self.state = "finishing"
                            continue
                        
                if self.state == "finishing":
                    self.drive(self.lane_steer, FORWARD_SPEED)
                    
                    self.rate.sleep()
        # ====================================================================================================
        
        
        # ====================================================================================================
        # ------------------------------------------main loop-------------------------------------------------
        else:
            self.run()
        # ====================================================================================================
        
    def run(self):
        
        while not rospy.is_shutdown():
            rospy.loginfo_throttle(0.5, f"state: {self.state}, filtered_points: {self.filtered_points}")

            if self.mode == "DEFAULT":
                self.drive(0,0)

            elif self.mode == "FINAL":
                if self.state == "lane_driving":             # 미션 시작해서 일직선으로 주행하는 단계
                    if not self.first_car_detected:
                        self.drive(self.lane_steer, FORWARD_SPEED)
                        print(f"&*$@^(*@^*&#)(@*&*)(#&@()$&()@&^$*@^)*&$#&@*()#&*)(@&$*()@&*($)&$) lane steer: {self.lane_steer}")
                    else:
                        self.state = "full_left_steer"
                        continue

                if self.state == "full_left_steer":              # 주차 공간 발견 후 왼쪽으로 살짝 꺾어서 각 만드는 단계
                    if not self.can_start_parking:
                        self.drive(FULL_LEFT_STEER, FORWARD_SPEED)
                    else:
                        self.state = "pause_after_left"
                        self.start_time = rospy.Time.now()
                        continue

                if self.state == "pause_after_left":
                    elapsed = (rospy.Time.now() - self.start_time).to_sec()
                    self.drive(0, 0)
                    if elapsed >= PAUSE_TIME:
                        self.start_time = None
                        self.state = "stanley"
                        continue
                    
                if self.state == "stanley":                 # 후진해서 주차하는 단계
                    if not self.parked:
                        self.drive(self.stanley_steer, BACKWARD_SPEED)
                        self.sonic_check()
                    else:
                        self.state = "stop"
                        self.start_time = rospy.Time.now()
                        continue
            
            
                if self.state == "stop":
                    elapsed = (rospy.Time.now() - self.start_time).to_sec()
                    self.drive(0, 0)
                    if elapsed >= PARKING_PAUSE_TIME:
                        self.start_time = None
                        self.state = "pull_out"
                        continue


                if self.state == "pull_out":
                    if not self.pulled_out:
                        self.drive(0.0, FORWARD_SPEED)
                        self.sonic_check()
                    else:
                        if self.start_time is None:
                            self.start_time = rospy.Time.now()

                        elapsed = (rospy.Time.now() - self.start_time).to_sec()
                        if elapsed < GOING_RIGHT_TIME:
                            self.drive(FULL_RIGHT_STEER, FORWARD_SPEED)
                        else:
                            self.start_time = None
                            self.state = "finishing"
                            continue
                        
                if self.state == "finishing":
                    self.drive(self.lane_steer, FORWARD_SPEED)


            self.rate.sleep()
              
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
        
    # def ultrasonic2_callback(self, msg): self.ultrasonics[0] = msg.data
    
    # def ultrasonic3_callback(self, msg): self.ultrasonics[1] = msg.data
    
    # def ultrasonic4_callback(self, msg): self.ultrasonics[2] = msg.data
    
    # def ultrasonic5_callback(self, msg): self.ultrasonics[3] = msg.data

    def detection_poses_callback(self, msg):
            
        points = np.array(
            [[pose.position.x, pose.position.y] for pose in msg.poses],
            dtype=float
        )

        # -------------------------------
        # (1) 항상 인지 로직은 돌린다
        # -------------------------------
        self.first_updated = False
        self.second_updated = False
        # 첫 번째 차 인지
        if not self.first_car_detected:
            self.detect_first_car(points)
        else:
            self.first_updated  = self.track_first_car(points)
            
        if self.state != self.prev_roi_state:
            self.clear_roi_markers()
            self.prev_roi_state = self.state
            
        # 두 번째 차 인지
        if self.first_car_detected and not self.second_car_detected:
            self.detect_second_car(points)
        elif self.second_car_detected:
            self.second_updated = self.track_second_car(points)
            

        # -------------------------------
        # (2) 두 차가 모두 잡혔으면 항상 추적
        # -------------------------------
        if self.first_car_detected and self.second_car_detected:
            self.both_updated = self.first_updated and self.second_updated

            if self.both_updated:
                self.filtered_points = np.vstack(
                    [self.first_car, self.second_car]
                )
        else:
            self.both_updated = False
        # -------------------------------
        # (3) FSM 단계별로 "사용만" 분기
        # -------------------------------
        if self.state == "full_left_steer":
            if self.both_updated:
                self.determine_can_parking(self.filtered_points)

        elif self.state in ["pause_after_left", "stanley"]:
            self.stanley_path(self.filtered_points)

        self.publish_debug_text()
            
                
    def detect_first_car(self, point):
        
        point = self.roi(point, "lane_driving")
        self.publish_lane_roi_filled()
        
        if point.shape[0] != 1:
            rospy.loginfo_throttle(0.5,"not first car")
            self.first_car_streak = 0
            return
 
        rospy.loginfo_throttle(0.5,"first car")
        self.first_car_streak += 1

        if self.first_car_streak >= 5:
            self.first_car_detected = True
            self.clear_roi_markers()
            self.first_car = point
            rospy.loginfo_once("First car detected")
    
    
    def detect_second_car(self, point):
        
        point = self.roi(point, "full_left_steer")
        center = self.first_car.reshape(2)
        self.publish_full_left_roi_filled(center)
        if point.shape[0] != 1:
            rospy.loginfo_throttle(0.5,"not second car")
            self.second_car_streak = 0
            return
        
        rospy.loginfo_throttle(0.5,"second car")
        self.second_car_streak += 1
        
        if self.second_car_streak >= 5:
            self.second_car_detected = True
            self.clear_roi_markers()
            self.second_car = point
            rospy.loginfo_once("Second car detected")
            

    def track_first_car(self, point, max_dist=0.5):
        """
        point: np.ndarray (N, 2) - 현재 프레임의 후보 점들
        max_dist: float - 추적 허용 반경 (m)
        """
        if point is None or len(point) == 0:
            return False

        # (1,2) → (2,)
        prev = self.first_car.reshape(2)

        # 거리 계산 (N,)
        dists = np.linalg.norm(point - prev, axis=1)

        # 가장 가까운 점
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist <= max_dist:
            # 추적 성공 → 업데이트
            self.first_car = point[min_idx].reshape(1, 2)
            rospy.loginfo_throttle(0.5,"first car updated")
            return True
        else:
            # 추적 실패
            rospy.logerr_once(f"!!!! first car update false !!! dist: {min_dist}")
            return False

        
        
    def track_second_car(self, point, max_dist = 0.5):
        """
        point: np.ndarray (N, 2) - 현재 프레임의 후보 점들
        max_dist: float - 추적 허용 반경 (m)
        """

        if point is None or len(point) == 0:
            return False

        # (1,2) → (2,)
        prev = self.second_car.reshape(2)

        # 거리 계산 (N,)
        dists = np.linalg.norm(point - prev, axis=1)

        # 가장 가까운 점
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        if min_dist <= max_dist:
            # 추적 성공 → 업데이트
            self.second_car = point[min_idx].reshape(1, 2)
            rospy.loginfo_throttle(0.5,"second car updated")
            return True
        else:
            # 추적 실패
            rospy.logerr_once(f"!!!! second car update false !!! dist: {min_dist}")
            return False
    
    
    def determine_can_parking(self, points):
        self.can_park_TH = points[0, 1] + points[1, 1]
        if self.can_park_TH > CAN_PARK_TH:
            self.can_start_parking = True
            rospy.loginfo_once("Parking lot detected")

        
    def roi(self, point, state, r_min=1.5, r_max=2.5):

        if point is None or len(point) == 0:
            return point

        # state가 바뀔 때만 ROI clear

        if state == "lane_driving":

            mask = (
                (point[:, 0] >= -1) & (point[:, 0] <= 1) &
                (point[:, 1] >= -3) & (point[:, 1] <= -1)
            )
            return point[mask]

        elif state == "full_left_steer":

            if not self.first_car_detected:
                return point

            center = self.first_car.reshape(2)

            dx = point[:, 0] - center[0]
            dy = point[:, 1] - center[1]

            # (1) 거리 조건 (도넛)
            dist2 = dx**2 + dy**2
            dist_mask = (dist2 >= r_min**2) & (dist2 <= r_max**2)

            # (2) 각도 조건 (laser 기준 -30 ~ +30 deg)
            angles = np.arctan2(dy, dx)   # rad
            angle_mask = (angles >= -math.radians(30)) & \
                        (angles <=  math.radians(30))

            # (3) 최종 ROI
            mask = dist_mask & angle_mask
            return point[mask]

                
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
        
        if self.state == "stanley":
            
            if all(x < ULTRASONIC_THRESHOLD for x in self.ultrasonics):
                self.parked_streak += 1
                if self.parked_streak >= 5:
                    self.parked = True
            else:
                self.parked_streak = 0
                
        elif self.state == "pull_out":
            
            sonics_to_use = [self.ultrasonics[0], self.ultrasonics[2]]  # TODO
            if all(x > ULTRASONIC_THRESHOLD + 1000 for x in sonics_to_use):
                self.pulled_streak += 1
                if self.pulled_streak >= 10:
                    self.pulled_out = True
            else:
                self.pulled_streak = 0
                
                
    def stanley_path(self, filtered_points):
        
        if not self.both_updated:
            empty_path = Path()
            empty_path.header.stamp = rospy.Time.now()
            empty_path.header.frame_id = "laser"
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

        back_len = 2.0    # 차량 반대 방향
        front_len = 5.0   # 차량 쪽
        step = 0.1
        s_vals = np.arange(-back_len, front_len + step, step)
        path_xy = np.array([dest + s * n_hat for s in s_vals])

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "laser" 
        
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
        
    def publish_lane_roi_filled(self):
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "roi_fill"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # 중심
        marker.pose.position.x = 0.0
        marker.pose.position.y = -2.0
        marker.pose.position.z = 0.0

        # 크기
        marker.scale.x = 2.0   # -1 ~ 1
        marker.scale.y = 2.0   # -3 ~ -1
        marker.scale.z = 0.01  # 얇게

        # 반투명 초록
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.25  # 🔥 반투명

        self.roi_marker_pub.publish(marker)
        
    def publish_full_left_roi_filled(self, center, r_min=1.5, r_max=2.5):
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

        # 반투명 파랑
        marker.color.r = 0.0
        marker.color.g = 0.4
        marker.color.b = 1.0
        marker.color.a = 0.25

        # -------------------------------
        # 각도 제한: -30 ~ +30 deg (laser 기준)
        # -------------------------------
        ang_min = -math.radians(30)
        ang_max =  math.radians(30)

        num = 40  # 부채꼴 해상도
        angles = np.linspace(ang_min, ang_max, num)

        for i in range(len(angles) - 1):
            th1 = angles[i]
            th2 = angles[i + 1]

            # inner arc
            i1 = Point(
                center[0] + r_min * math.cos(th1),
                center[1] + r_min * math.sin(th1),
                0.0
            )
            i2 = Point(
                center[0] + r_min * math.cos(th2),
                center[1] + r_min * math.sin(th2),
                0.0
            )

            # outer arc
            o1 = Point(
                center[0] + r_max * math.cos(th1),
                center[1] + r_max * math.sin(th1),
                0.0
            )
            o2 = Point(
                center[0] + r_max * math.cos(th2),
                center[1] + r_max * math.sin(th2),
                0.0
            )

            # 도넛 부채꼴 채우기 (2 triangles)
            marker.points += [o1, i1, i2]
            marker.points += [o1, i2, o2]

        self.roi_marker_pub.publish(marker)

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
        for mid in [0, 1]:
            m = Marker()
            m.header.frame_id = "laser"
            m.header.stamp = now
            m.ns = "roi_fill"
            m.id = mid
            m.action = Marker.DELETE
            self.roi_marker_pub.publish(m)
    
    def publish_debug_text(self):

        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "debug_text"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # 화면 왼쪽 고정 느낌 위치
        marker.pose.position.x = -1.5
        marker.pose.position.y = 1.5
        marker.pose.position.z = 1.2

        marker.scale.z = 0.25  # 글자 크기

        # 기본 흰색
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        def fmt(p):
            if np.any(np.isnan(p)):
                return "(NaN, NaN)"
            return f"({p[0]:+.2f}, {p[1]:+.2f})"

        # 색상 표시용 이모지 스타일
        f_color = "BLUE" if self.first_updated else "RED"
        s_color = "BLUE" if self.second_updated else "RED"

        text = f"""
        STATE : {self.state}

        FIRST CAR  [{f_color}]
        detected : {self.first_car_detected}
        pos      : {fmt(self.first_car.reshape(2))}

        SECOND CAR [{s_color}]
        detected : {self.second_car_detected}
        pos      : {fmt(self.second_car.reshape(2))}

        STEER SOURCE : {self.steer_source}
        TO STANLEY TH ( > -2 check): {self.can_park_TH}
        """

        marker.text = text.strip()

        # 색상 반영 (전체 텍스트 기준)
        if not (self.first_updated and self.second_updated):
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
        else:
            marker.color.r = 0.2
            marker.color.g = 0.6
            marker.color.b = 1.0

        self.roi_marker_pub.publish(marker)

    # ====================================================================================================

if __name__ == "__main__":
    try:
        Parking()
    except rospy.ROSInterruptException:
        pass
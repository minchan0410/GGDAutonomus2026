#!/usr/bin/env python3

import rospy, math
import numpy as np
from std_msgs.msg import Int16, Bool
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from collections import deque
from tf.transformations import quaternion_from_euler
import threading

L = 0.8 # 휠베이스
K = 20
TH_MAX               = 1
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
        rospy.Subscriber("/is_lane_detected", Bool, self.lane_detected_callback, queue_size=1)
        
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)
        self.stanley_path_pub = rospy.Publisher("/stanley_path",Path,queue_size=1)
        self.dest_marker_pub = rospy.Publisher("/parking_destination_marker",Marker,queue_size=1)
        self.filtered_points_marker_pub = rospy.Publisher("/filtered_points_marker",Marker,queue_size=1)
        
        self.ultrasonics = [10000, 10000, 10000, 10000]
        
        
        self.parking_lot_streak = 0
        self.first_car_streak = 0
        self.first_car_detected = False
        self.finish_streak = 0
        self.parked_streak = 0
        self.pulled_streak = 0
        self.stanley_steer = 0
        self.is_lane_detected = False
        self.filtered_point = None
        self.lane_detect_queue = deque(maxlen=10)
        self.to_finish = False
        self.can_start_parking = False
        self.parked = False
        self.start_time = None
        self.mission_completed = False
        self.pulled_out = False
        self.filtered_point = np.empty((2, 2))     # (2,2) numpy array
        self.reject_count = 0       # 유사성 실패 누적
        self.mode = "DEFAULT"
        self.state = None
        self.prev_state = None
        
        self.lane_driving_enter_time = None
        self.lane_driving_delay = rospy.Duration(5.0)  # 5초

        
        self.lock = threading.Lock()
        th = threading.Thread(target=self.keyboard_listener, daemon=True)
        th.start()
        
        self.stanley_debug = True   # TODO
        # self.stanley_debug = False   # TODO
        
        self.rate = rospy.Rate(20)
        
        if self.stanley_debug:
            rospy.logwarn_once("STANLEY DEBUG MODE ENABLED")
            # self.filtered_point = np.array([[-5, 1.0],[-4.5, 2.0]])
            # self.filtered_point = np.array([[-5, 1.0],[-4.5, 2.0]])
            
            self.state = "stanley"
            while not rospy.is_shutdown():
                rospy.loginfo_throttle(1.0, f"state: {self.state}, destination: {np.mean(self.filtered_point, axis=0)[0]:.2f}, {np.mean(self.filtered_point, axis=0)[1]:.2f}")
                self.stanley_path(self.filtered_point) # fixed point
                self.rate.sleep()
                
        else:
            self.run()
    
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

                    
    def run(self):
        
        while not rospy.is_shutdown():
            rospy.loginfo_throttle(0.5, f"state: {self.state}, filtered_point: {self.filtered_point}")

            if self.mode == "DEFAULT":
                self.drive(0,0)

            elif self.mode == "FINAL":
                if self.state == "lane_driving":             # 미션 시작해서 일직선으로 주행하는 단계
                    if not self.first_car_detected:
                        self.drive(self.lane_steer, FORWARD_SPEED)
                    else:
                        self.state = "full_left_steer"
                        self.start_time = None
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
                    
    
    def lane_steer_callback(self, msg): self.lane_steer = msg.data

    def lane_detected_callback(self, msg):
        
        self.lane_detect_queue.append(msg.data)
        true_count = sum(self.lane_detect_queue)

        if not self.is_lane_detected:
            if true_count >= 7:
                self.is_lane_detected = True
        else:
            if true_count <= 3:
                self.is_lane_detected = False
    
    def stanley_steer_callback(self, msg): self.stanley_steer = msg.data
        
    # def ultrasonic2_callback(self, msg): self.ultrasonics[0] = msg.data
    
    # def ultrasonic3_callback(self, msg): self.ultrasonics[1] = msg.data
    
    # def ultrasonic4_callback(self, msg): self.ultrasonics[2] = msg.data
    
    # def ultrasonic5_callback(self, msg): self.ultrasonics[3] = msg.data

    def detection_poses_callback(self, msg):
        point = np.array([[pose.position.x, pose.position.y] for pose in msg.poses],dtype=float)
        
        point_debug = np.asarray(point)

        if point_debug.ndim == 1:
            rospy.logwarn_throttle(
                1.0, "[detection] single point received, skipping 2car check"
            )
            return

        if point_debug.shape[0] < 2:
            rospy.logwarn_throttle(
                1.0, "[detection] insufficient points for 2car 판단"
            )
            return

        if self.state == "lane_driving":
            now = rospy.Time.now()
            
            if self.lane_driving_enter_time is None:
                self.lane_driving_enter_time = now
                
            if now - self.lane_driving_enter_time < self.lane_driving_delay:
                return

            if not self.first_car_detected:
                if not self.is_first_car(point):
                    print("not first car")
                    self.first_car_streak = 0
                    return
                print("first car")
                self.first_car_streak += 1

                if self.first_car_streak >= 5:
                    self.first_car_detected = True
                    rospy.loginfo_once("First car detected")
            
        else:
            self.lane_driving_enter_time = None

            if not self.can_start_parking:
                if not self.is_2car1(point): # TODO 꼭 2개가 아니어도 ROI 내 좌표들의 부호를 가지고 판단할 수 있어야 함
                    self.parking_lot_streak = 0
                    return
                # 2개 보장
                self.parking_lot_streak += 1

                if self.parking_lot_streak >= 3 and self.is_stop_cond(point):
                    self.can_start_parking = True

                    self.filtered_point = point.copy()
                    rospy.loginfo_once("Parking lot detected")

            else: # TODO 점 3~ 4개 나와도 그거 활용하면서 2개 뽑는 로직도 있어야 함

                if not self.is_2car2(point):
                    self.reject_count += 1
                    print("not two car")
                    return
                # 2개 보장
                print("two car")

                dist = self.pair_distance(self.filtered_point, point)
                adaptive_thresh = min(TH_MAX, 0.5 + 0.2 * self.reject_count)

                if dist <= adaptive_thresh:
                    self.filtered_point = point.copy()
                    self.reject_count = 0
                    print(f"destination updated, avg dist: {dist/2:.2f}")
                else:
                    self.reject_count += 1
                    if self.reject_count >= 20:
                        self.filtered_point = point.copy()
                        self.reject_count = 0
                        print("Forced parking point update")
                
                self.stanley_path(self.filtered_point) # debug ＃ TODO
                

    def is_first_car(self, point):
        
        mask = (
            (point[:, 0] >= -1) & (point[:, 0] <= 1) &
            (point[:, 1] >=  -3) & (point[:, 1] <= -1)
        )

        point = point[mask]
        
        if point.shape[0] != 1:
            return False
        

        return True

    def is_2car1(self, point):
        """
        두 클러스터가 '주차된 두 차량'이라고 볼 수 있는지 판정:
        1) 점이 정확히 2개
        2) 두 점 사이 거리 > 0.8m
        """
        r_min = 1.0   # 안쪽 원 반지름
        r_max = 5.0   # 바깥 원 반지름

        r = np.linalg.norm(point, axis=1)

        # 두 원 사이에 있는 점만 선택
        mask = (r >= r_min) & (r <= r_max)
        point = point[mask]
        # TODO ROI 각도 설정 까지 추가
        
        if point.shape[0] != 2:
            return False
        
        dx = point[1, 0] - point[0, 0]
        dy = point[1, 1] - point[0, 1]
        if not 1.5 <= math.sqrt(dx**2 + dy**2) <= 3:
            return False
        
        return True
    
    def is_2car2(self, point):
        point = np.asarray(point)

        # 최소 2개 필요
        if point.ndim != 2 or point.shape[0] < 2:
            print("1")
            return False

        # -------------------------------
        # 🔥 가장 가까운 2개 포인트 선택
        # -------------------------------
        N = point.shape[0]
        min_dist2 = float("inf")
        best_pair = None

        for i in range(N):
            for j in range(i + 1, N):
                dx = point[i, 0] - point[j, 0]
                dy = point[i, 1] - point[j, 1]
                d2 = dx*dx + dy*dy

                if d2 < min_dist2:
                    min_dist2 = d2
                    best_pair = (i, j)

        # 두 개로 point 재정의
        point = point[list(best_pair), :]   # shape (2,2)

        # -------------------------------
        # 기존 로직
        # -------------------------------

        # 각 포인트 5m 이내
        # if (point[:, 0]**2 + point[:, 1]**2 > 5.0**2).any():
        #     print("2")
        #     return False

        # # 두 점 사이 거리
        # dx = point[1, 0] - point[0, 0]
        # dy = point[1, 1] - point[0, 1]
        # dist2 = dx*dx + dy*dy

        # if not (0.8**2 <= dist2 <= 1.5**2):
        #     print("3")
        #     return False

        return True

    
    def is_stop_cond(self, point):
        
        return point[0, 1] * point[1, 1] < 0

    def pair_distance(self, A, B):
        d1 = np.linalg.norm(A[0] - B[0]) + np.linalg.norm(A[1] - B[1])
        d2 = np.linalg.norm(A[0] - B[1]) + np.linalg.norm(A[1] - B[0])
        return min(d1, d2)

    def drive(self, des_steer, long_cmd):
        # return
        self.motor_cmd_steer_pub.publish(Int16(int(des_steer)))
        self.motor_long_pub.publish(Int16(int(long_cmd)))
    
    def stanley_path(self, filtered_point):
        if filtered_point is None or len(filtered_point) < 2:
            rospy.logwarn_throttle(1.0, "[stanley_path] need at least 2 points")
            return None
        dest = np.mean(filtered_point, axis=0)
        v = filtered_point[1] - filtered_point[0]

        n = np.array([-v[1], v[0]])
        n_hat = n / np.linalg.norm(n)

        if np.dot(n_hat, -dest) < 0:
            n_hat = -n_hat

        back_len = 2.0    # 차량 반대 방향
        front_len = 5.0   # 차량 쪽
        step = 0.1

        s_vals = np.arange(-back_len, front_len + step, step)
        path_xy = np.array([dest + s * n_hat for s in s_vals])

        # ---------- 6. Path 메시지 생성 ----------
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

        # ---------- 7. publish ----------
        self.stanley_path_pub.publish(path_msg)
        self.publish_destination_marker(dest)
        self.publish_filtered_points_marker(filtered_point)
        self.publish_filtered_points_line(filtered_point)
        
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

        radius = 0.4        # 원 반지름 (m)
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
        
if __name__ == "__main__":
    try:
        Parking()
    except rospy.ROSInterruptException:
        pass
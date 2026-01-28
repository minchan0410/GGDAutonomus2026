#!/usr/bin/env python3

import rospy, cv2, math
import numpy as np
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32MultiArray, Int16
from cv_bridge import CvBridge

P = 0.1
L = 0.8 # 휠베이스
K = 1
TH_MAX               = 1
FORWARD_SPEED        = 150
BACKWARD_SPEED       = 100
PAUSE_TIME           = 0.5
GOING_LEFT_TIME      = 1.5
GOING_RIGHT_TIME     = 2.5
GOING_STRAIGHT_TIME  = 3
ULTRASONIC_THRESHOLD = 0.5
HIST_THRESHOLD       = 30

class Parking:
    def __init__(self):
        rospy.init_node("parking")

        # =============================
        # 🔴 VIDEO MODE SWITCH
        # =============================
        self.use_video = True
        self.video_path = "/home/kdy/catkin_ws/src/parking/scripts/curv.mp4"

        if not self.use_video:
            rospy.Subscriber("/usb_cam/image_raw", Image, self.usbcam_callback, queue_size=1)
        else:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                rospy.logerr("Failed to open video file")
                exit(1)

        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)

        self.bridge = CvBridge()

        # 상태 변수들
        self.parking_lot_streak = 0
        self.finish_streak = 0
        self.lane_detected = False
        self.to_finish = False
        self.parking_lot_detected = False
        self.mission_completed = False

        self.state = "start"

        self.rate = rospy.Rate(20)

        # =============================
        # 🔴 MAIN LOOP
        # =============================
        while not rospy.is_shutdown() and not self.mission_completed:

            if self.use_video:
                ret, frame = self.cap.read()
                if not ret:
                    rospy.loginfo("Video finished")
                    break
                self.image = frame.copy()

            if self.state == "start":
                if not self.parking_lot_detected:
                    self.lane_detection()

            self.rate.sleep()

            #     else:
            #         self.state = "little_left"
            #         continue
                    
                    
            # if self.state == "little_left":              # 주차 공간 발견 후 왼쪽으로 살짝 꺾어서 각 만드는 단계
                
            #     if self.start_time is None:
            #         self.start_time = rospy.Time.now()
            #         rospy.loginfo_once("start little left")
            
            #     elapsed = (rospy.Time.now() - self.start_time).to_sec()
            #     if elapsed < GOING_LEFT_TIME:
            #         self.drive(15, 100)
            #     else:
            #         self.start_time = rospy.Time.now()
            #         self.state = "pause_after_left"
            #         rospy.loginfo_once("little left moved, pause start")
            #         continue
                    
                    
            # if self.state == "pause_after_left":
            #     elapsed = (rospy.Time.now() - self.start_time).to_sec()
            #     self.drive(0, 0)
            #     if elapsed >= PAUSE_TIME:
            #         self.start_time = None
            #         self.state = "parking"
            #         rospy.loginfo_once("pause finished")
            #         continue
                    
            
            # if self.state == "parking":                 # 후진해서 주차하는 단계
            #     if not self.parked:
            #         self.pure_pursuit()
            #     else:
            #         self.start_time = rospy.Time.now()
            #         self.state = "pause_after_park"
            #         rospy.loginfo_once("parked, pause start")
            #         continue

            
            # if self.state == "pause_after_park":
            #     elapsed = (rospy.Time.now() - self.start_time).to_sec()
            #     self.drive(0, 0)
            #     if elapsed >= PAUSE_TIME:
            #         self.start_time = None
            #         self.state = "little_right"
            #         rospy.loginfo_once("pause finished")
            #         continue
                    
            
            # if self.state == "little_right":              # 주차 공간 나와서 오른쪽으로 꺾어 가이드 선에 정렬하는 단계
                
            #     if self.start_time is None:
            #         self.start_time = rospy.Time.now()
            #         rospy.loginfo_once("start little left")
            
            #     elapsed = (rospy.Time.now() - self.start_time).to_sec()
            #     if elapsed < GOING_RIGHT_TIME and not self.lane_detected:
            #         self.drive(-10, 200)
            #     else:
            #         self.start_time = None
            #         self.state = "go_straight"
            #         rospy.loginfo_once("little right moved")
            #         continue
                    
            
            # if self.state == "go_straight":
            #     if not self.to_finish:
            #         self.lane_detection()
            #     else:
            #         self.state = "to_finish"
            #         continue
            
            
            # if self.state == "to_finish":
            #     if self.start_time is None:
            #         self.start_time = rospy.Time.now()
            #         rospy.loginfo_once("go to finish")
            
            #     elapsed = (rospy.Time.now() - self.start_time).to_sec()
            #     if elapsed < GOING_STRAIGHT_TIME:
            #         self.drive(0, 200)
            #     else:
            #         self.start_time = rospy.Time.now()
            #         self.state = "mission_done"
            #         rospy.loginfo_once("mission done, pause start")
            #         continue
            
            
            # if self.state == "mission_done":
            #     elapsed = (rospy.Time.now() - self.start_time).to_sec()
            #     self.drive(0, 0)
            #     if elapsed >= PAUSE_TIME:
            #         self.start_time = None
            #         self.mission_completed = True
            #         rospy.loginfo_once("pause finished")
                
            self.rate.sleep()
        
    
    # def ultrasonic2_callback(self, msg): self.ultrasonics[0] = msg.data
    
    # def ultrasonic3_callback(self, msg): self.ultrasonics[1] = msg.data
    
    # def ultrasonic4_callback(self, msg): self.ultrasonics[2] = msg.data
    
    # def ultrasonic5_callback(self, msg): self.ultrasonics[3] = msg.data
    
    
    # def centers_callback(self, msg):
        
    #     data = np.array(msg.data, dtype=float)
    
    #     if len(data) % 2 != 0:
    #         self.parking_lot_streak = 0
    #         return

    #     points = data.reshape(-1, 2)

    #     if not self.parking_lot_detected:
    #         if not self.is_2car(points):
    #             self.parking_lot_streak = 0
    #             return

    #         self.parking_lot_streak += 1
            
    #         if self.parking_lot_streak >= 10:
    #             self.parking_lot_detected = True

    #             self.filtered_points = points.copy()
    #             rospy.loginfo_once("Parking lot detected")
            
    #     else:
    #         if not self.is_2car(points):
    #             self.reject_count += 1
    #             return
            
    #         dist = self.pair_distance(self.filtered_points, points)

    #         adaptive_thresh = min(TH_MAX, 0.5 + 0.2 * self.reject_count)

    #         if dist <= adaptive_thresh:
    #             self.filtered_points = points.copy()
    #             self.reject_count = 0
    #             print(f"destination updated, avg dist: {dist/2:.2f}")
    #         else:
    #             self.reject_count += 1
    #             if self.reject_count >= 20:
    #                 self.filtered_points = points.copy()
    #                 self.reject_count = 0
    #                 rospy.logwarn_once("Forced parking point update")


    # def is_2car(self, points):  # 두 차량 사이 최단거리 이상의 두 클러스터이면 주차공간에 배치된 두 차량이라고 간주
    #     """
    #     두 클러스터가 '주차된 두 차량'이라고 볼 수 있는지 판정:
    #     1) 점이 정확히 2개
    #     2) 두 점 사이 거리 > 0.8m
    #     3) 각 점이 원점(차량/센서 기준 local 좌표)에서 5m 이내
    #     4) 각 점의 local y 좌표가 양수 (y > 0)
    #     """
    #     if points.shape[0] != 2:
    #         return False
        
    #     dx = points[1, 0] - points[0, 0]
    #     dy = points[1, 1] - points[0, 1]
    #     if math.sqrt(dx**2 + dy**2) <= 0.8:
    #         return False
        
    #     r0 = math.sqrt(points[0, 0]**2 + points[0, 1]**2)
    #     r1 = math.sqrt(points[1, 0]**2 + points[1, 1]**2)
    #     if (r0 > 5.0) or (r1 > 5.0):
    #         return False

    #     if (points[0, 1] <= 0.0) or (points[1, 1] <= 0.0):
    #         return False

    #     return True
    
    
    # def pair_distance(self, A, B):
    #     d1 = np.linalg.norm(A[0] - B[0]) + np.linalg.norm(A[1] - B[1])
    #     d2 = np.linalg.norm(A[0] - B[1]) + np.linalg.norm(A[1] - B[0])
    #     return min(d1, d2)


    def usbcam_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imshow("original", self.image)
        # cv2.waitKey(1)

        
    def lane_detection(self):
        
        if self.finish_streak >= 20:
            self.to_finish = True
            return
            
        cv_img = self.image
        y, x, _ = cv_img.shape  # 480, 640  curv.mp4 = 720, 720
        # print(f"y: {y}, x: {x}")
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 130])
        white_upper = np.array([179, 60, 255])
        white_filter = cv2.inRange(hsv_img, white_lower, white_upper)

        and_img = cv2.bitwise_and(cv_img, cv_img, mask=white_filter)
        margin_x1 = 0
        margin_x2 = 120
        margin_y = 400

        src_pt1 = (margin_x1, y)
        src_pt2 = (margin_x2, margin_y)
        src_pt3 = (x - margin_x2, margin_y)
        src_pt4 = (x - margin_x1, y)
        src_pts = np.float32([src_pt1, src_pt2, src_pt3, src_pt4])

        dst_margin_x = 120

        dst_pt1 = (dst_margin_x, y)
        dst_pt2 = (dst_margin_x, 0)
        dst_pt3 = (x - dst_margin_x, 0)
        dst_pt4 = (x - dst_margin_x, y)
        dst_pts = np.float32([dst_pt1, dst_pt2, dst_pt3, dst_pt4])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        matrix_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        warp_img = cv2.warpPerspective(and_img, matrix, (x, y))     # BEV
        warp_img_color = cv2.warpPerspective(cv_img, matrix, (x, y))
        warp_hsv = cv2.cvtColor(warp_img_color, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(warp_hsv, white_lower, white_upper)

        bin_img = np.zeros_like(white_mask)
        bin_img[white_mask != 0] = 1
        center_index = x // 2   # 320

        window_num = 8
        margin = 160
        window_y_size = y // window_num  # 60
        indices = []

        for i in range(0, window_num):
            upper_y = y - window_y_size * (i + 1)   # 420, 360, 300, ...
            lower_y = y - window_y_size * i         # 480, 420, 360, ...

            window = bin_img[upper_y:lower_y, :center_index]
            histogram = np.sum(window, axis=0)
            histogram[histogram < HIST_THRESHOLD] = 0

            try:
                nonzero = np.nonzero(histogram)[0]
                
                if len(nonzero) > 0:
                    groups = self.split_contiguous(nonzero)

                    rightmost_group = max(groups, key=lambda g: g[-1])

                    avg_index = (rightmost_group[0] + rightmost_group[-1]) // 2
                    indices.append(avg_index)
                    
                    cv2.line(warp_img, (avg_index, upper_y + window_y_size // 2), (avg_index, upper_y + window_y_size // 2), (0, 0, 255), 10)
                    cv2.rectangle(warp_img, (avg_index - margin, upper_y), (avg_index + margin, lower_y), (255, 0, 0), 3)
            except:
                pass
        
        if len(indices) == 0:
            avg_indices = 320
            direction = 'forward'
            error_index = 0
            self.lane_detected = False
            if self.state == "go_straight":
                self.finish_streak += 1
                
        else:
            avg_indices = int(np.average(indices))
            center_index = margin
            error_index = center_index - avg_indices
            direction = 'left' if error_index > 0 else 'right'
            self.lane_detected = True
            self.finish_streak = 0
            
            
        cv2.line(warp_img, (avg_indices, 0), (avg_indices, y), (0, 255, 255), 3)    # yellow
        cv2.putText(
            warp_img,
            "cur",
            (avg_indices + 10, 30),        # x는 선 오른쪽, y는 위쪽
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                           # 글자 크기
            (0, 255, 255),                 # yellow
            2,
            cv2.LINE_AA
        )
        cv2.line(warp_img, (center_index, 0), (center_index, y), (0, 255, 0), 3)    # green
        cv2.putText(
            warp_img,
            "purpose",
            (center_index + 10, 60),        # x는 선 오른쪽, y는 위쪽
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                           # 글자 크기
            (0, 255, 0),                   # green
            2,
            cv2.LINE_AA
        )
        self.lkas_steer = error_index * P
        
        
        rospy.loginfo_throttle(0.2, f"lane found: {len(indices) != 0}, direction: {direction}, steering: {self.lkas_steer:.2f}")
        rospy.loginfo_throttle(0.2, f"avg: {avg_indices}, purpose: {center_index}")
        

        self.drive(self.lkas_steer, FORWARD_SPEED)

        warp_inv_img = cv2.warpPerspective(warp_img, matrix_inv, (x, y))
        cv2.circle(cv_img, src_pt1, 10, (255, 0, 0), -1)
        cv2.circle(cv_img, src_pt2, 10, (0, 255, 0), -1)
        cv2.circle(cv_img, src_pt3, 10, (0, 0, 255), -1)
        cv2.circle(cv_img, src_pt4, 10, (0, 255, 255), -1)
        cv2.imshow("cv_img", cv_img)                # 원본 카메라 + ROI 네 꼭짓점
        # cv2.imshow("and_img", and_img)              # 색 필터링 결과
        cv2.imshow("warp_img_color", warp_img_color)# 색 유지 BEV
        
        cv2.putText(
            warp_img,
            f"desired steer: {self.lkas_steer:.2f}",
            (x - 400, 400),                  # 우측 상단
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,                            # 글자 크기
            (0, 0, 255),                    # 빨간색 (BGR)
            2,
            cv2.LINE_AA
        )
        cv2.imshow("warp_img", warp_img)            # BEV + 슬라이딩 윈도우(박스, 중심점, 평균차선위치)
        # cv2.imshow("warp_inv_img", warp_inv_img)    # BEV만 역변환한 것

        cv2.waitKey(1)


    def split_contiguous(self, indices):
        groups = []
        current = [indices[0]]

        for i in indices[1:]:
            if i == current[-1] + 1:
                current.append(i)
            else:
                groups.append(current)
                current = [i]

        groups.append(current)
        return groups 
    
    
    def drive(self, des_steer, long_cmd):
        self.motor_cmd_steer_pub.publish(Int16(des_steer))
        self.motor_long_pub.publish(Int16(long_cmd))
    

    def pure_pursuit(self):
            
        dest_x, dest_y = np.mean(self.filtered_points, axis=0)
        Ld2 = dest_x**2 + dest_y**2
        curvature = 2 * dest_y / Ld2
        
        if Ld2 < 1e-2:
            des_steer = 0.0
        else:
            des_steer = - math.atan(L * curvature) * K
            
        self.drive(des_steer, BACKWARD_SPEED)
        
        if all(x < ULTRASONIC_THRESHOLD for x in self.ultrasonics):
            self.parked_streak += 1
            if self.parked_streak >= 20:
                self.parked = True
        else:
            self.parked_streak = 0
        

if __name__ == "__main__":
    try:
        Parking()
    except rospy.ROSInterruptException:
        pass
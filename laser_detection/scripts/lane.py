#!/usr/bin/env python3

import rospy, cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Bool
from cv_bridge import CvBridge


P = 0.1
HIST_THRESHOLD = 30

class ParkingLane:
    def __init__(self):
        rospy.init_node("parking_lane")
        
        rospy.Subscriber("/cam1/usb_cam/image_raw", Image, self.usbcam_callback, queue_size=1)
        self.lane_steer_pub = rospy.Publisher("/parking_lane_steer", Int16, queue_size=1)
        self.lane_detected_pub = rospy.Publisher("/is_lane_detected", Bool, queue_size=1)
        self.bridge = CvBridge()
        self.lane_detected = False
        
        self.rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            # print("sd")
            self.lane_detection()
            self.rate.sleep()
        
    def usbcam_callback(self, msg):
        print("111")
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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
    
    def lane_detection(self):
        if not hasattr(self, "image"):
            # print("qqq")
            return
        cv_img = self.image
        y, x, _ = cv_img.shape  # 480, 640
        print(f"y: {y}, x: {x}")
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        white_lower = np.array([0, 0, 130])
        white_upper = np.array([179, 60, 255])
        white_filter = cv2.inRange(hsv_img, white_lower, white_upper)

        and_img = cv2.bitwise_and(cv_img, cv_img, mask=white_filter)
        margin_x1 = 0
        margin_x2 = 240 # TODO
        margin_y = 100

        src_pt1 = (margin_x1, y-100)
        src_pt2 = (margin_x2, margin_y)
        src_pt3 = (x - margin_x2, margin_y)
        src_pt4 = (x - margin_x1, y-100)
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
        margin = 320
        window_y_size = y // window_num  # 60
        indices = []

        for i in range(0, window_num):
            upper_y = y - window_y_size * (i + 1)   # 420, 360, 300, ...
            lower_y = y - window_y_size * i         # 480, 420, 360, ...

            window = bin_img[upper_y:lower_y, :center_index + int(4/x)]
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
            error_index = 0
            self.lane_detected_pub.publish(False)
        else:
            avg_indices = int(np.average(indices))
            center_index = margin
            error_index = center_index - avg_indices
            self.lane_detected_pub.publish(True)
            
            
            
        cv2.line(warp_img, (avg_indices, 0), (avg_indices, y), (0, 255, 255), 3)    # yellow
        cv2.putText(
            warp_img,
            "cur",
            (avg_indices + 10, 30),        
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                           
            (0, 255, 255),                 
            2,
            cv2.LINE_AA
        )
        cv2.line(warp_img, (center_index, 0), (center_index, y), (0, 255, 0), 3)    # green
        cv2.putText(
            warp_img,
            "purpose",
            (center_index + 10, 60),       
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,                           
            (0, 255, 0),                   
            2,
            cv2.LINE_AA
        )
        
        self.lkas_steer = max(min(error_index * P, 22.5), -22.5)
        # self.lane_steer_pub.publish(int(self.lkas_steer))
        self.lane_steer_pub.publish(int(0))
        rospy.loginfo_throttle(0.3, f"lane found: {len(indices) != 0}, steering: {self.lkas_steer:.2f}\n")

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

if __name__ == "__main__":
    try:
        ParkingLane()
    except rospy.ROSInterruptException:
        pass
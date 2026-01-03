#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int16
import cv2
import numpy as np
import os
import math
from sklearn.cluster import DBSCAN

class LaneDetector:
    def __init__(self):
        # 1. ROS 노드 초기화
        rospy.init_node('lane_detector_node', anonymous=False)
        
        # 2. Publisher 설정 (Int16 타입)
        self.angle_pub = rospy.Publisher('/lane_angle', Int16, queue_size=10)
        
        # 3. 설정 변수들
        self.height_usage_ratio = 0.4
        self.bottom_shrink_ratio = 0.68
        self.dist_weight = 0.05
        self.dbscan_eps = 25
        self.cluster_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
        
        # 4. 비디오 소스 설정 (파일 경로 또는 카메라 인덱스 0)
        # 실제 주행 시에는 0 또는 '/dev/video0' 등을 사용
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.video_path = os.path.join(BASE_DIR, "runnig_data", "curv.mp4")
        
        # 파일이 없으면 웹캠(0) 사용 시도
        if not os.path.exists(self.video_path):
            rospy.logwarn(f"파일을 찾을 수 없습니다: {self.video_path}. 웹캠(0)을 시도합니다.")
            self.video_source = 0
        else:
            self.video_source = self.video_path

    # --- 헬퍼 함수들 ---
    def calculate_weighted_average_angle(self, data_list):
        if not data_list: return None
        
        total_weighted_angle = 0.0
        total_weight = 0.0
        
        for angle, weight in data_list:
            w = weight 
            total_weighted_angle += angle * w
            total_weight += w
            
        if total_weight == 0: return 0.0
        
        return total_weighted_angle / total_weight

    def calc_dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            rospy.logerr("영상을 열 수 없습니다.")
            return

        # --- BEV 초기 설정 (한 번만 수행) ---
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if w == 0 or h == 0:
            rospy.logerr("영상 크기를 읽을 수 없습니다.")
            return

        roi_h = int(h * self.height_usage_ratio)
        start_y = h - roi_h
        src_pts = np.float32([[0, 0], [w, 0], [w, roi_h], [0, roi_h]])
        shrink_pixel = int(w * self.bottom_shrink_ratio / 2)
        dst_pts = np.float32([[0, 0], [w, 0], [w - shrink_pixel, roi_h], [shrink_pixel, roi_h]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        arrow_start_pt = (w // 2, roi_h - 50) 
        arrow_len = 100
        current_avg_angle = 0.0
        
        rate = rospy.Rate(30) # 30Hz 루프

        rospy.loginfo("Lane Detection Node Started.")

        # --- 메인 루프 ---
        while not rospy.is_shutdown() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 영상이 끝나면 반복할지, 종료할지 결정 (여기선 루프 종료)
                rospy.loginfo("영상 종료")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 무한 반복 원할 시 주석 해제
                continue

            roi_img = frame[start_y:h, 0:w]
            bev_img = cv2.warpPerspective(roi_img, M, (w, roi_h), flags=cv2.INTER_LINEAR)

            gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 5)
            edges = cv2.Canny(blur, 50, 150)

            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
            edges = cv2.bitwise_and(edges, edges, mask=mask)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=30, maxLineGap=20)
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            bev_viz = bev_img.copy()

            largest_cluster_data = [] 

            if lines is not None:
                data_angles = []
                valid_indices = []
                valid_line_coords = [] 

                for i, line in enumerate(lines):
                    x1, y1, x2, y2 = line[0]

                    if y1 > y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1

                    dx = x1 - x2
                    dy = y2 - y1 

                    if dy == 0: 
                        angle = 90.0 if dx > 0 else -90.0
                    else:
                        angle = math.degrees(math.atan2(dx, dy))

                    if abs(angle) > 80: continue
                    
                    data_angles.append(angle)
                    valid_indices.append(i)
                    valid_line_coords.append(((x1, y1), (x2, y2)))

                num_samples = len(data_angles)
                if num_samples > 0:
                    angles_np = np.array(data_angles)
                    angle_diff_matrix = np.abs(angles_np[:, None] - angles_np[None, :])
                    
                    dist_matrix_spatial = np.zeros((num_samples, num_samples))
                    for i in range(num_samples):
                        for j in range(i + 1, num_samples):
                            p1_a, p2_a = valid_line_coords[i]
                            p1_b, p2_b = valid_line_coords[j]
                            
                            d1 = self.calc_dist(p1_a, p1_b) + self.calc_dist(p2_a, p2_b)
                            d2 = self.calc_dist(p1_a, p2_b) + self.calc_dist(p2_a, p1_b)
                            final_dist = min(d1, d2)
                            dist_matrix_spatial[i][j] = final_dist
                            dist_matrix_spatial[j][i] = final_dist

                    combined_matrix = angle_diff_matrix + (dist_matrix_spatial * self.dist_weight)
                    
                    db = DBSCAN(eps=self.dbscan_eps, min_samples=3, metric='precomputed').fit(combined_matrix)
                    labels = db.labels_

                    unique_labels = set(labels)
                    if -1 in unique_labels: unique_labels.remove(-1)
                    
                    if len(unique_labels) > 0:
                        label_counts = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
                        sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
                        rank_map = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

                        for idx, label in enumerate(labels):
                            line_idx = valid_indices[idx]
                            x1, y1, x2, y2 = lines[line_idx][0]
                            angle = data_angles[idx]

                            if label == -1:
                                cv2.line(bev_viz, (x1, y1), (x2, y2), (100, 100, 100), 1)
                            else:
                                rank = rank_map[label]
                                color = self.cluster_colors[rank % len(self.cluster_colors)]
                                
                                thickness = 3 if rank == 0 else 1
                                cv2.line(edges_color, (x1, y1), (x2, y2), color, 2)
                                cv2.line(bev_viz, (x1, y1), (x2, y2), color, thickness)
                                
                                if rank == 0:
                                    mid_y = (y1 + y2) / 2.0
                                    largest_cluster_data.append((angle, mid_y))

            # 각도 계산 및 업데이트
            weighted_avg_angle = self.calculate_weighted_average_angle(largest_cluster_data)
            if weighted_avg_angle is not None:
                current_avg_angle = weighted_avg_angle

            # --- [중요] ROS Topic Publish (Int16) ---
            angle_msg = Int16()
            angle_msg.data = int(current_avg_angle)
            self.angle_pub.publish(angle_msg)

            # --- 시각화 ---
            angle_rad = math.radians(current_avg_angle)
            arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
            arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
            
            cv2.arrowedLine(edges_color, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
            cv2.arrowedLine(bev_viz, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
            
            cv2.putText(bev_viz, f"Avg Angle: {int(current_avg_angle)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            combined_result = cv2.hconcat([edges_color, bev_viz])
            cv2.imshow("Lane Detection Result", combined_result)

            # ROS는 waitKey(0)을 쓰면 멈춥니다. waitKey(1)로 변경해야 합니다.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 루프 속도 조절
            rate.sleep()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        lane_detector = LaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass
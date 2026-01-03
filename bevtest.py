import cv2
import numpy as np
import os
import math
from sklearn.cluster import DBSCAN

# [수정됨] 가중 평균 계산 함수
# data_list: [(angle1, weight1), (angle2, weight2), ...] 형식
def calculate_weighted_average_angle(data_list):
    if not data_list: return None
    
    total_weighted_angle = 0.0
    total_weight = 0.0
    
    for angle, weight in data_list:
        # 가중치(y좌표)를 제곱하면 하단 직선의 영향력을 더 극대화할 수 있음.
        # 여기서는 단순히 y좌표(weight)를 그대로 사용하여 선형적인 가중치를 줌.
        w = weight 
        total_weighted_angle += angle * w
        total_weight += w
        
    if total_weight == 0: return 0.0
    
    return total_weighted_angle / total_weight

# 두 점 사이의 거리 계산 헬퍼 함수
def calc_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def run_lane_detection_arrow_on_bev(video_source, height_usage_ratio=0.6, bottom_shrink_ratio=0.73):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[에러] 영상을 열 수 없습니다.")
        return

    # --- BEV 설정 ---
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_h = int(h * height_usage_ratio)
    start_y = h - roi_h
    src_pts = np.float32([[0, 0], [w, 0], [w, roi_h], [0, roi_h]])
    shrink_pixel = int(w * bottom_shrink_ratio / 2)
    dst_pts = np.float32([[0, 0], [w, 0], [w - shrink_pixel, roi_h], [shrink_pixel, roi_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    cluster_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    arrow_start_pt = (w // 2, roi_h - 50) 
    arrow_len = 100
    
    current_avg_angle = 0.0
    dist_weight = 0.05 
    dbscan_eps = 25 

    print("--- [Space] 다음 프레임 / [q] 종료 ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

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

        # [수정됨] 각도뿐만 아니라 무게중심(y)도 저장할 리스트
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
                        d1 = calc_dist(p1_a, p1_b) + calc_dist(p2_a, p2_b)
                        d2 = calc_dist(p1_a, p2_b) + calc_dist(p2_a, p1_b)
                        final_dist = min(d1, d2)
                        dist_matrix_spatial[i][j] = final_dist
                        dist_matrix_spatial[j][i] = final_dist

                combined_matrix = angle_diff_matrix + (dist_matrix_spatial * dist_weight)
                
                db = DBSCAN(eps=dbscan_eps, min_samples=3, metric='precomputed').fit(combined_matrix)
                labels = db.labels_

                unique_labels = set(labels)
                if -1 in unique_labels: unique_labels.remove(-1)
                
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
                        color = cluster_colors[rank % len(cluster_colors)]
                        
                        thickness = 3 if rank == 0 else 1
                        cv2.line(edges_color, (x1, y1), (x2, y2), color, 2)
                        cv2.line(bev_viz, (x1, y1), (x2, y2), color, thickness)
                        
                        # [수정됨] 랭크 0인 경우 각도와 가중치(y좌표) 저장
                        if rank == 0:
                            # 직선의 중심 y좌표를 가중치로 사용
                            # y가 클수록(화면 아래쪽일수록) 가중치 높음
                            mid_y = (y1 + y2) / 2.0
                            largest_cluster_data.append((angle, mid_y))

        # [수정됨] 가중 평균 함수 호출
        weighted_avg_angle = calculate_weighted_average_angle(largest_cluster_data)
        if weighted_avg_angle is not None:
            current_avg_angle = weighted_avg_angle

        # --- 화살표 시각화 (동일) ---
        angle_rad = math.radians(current_avg_angle)
        arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
        
        cv2.arrowedLine(edges_color, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.arrowedLine(bev_viz, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        
        cv2.putText(bev_viz, f"Avg Angle: {current_avg_angle:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        combined_result = cv2.hconcat([edges_color, bev_viz])
        cv2.imshow("res", combined_result)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(BASE_DIR, "runnig_data", "curv.mp4")
    
    if not os.path.exists(video_path):
        print(f"[경고] 파일을 찾을 수 없습니다: {video_path}")
    else:
        run_lane_detection_arrow_on_bev(video_path, 0.4, 0.68)                 
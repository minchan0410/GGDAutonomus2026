import cv2
import numpy as np
import os
import math
from sklearn.cluster import DBSCAN

# 각도 평균 계산 함수 (단순 산술 평균)
def calculate_average_angle(angles_deg):
    if not angles_deg: return None
    return sum(angles_deg) / len(angles_deg)

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

    # 화살표 설정 (BEV 이미지 기준 하단 중앙)
    arrow_start_pt = (w // 2, roi_h - 50) 
    arrow_len = 100
    
    # 초기값: 0도 (정면)
    current_avg_angle = 0.0

    print("--- [Space] 다음 프레임 / [q] 종료 ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. BEV & 전처리
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

        largest_cluster_angles = []

        # 2. 각도 계산 및 DBSCAN
        if lines is not None:
            data_angles = []
            valid_indices = []

            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]

                # (x1, y1)을 항상 화면 상단(멀리 있는 점, y값이 작은 점)으로 배치
                # (x2, y2)는 화면 하단(가까운 점, y값이 큰 점)
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1

                # 기준: 정면(위쪽)이 0도, 오른쪽이 +, 왼쪽이 -
                # dx: 상단 x - 하단 x (오른쪽으로 기울면 양수, 왼쪽이면 음수)
                # dy: 하단 y - 상단 y (항상 양수, 선분의 높이)
                dx = x1 - x2
                dy = y2 - y1 # y축은 아래로 증가하므로 하단값 - 상단값

                if dy == 0: 
                    # 완전 수평선인 경우 (거의 없겠지만 예외처리)
                    angle = 90.0 if dx > 0 else -90.0
                else:
                    # atan2(dx, dy) -> dy(수직)가 기준축이 됨
                    angle = math.degrees(math.atan2(dx, dy))

                # 너무 수평에 가까운 선(노이즈) 제거 (예: +- 80도 이상은 무시)
                if abs(angle) > 80: continue
                
                data_angles.append(angle)
                valid_indices.append(i)

            if len(data_angles) > 0:
                angles_np = np.array(data_angles)
                # 각도 차이 계산 (이제 -90 ~ 90 범위이므로 단순 차이 사용)
                diff_matrix = np.abs(angles_np[:, None] - angles_np[None, :])
                
                # eps=20 (이전 요청사항 유지)
                db = DBSCAN(eps=20, min_samples=5, metric='precomputed').fit(diff_matrix)
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
                        cv2.line(edges_color, (x1, y1), (x2, y2), color, 2)
                        cv2.line(bev_viz, (x1, y1), (x2, y2), color, 3)
                        
                        # 가장 큰 클러스터의 각도만 수집
                        if rank == 0: 
                            largest_cluster_angles.append(angle)

        # ==================================================
        # [수정됨] 화살표 그리기 (새로운 좌표계 적용)
        # ==================================================
        avg_angle = calculate_average_angle(largest_cluster_angles)
        if avg_angle is not None:
            current_avg_angle = avg_angle

        # 디버깅용 출력
        # print(f"Current Steering Angle: {current_avg_angle:.2f} deg (Neg: Left, Pos: Right)")

        # 각도를 라디안으로 변환
        angle_rad = math.radians(current_avg_angle)

        # 끝점 계산
        # 0도일 때: sin(0)=0 (x변화 없음), cos(0)=1 (y 감소 = 위로)
        # +각도(우회전)일 때: sin(+), cos(+) -> x 증가(우), y 감소(위) -> 우상단
        # -각도(좌회전)일 때: sin(-), cos(+) -> x 감소(좌), y 감소(위) -> 좌상단
        arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
        
        arrow_end_pt = (arrow_end_x, arrow_end_y)

        cv2.arrowedLine(edges_color, arrow_start_pt, arrow_end_pt, (0, 255, 255), 3, tipLength=0.3)
        cv2.arrowedLine(bev_viz, arrow_start_pt, arrow_end_pt, (0, 255, 255), 3, tipLength=0.3)
        
        # 텍스트로 각도 표시
        cv2.putText(bev_viz, f"Angle: {current_avg_angle:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # ==================================================

        combined_result = cv2.hconcat([edges_color, bev_viz])
        cv2.imshow("res", combined_result)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:\Users\VIC_26\Desktop\Gyeonggi_AutoDriving_SW_Competition\advanced_exercise\curv.mp4"
    if isinstance(video_path, str) and not os.path.exists(video_path):
        print(f"[경고] 파일을 찾을 수 없습니다.")
    else:
        run_lane_detection_arrow_on_bev(video_path, 0.4, 0.68)                     
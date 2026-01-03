import cv2
import numpy as np
import os
import math
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# --- [설정] ---
RANSAC_RESIDUAL_THRESHOLD = 10.0 
POLY_DEGREE = 2  # 2차 곡선 (S자 코스라면 3으로 변경)

def calculate_angle_from_derivative(derivative):
    """
    곡선의 접선 기울기(미분값)를 받아 각도로 변환
    derivative (dx/dy): y가 1픽셀 증가할 때(아래로 갈 때) x가 얼마나 변하는가
    """
    # dx/dy 가 바로 이전 코드의 slope와 같은 개념입니다.
    return math.degrees(math.atan2(-derivative, 1))

def fit_polynomial_ransac(y_coords, x_coords, img_height):
    """
    RANSAC으로 다항식 곡선 피팅 (x = ay^2 + by + c)
    """
    if len(y_coords) < 10: return None, None, None

    # 입력: y좌표, 출력: x좌표
    Y = y_coords.reshape(-1, 1)
    X = x_coords.reshape(-1, 1)

    # 파이프라인: 다항식 변환 -> RANSAC(선형회귀)
    # RANSACRegressor 내부에 estimator를 넣어서 다항회귀를 구현합니다.
    # 하지만 더 직관적인 제어를 위해 데이터를 먼저 변환하고 RANSAC을 돌립니다.
    
    poly_features = PolynomialFeatures(degree=POLY_DEGREE)
    Y_poly = poly_features.fit_transform(Y) # [1, y, y^2] 형태로 변환됨

    ransac = RANSACRegressor(residual_threshold=RANSAC_RESIDUAL_THRESHOLD, random_state=0)
    
    try:
        ransac.fit(Y_poly, X)
    except ValueError:
        return None, None, None

    # --- 시각화를 위한 곡선 점 생성 ---
    # y값 0부터 height까지 균등하게 생성
    plot_y = np.linspace(0, img_height, num=img_height).reshape(-1, 1)
    plot_y_poly = poly_features.transform(plot_y)
    predicted_x = ransac.predict(plot_y_poly)

    # 그리기 좋게 (x, y) 포인트 배열로 변환
    curve_points = np.column_stack((predicted_x, plot_y)).astype(np.int32)

    # --- 차량 바로 앞(이미지 하단, y=img_height)에서의 기울기 계산 ---
    # 모델 계수: c + b*y + a*y^2 ...
    # estimator_.coef_ 는 [0, b, a] 순서 (intercept 별도)
    coeffs = ransac.estimator_.coef_.flatten() # [0, 1차항계수, 2차항계수]
    
    # 2차 곡선 미분: dx/dy = b + 2ay
    # 3차 곡선 미분: dx/dy = b + 2ay + 3cy^2
    
    derivative = 0
    # coeffs[0]은 상수항(항상 0 for poly features bias), coeffs[1]이 1차, coeffs[2]가 2차...
    # 주의: fit_intercept=True(기본값)이면 intercept_에 상수가 있고 coef_는 1차항부터 시작일 수 있음.
    # scikit-learn 버전에 따라 다르지만 보통 PolynomialFeatures(include_bias=True) 쓰면:
    # Y_poly는 [1, y, y^2]
    # RANSAC은 fit_intercept=False로 해야 중복 안됨. 
    # 편의상 수동 미분 대신 예측된 포인트의 마지막 두 점으로 기울기 근사
    
    dx = curve_points[-1][0] - curve_points[-5][0] # 하단 끝점과 5픽셀 위 점 비교
    dy = curve_points[-1][1] - curve_points[-5][1] # 보통 5
    
    if dy == 0: current_derivative = 0
    else: current_derivative = dx / dy

    return curve_points, current_derivative, ransac

def run_lane_detection_polynomial(video_source, height_usage_ratio=0.6, bottom_shrink_ratio=0.73):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened(): return

    # --- BEV 설정 (이전과 동일) ---
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_h = int(h * height_usage_ratio)
    start_y = h - roi_h
    src_pts = np.float32([[0, 0], [w, 0], [w, roi_h], [0, roi_h]])
    shrink_pixel = int(w * bottom_shrink_ratio / 2)
    dst_pts = np.float32([[0, 0], [w, 0], [w - shrink_pixel, roi_h], [shrink_pixel, roi_h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    arrow_start_pt = (w // 2, roi_h - 50)
    arrow_len = 100
    current_avg_angle = 0.0

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

        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        bev_viz = bev_img.copy()

        # 데이터 추출
        nonzero = np.argwhere(edges > 0)
        all_y = nonzero[:, 0]
        all_x = nonzero[:, 1]
        mid_x = w // 2
        
        left_mask = all_x < mid_x
        right_mask = all_x >= mid_x

        detected_angles = []

        # [왼쪽 차선] 2차 곡선 피팅
        curve_l, deriv_l, _ = fit_polynomial_ransac(all_y[left_mask], all_x[left_mask], roi_h)
        if curve_l is not None:
            # cv2.polylines를 사용하여 곡선 그리기
            cv2.polylines(edges_color, [curve_l], False, (0, 0, 255), 3) 
            cv2.polylines(bev_viz, [curve_l], False, (0, 0, 255), 3)
            detected_angles.append(calculate_angle_from_derivative(deriv_l))

        # [오른쪽 차선] 2차 곡선 피팅
        curve_r, deriv_r, _ = fit_polynomial_ransac(all_y[right_mask], all_x[right_mask], roi_h)
        if curve_r is not None:
            cv2.polylines(edges_color, [curve_r], False, (0, 255, 0), 3)
            cv2.polylines(bev_viz, [curve_r], False, (0, 255, 0), 3)
            detected_angles.append(calculate_angle_from_derivative(deriv_r))

        # 각도 평균 및 화살표
        if detected_angles:
            current_avg_angle = sum(detected_angles) / len(detected_angles)
        
        angle_rad = math.radians(current_avg_angle)
        arrow_end_x = int(arrow_start_pt[0] + arrow_len * math.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - arrow_len * math.cos(angle_rad))
        
        cv2.arrowedLine(bev_img, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(bev_img, f"Curve Angle: {current_avg_angle:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        combined_result = cv2.hconcat([edges_color, bev_viz])
        cv2.imshow("Polynomial RANSAC", combined_result)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r"C:\Users\VIC_26\Desktop\Gyeonggi_AutoDriving_SW_Competition\advanced_exercise\curv.mp4"
    run_lane_detection_polynomial(video_path, 0.4, 0.68)
import cv2
import os
import numpy as np

# 현재 파일 위치 기준 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(current_dir, 'checkerboard.jpg')
SAVE_H_PATH = os.path.join(current_dir, 'bev_matrix.npy')
SAVE_SIZE_PATH = os.path.join(current_dir, 'bev_size.npy')

# --- [설정] 사용자 환경에 맞게 수정하세요 ---
CHECKERBOARD = (8, 5)  # 체커보드 코너 개수
TOP_CROP_RATIO = 0.40  # 상단 몇 %를 날릴 것인지 (0.45 = 상단 45% 잘라냄)
SCALE = 10             # BEV 변환 시 격자 하나의 픽셀 크기 (해상도)

# [추가됨] 이미지가 뒤집혀 나온다면 True로 설정하세요.
ROTATE_180 = True      
# ----------------------------------------

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {IMAGE_PATH}")
        return
    
    h_orig, w_orig = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 체커보드 검출
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print("체커보드 검출 성공!")
        
        # 2. 목적지 좌표(World Coordinate) 생성
        dst_pts = []
        for i in range(CHECKERBOARD[1]):
            for j in range(CHECKERBOARD[0]):
                dst_pts.append([j * SCALE, i * SCALE])
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # 3. 초기 호모그래피 행렬 계산
        H_init, _ = cv2.findHomography(corners, dst_pts)

        # ---------------------------------------------------------
        # 4. 스마트 캔버스 크기 계산 (ROI 기반)
        # ---------------------------------------------------------
        crop_h = int(h_orig * TOP_CROP_RATIO) 
        
        roi_corners = np.array([
            [0, crop_h],        # 좌상
            [w_orig, crop_h],   # 우상
            [w_orig, h_orig],   # 우하
            [0, h_orig]         # 좌하
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed_corners = cv2.perspectiveTransform(roi_corners, H_init)

        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # 5. 평행 이동 행렬 (Translation Matrix)
        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        H_final = translation_matrix @ H_init
        
        # 최종 도화지 크기
        new_w = int(np.ceil(x_max - x_min))
        new_h = int(np.ceil(y_max - y_min))

        # ---------------------------------------------------------
        # [수정] 6. 180도 회전 보정 (뒤집힘 해결)
        # ---------------------------------------------------------
        if ROTATE_180:
            print(">> 180도 회전 보정을 적용합니다.")
            # 이미지 중심을 기준으로 180도 돌리는 것이 아니라,
            # 좌표축을 (W-x, H-y)로 뒤집는 행렬을 곱합니다.
            rotation_matrix = np.array([
                [-1,  0, new_w],  # x축 반전 후 w만큼 이동
                [ 0, -1, new_h],  # y축 반전 후 h만큼 이동
                [ 0,  0,     1]
            ])
            
            # 최종 행렬에 회전 행렬을 추가로 곱함
            H_final = rotation_matrix @ H_final

        # 7. 결과 저장
        np.save(SAVE_H_PATH, H_final)
        np.save(SAVE_SIZE_PATH, np.array([new_w, new_h]))
        print(f"\n[완료] 결과 저장됨")
        print(f" - Matrix Path: {SAVE_H_PATH}")
        print(f" - Target Size: {new_w} x {new_h}")

        # 8. 미리보기
        preview = cv2.warpPerspective(img, H_final, (new_w, new_h))
        
        display_h = 400
        display_w = int(display_h * new_w / new_h) if new_h > 0 else 400
        
        cv2.imshow('Smart Cropped BEV', cv2.resize(preview, (display_w, display_h)))
        
        print("\n아무 키나 누르면 종료합니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("검출 실패! 코너 개수나 이미지를 확인하세요.")

if __name__ == "__main__":
    main()
import cv2
import os
import numpy as np

# 현재 파일 위치 기준 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(current_dir, 'checkerboard.jpg')
SAVE_H_PATH = os.path.join(current_dir, 'bev_matrix.npy')
SAVE_SIZE_PATH = os.path.join(current_dir, 'bev_size.npy')

# --- [설정] 사용자 환경에 맞게 수정하세요 ---
CHECKERBOARD = (8, 6)  # 체커보드 코너 개수
TOP_CROP_RATIO = 0.50  # 상단 몇 %를 날릴 것인지 (0.45 = 상단 45% 잘라냄)
SCALE = 10            # BEV 변환 시 격자 하나의 픽셀 크기 (해상도)
# ----------------------------------------

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {IMAGE_PATH}")
        return
    
    h_orig, w_orig = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 체커보드 검출 (자르지 않은 원본에서 수행하여 정확도 확보)
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
        # 4. [수정됨] 스마트 캔버스 크기 계산 (ROI 기반)
        # ---------------------------------------------------------
        # 전체 이미지(0,0)를 변환하면 상단이 무한대로 늘어나므로,
        # 우리가 실제로 쓸 '하단 영역'의 모서리만 변환해서 크기를 잽니다.
        
        crop_h = int(h_orig * TOP_CROP_RATIO) # 자르기 시작할 Y 위치
        
        # 관심 영역(ROI)의 네 모서리 좌표 정의
        # [좌상, 우상, 우하, 좌하] 순서 (좌상은 0,0이 아니라 0, crop_h 입니다)
        roi_corners = np.array([
            [0, crop_h],        # 좌상 (Crop Line)
            [w_orig, crop_h],   # 우상 (Crop Line)
            [w_orig, h_orig],   # 우하 (Bottom)
            [0, h_orig]         # 좌하 (Bottom)
        ], dtype=np.float32).reshape(-1, 1, 2)

        # 이 모서리들이 BEV 상에서 어디로 가는지 계산
        transformed_corners = cv2.perspectiveTransform(roi_corners, H_init)

        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        # 변환된 좌표들의 최소/최대값 찾기
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # 5. 평행 이동 행렬 (Translation Matrix)
        # ROI의 가장 왼쪽, 가장 위쪽이 (0,0)에 오도록 당겨줍니다.
        # 이렇게 하면 Crop Line 위쪽 영역은 음수 좌표가 되어 잘려 나갑니다.
        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        # 최종 행렬 = 이동 행렬 @ 초기 행렬
        H_final = translation_matrix @ H_init
        
        # 최종 도화지 크기 계산
        new_w = int(np.ceil(x_max - x_min))
        new_h = int(np.ceil(y_max - y_min))

        # ---------------------------------------------------------

        # 6. 결과 저장
        np.save(SAVE_H_PATH, H_final)
        np.save(SAVE_SIZE_PATH, np.array([new_w, new_h]))
        print(f"\n[완료] 결과 저장됨")
        print(f" - Matrix Path: {SAVE_H_PATH}")
        print(f" - Target Size: {new_w} x {new_h}")
        print(f" - Crop Ratio : 상단 {TOP_CROP_RATIO*100}% 제거됨")

        # 7. 미리보기
        # warpPerspective는 원본 이미지를 넣지만, H_final에 이동 정보가 있어서
        # 자동으로 상단은 잘리고 하단 영역만 new_w, new_h 안에 들어옵니다.
        preview = cv2.warpPerspective(img, H_final, (new_w, new_h))
        
        # 화면에 꽉 차면 보기 힘드므로 리사이즈해서 표시
        display_h = 600
        display_w = int(display_h * new_w / new_h)
        cv2.imshow('Smart Cropped BEV', cv2.resize(preview, (display_w, display_h)))
        
        print("\n아무 키나 누르면 종료합니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("검출 실패! 코너 개수나 이미지를 확인하세요.")

if __name__ == "__main__":
    main()
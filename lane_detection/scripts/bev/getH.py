import cv2
import os
import numpy as np

# 현재 파일(getH.py)의 위치를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(current_dir, 'checkerboard.jpg') # 사진 파일명 확인 필수
SAVE_H_PATH = os.path.join(current_dir, 'bev_matrix.npy')
SAVE_SIZE_PATH = os.path.join(current_dir, 'bev_size.npy')

# 본인의 체커보드 내부 점(코너) 개수
CHECKERBOARD = (8, 6) 

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
        
        scale = 30 # 해상도
        dst_pts = []
        for i in range(CHECKERBOARD[1]):
            for j in range(CHECKERBOARD[0]):
                dst_pts.append([j * scale, i * scale])
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # 3. 임시 호모그래피 계산
        H_init, _ = cv2.findHomography(corners, dst_pts)

        # 4. [핵심] 모든 정보를 담기 위한 자동 캔버스 계산
        # 원본 이미지의 네 귀퉁이 좌표
        img_corners = np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]], dtype=np.float32).reshape(-1, 1, 2)
        # 변환 후 각 귀퉁이가 갈 위치 계산
        transformed_corners = cv2.perspectiveTransform(img_corners, H_init)

        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        # 전체를 다 담기 위한 최소/최대 좌표
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # 마이너스 영역으로 나간 만큼 평행 이동시키는 행렬 생성
        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        # 최종 행렬 = 이동 행렬 x 초기 행렬
        H_final = translation_matrix @ H_init
        
        # 최종 도화지 크기 계산
        new_w = int(np.ceil(x_max - x_min))
        new_h = int(np.ceil(y_max - y_min))

        # 5. 결과 저장 (행렬과 크기)
        np.save(SAVE_H_PATH, H_final)
        np.save(SAVE_SIZE_PATH, np.array([new_w, new_h]))
        print(f"행렬 및 크기({new_w}x{new_h}) 저장 완료!")

        # 미리보기 (너무 크면 리사이즈해서 출력)
        preview = cv2.warpPerspective(img, H_final, (new_w, new_h))
        cv2.imshow('Full FOV BEV Preview', cv2.resize(preview, (800, int(800 * new_h / new_w))))
        cv2.waitKey(0)
    else:
        print("검출 실패! 코너 개수나 조명을 확인하세요.")

if __name__ == "__main__":
    main()
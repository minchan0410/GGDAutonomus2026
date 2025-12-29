import cv2
import numpy as np
import os

def run_bev_video(video_source, height_usage_ratio=0.6, bottom_shrink_ratio=0.73):
    """
    Args:
        video_source: 동영상 파일 경로 (문자열) 또는 웹캠 번호 (정수, 보통 0)
        height_usage_ratio: 이미지 바닥부터 위로 몇 %를 사용할지
        bottom_shrink_ratio: 밑변을 얼마나 좁힐지
    """
    # 1. 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[에러] 영상을 열 수 없습니다: {video_source}")
        return

    # 2. 비디오 속성 읽기 및 변환 행렬 사전 계산 (최적화)
    # 첫 프레임의 크기를 기준으로 변환 행렬을 미리 만듭니다.
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"--- 비디오 정보: 너비={w}, 높이={h} ---")

    # ROI 높이 및 시작 Y 좌표 계산
    roi_h = int(h * height_usage_ratio)
    start_y = h - roi_h

    # --- 좌표 설정 및 행렬 계산 (루프 밖에서 한 번만 수행) ---
    src_pts = np.float32([
        [0, 0],             # TL
        [w, 0],             # TR
        [w, roi_h],         # BR
        [0, roi_h]          # BL
    ])

    shrink_pixel = int(w * bottom_shrink_ratio / 2)
    
    dst_pts = np.float32([
        [0, 0],                         # TL (고정)
        [w, 0],                         # TR (고정)
        [w - shrink_pixel, roi_h],      # BR (안쪽 이동)
        [shrink_pixel, roi_h]           # BL (안쪽 이동)
    ])

    # 변환 행렬 M 구하기
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    print(f"[설정] 높이 사용 비율: {height_usage_ratio*100}%")
    print(f"[설정] 밑변 축소 비율: {bottom_shrink_ratio*100}%")
    print("--- 'q' 키를 누르면 종료됩니다 ---")

    # 3. 프레임 반복 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[알림] 영상이 끝났거나 프레임을 읽을 수 없습니다.")
            break

        # (1) ROI 자르기
        roi_img = frame[start_y:h, 0:w]

        # (2) BEV 변환 적용 (미리 계산된 M 사용)
        result_img = cv2.warpPerspective(
            roi_img, 
            M, 
            (w, roi_h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0,0,0)
        )

        # (3) 결과 출력
        # 원본 영상의 잘린 부분(ROI)과 결과를 위아래로 붙여서 보거나 따로 띄웁니다.
        cv2.imshow("Original ROI (Crops)", roi_img)
        cv2.imshow("BEV Result", result_img)

        # 'q' 키를 누르면 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# [사용자 설정 영역]
# ==========================================
if __name__ == "__main__":
    # 1. 비디오 파일 경로 지정
    # 예: video_path = "C:/Users/VIC_26/Desktop/driving_video.mp4"
    # 웹캠을 사용하려면 숫자 0을 입력하세요: video_path = 0
    
    video_path = r"C:\Users\VIC_26\Desktop\Gyeonggi_AutoDriving_SW_Competition\advanced_exercise\curv.mp4"

    # 파일이 실제로 존재하는지, 혹은 웹캠(숫자)인지 확인
    if isinstance(video_path, str) and not os.path.exists(video_path):
        print(f"[경고] 파일을 찾을 수 없습니다. 경로를 확인하세요: {video_path}")
    else:
        run_bev_video(
            video_source=video_path, 
            height_usage_ratio=0.4,    # 하단 60% 사용
            bottom_shrink_ratio=0.7,   # 밑변 73% 축소
        )
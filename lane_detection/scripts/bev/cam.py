import cv2
import os

def main():
    # 0번은 기본 웹캠입니다.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("--- 정밀 사진 촬영 모드 ---")
    print("1. 화면의 얇은 격자선을 참고하여 체커보드를 정렬하세요.")
    print("2. 's' 키를 누르면 선이 없는 깨끗한 원본 사진이 저장됩니다.")
    print("3. 'q'를 누르면 종료합니다.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, 'checkerboard.jpg')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [중요] 저장용 깨끗한 원본 복사
        clean_frame = frame.copy()

        # 출력용 이미지에 가이드라인 그리기
        display_frame = frame.copy()
        height, width = display_frame.shape[:2]
        
        # 선 설정: 색상(B,G,R), 두께(thickness)
        line_color = (0, 255, 0) # 초록색
        thickness = 1            # 가장 얇게

        # 1. 세로 중앙선
        center_x = width // 2
        cv2.line(display_frame, (center_x, 0), (center_x, height), line_color, thickness)

        # 2. 가로선 여러 개 추가 (4등분 지점: 25%, 50%, 75%)
        # 필요에 따라 범위를 조절하여 더 많이 그릴 수 있습니다.
        for i in range(1, 4):
            line_y = (height // 4) * i
            cv2.line(display_frame, (0, line_y), (width, line_y), line_color, thickness)

        # 3. 보조 세로선 (선택 사항: 좌우 대칭 확인용)
        for i in range(1, 4):
            line_x = (width // 4) * i
            if line_x == center_x: continue # 중앙선은 이미 그림
            cv2.line(display_frame, (line_x, 0), (line_x, height), (100, 150, 0), thickness)

        # 화면에는 선이 있는 이미지를 출력
        cv2.imshow('Calibration Guide (Grid)', display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # 's' 키를 누르면 '선이 없는' clean_frame을 저장
        if key == ord('s'):
            filename = 'checkerboard_on_ground.jpg'
            cv2.imwrite(save_path, clean_frame) 
            print(f"\n[성공] 사진이 '{save_path}'에 저장되었습니다.")
            break
        
        elif key == ord('q'):
            print("\n촬영을 취소합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
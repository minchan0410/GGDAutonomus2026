import cv2
import numpy as np
import json
import os

# --- 기존 BEVConverter 클래스 (변경 없음) ---
class BEVConverter:
    def __init__(self, config_file='bev_config.json'):
        self.config_file = config_file
        self.src_pts = []
        self.matrix = None
        self.width = 0
        self.height = 0

    def select_points(self, img):
        print("\n[설정 모드] 이미지의 4개 점을 순서대로 클릭하세요: 좌상 -> 우상 -> 우하 -> 좌하")
        self.src_pts = []
        temp_img = img.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.src_pts) < 4:
                    self.src_pts.append([x, y])
                    cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Calibration", temp_img)
                    print(f"포인트 추가: {x}, {y}")
                    
                    if len(self.src_pts) == 4:
                        print("4개 점 선택 완료! 설정 파일로 저장합니다.")
                        self.save_config()
                        # 창을 닫지 않고 대기 (사용자가 확인하도록)
                        print("아무 키나 누르면 비디오 재생을 시작합니다.")

        cv2.imshow("Calibration", temp_img)
        cv2.setMouseCallback("Calibration", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration")

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.src_pts, f)
        print(f"설정 저장 완료: {self.config_file}")

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.src_pts = json.load(f)
            return True
        return False

    def update_matrix(self, w, h):
        self.width, self.height = w, h
        src = np.float32(self.src_pts)
        
        # [영상용 팁] 영상 비율에 따라 margin을 조절하세요.
        margin_x = w * 0.25 
        dst = np.float32([
            [margin_x, 0], 
            [w - margin_x, 0], 
            [w - margin_x, h], 
            [margin_x, h]
        ])
        self.matrix = cv2.getPerspectiveTransform(src, dst)

    def warp(self, img):
        if self.matrix is None:
            self.update_matrix(img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.matrix, (self.width, self.height))

# --- 메인 실행부 (비디오 처리용으로 수정됨) ---
if __name__ == "__main__":
    # 1. 비디오 파일 경로 설정 (본인 경로로 수정)
    video_path = r'C:\Users\Minchan\Desktop\VSC\Gyeonggi_AutoDriving_SW_Competition\advanced_exercise\curv.mp4'
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {video_path}")
        exit()

    # 2. 첫 번째 프레임 읽기 (캘리브레이션 용도)
    ret, first_frame = cap.read()
    if not ret:
        print("비디오 프레임을 읽을 수 없습니다.")
        exit()

    # 3. 변환기 초기화 및 설정 확인
    converter = BEVConverter()

    if not converter.load_config():
        print("⚠️ 설정 파일이 없습니다. 첫 프레임으로 캘리브레이션을 시작합니다.")
        converter.select_points(first_frame)
    else:
        print("✅ 기존 설정을 불러왔습니다. 비디오 재생을 시작합니다.")

    # 4. 비디오 재생 루프
    while True:
        # 비디오는 연속적이므로 캘리브레이션 후 계속 읽어야 함
        # 첫 프레임은 이미 읽었지만 쿨하게 무시하고 다음 프레임부터 처리하거나
        # 다시 처음부터 보고 싶으면 cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 사용
        
        ret, frame = cap.read()
        
        if not ret: # 비디오가 끝나면 종료 (또는 반복하려면 break 대신 cap.set... 사용)
            print("비디오 종료")
            break

        # BEV 변환 수행
        bev_img = converter.warp(frame)

        # (선택사항) 원본에 영역 표시해서 같이 보기
        pts = np.array(converter.src_pts, np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        
        # 결과 화면 크기 조절 (너무 크면 보기 힘드니까)
        display_frame = cv2.resize(frame, (640, 360)) 
        display_bev = cv2.resize(bev_img, (640, 360))

        cv2.imshow("Original Video", display_frame)
        cv2.imshow("BEV Video", display_bev)

        # 키 입력 대기 (33ms = 약 30fps)
        key = cv2.waitKey(33) 

        if key == ord('q'): # q 누르면 종료
            break
        elif key == ord('r'): # r 누르면 설정 초기화 후 종료
            if os.path.exists('bev_config.json'):
                os.remove('bev_config.json')
                print("설정 삭제됨. 프로그램을 재실행하여 다시 설정하세요.")
            break

    cap.release()
    cv2.destroyAllWindows()
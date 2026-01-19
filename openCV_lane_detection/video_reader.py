# io/video_reader.py
import cv2

class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"비디오를 열 수 없음: {path}")

    def read(self):
        return self.cap.read()

    def fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30

    def release(self):
        self.cap.release()

# vision/preprocess.py
import cv2
import numpy as np

class Preprocess:
    def __init__(self, kernel_size=5, low=50, high=200):
        self.kernel_size = kernel_size
        self.low = low
        self.high = high

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, img):
        k = self.kernel_size
        return cv2.GaussianBlur(img, (k, k), 1.5)

    def canny(self, img):
        return cv2.Canny(img, self.low, self.high)








class BEVTransformer:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_src_dst(self, w, h):
        c = self.cfg
        src = np.float32([
            [c.src_x_left_top_ratio*w,    c.src_y_top_ratio*h],
            [c.src_x_right_top_ratio*w,   c.src_y_top_ratio*h],
            [c.src_x_right_bottom_ratio*w,c.src_y_bottom_ratio*h],
            [c.src_x_left_bottom_ratio*w, c.src_y_bottom_ratio*h]
        ])
        dst = np.float32([
            [0,   0],
            [w-1, 0],
            [w-1, h-1],
            [0,   h-1]
        ])
        return src, dst

    def warp(self, img, src, dst, size):
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
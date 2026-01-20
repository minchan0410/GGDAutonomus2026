# config.py
import numpy as np

class Config:
    video_path = "curv.mp4"

    # 원본 이미지에서 BEV 영역 설정 
    src_y_top_ratio = 0.7
    src_y_bottom_ratio = 0.9
    src_x_left_top_ratio = 0.2
    src_x_right_top_ratio = 0.8
    src_x_left_bottom_ratio = 0.05
    src_x_right_bottom_ratio = 0.95

    # Preprocess
    kernel_size = 5
    low_threshold = 50
    high_threshold = 200

    # Hough
    hough_rho = 1
    hough_theta = np.pi/180
    hough_threshold = 50
    hough_min_line_len = 60
    hough_max_line_gap = 50

    # Seed 기반 피팅
    seed_r = 25
    sigma = 0.3
    power = 3.0

    # 필터링
    min_abs_slope_deg = 80
    slope_tol_deg = 5

    # Centerline
    lane_w_px = 670
    yA_offset = 50
    yB_offset = 30
    yC_offset = 70

    y_ref_offset = 40
    miss_decay_after = 15
    decay = 0.9

    # scanline
    scan_y1_offset = 10
    scan_y2_offset = 30
    lane_band_left = 700
    lane_band_right = 640

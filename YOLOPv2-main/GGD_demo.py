import argparse
import time
import csv
from pathlib import Path
import cv2
import torch
import numpy as np
import math

save_mp4_toggle = False
save_csv_toggle = False

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized, select_device, increment_path,\
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,\
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,\
    AverageMeter, \
    LoadImages

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser

def calculate_midpoint_score(x1, x2, w):
    hw = w/2
    midpoint = int((x1 + x2) / 2)
    return (midpoint - hw) / hw * 100



def calculate_line_score(x1, y1, x2, y2):
    # 1. 분모가 0인 경우 (수직선, 기울기 무한대) -> 점수 0
    if x2 - x1 == 0:
        return 0.0

    # 2. 기울기(m) 계산 (OpenCV 이미지 좌표계: y가 아래로 증가함에 유의)
    # m > 0: 왼쪽 상단 -> 오른쪽 하단 (↘ 방향)
    # m < 0: 왼쪽 하단 -> 오른쪽 상단 (↗ 방향)
    slope = (y2 - y1) / (x2 - x1)
    
    # 3. 기울기 절댓값 확인 (0.3 미만은 필터링된다고 가정했으나, 안전장치로 0 반환)
    abs_slope = abs(slope)
    if abs_slope < 0.3:
        return 0.0 # 혹은 예외 처리

    # 4. 각도(Theta) 계산 (Degree)
    # theta는 0 ~ 90도 사이의 값을 가짐
    theta_deg = math.degrees(math.atan(abs_slope))
    
    # 5. 스코어링 기준 각도 설정
    # 최소 각도: 기울기 0.3일 때의 각도 (약 16.7도)
    # 최대 각도: 90도
    min_theta = math.degrees(math.atan(0.3))
    max_theta = 90.0
    
    # 6. 각도에 따른 선형 보간 (Linear Interpolation)
    # theta가 min_theta일 때 -> score 100
    # theta가 max_theta일 때 -> score 0
    # 공식: 100 * (max_theta - 현재theta) / (max_theta - min_theta)
    
    # 입력 각도가 90도를 아주 미세하게 넘거나 min보다 작을 경우를 대비한 클리핑
    if theta_deg >= max_theta:
        score_magnitude = 0.0
    elif theta_deg <= min_theta:
        score_magnitude = 100.0
    else:
        score_magnitude = 100 * (max_theta - theta_deg) / (max_theta - min_theta)

    # 7. 부호 적용 (양의 기울기는 양수, 음의 기울기는 음수)
    if slope > 0:
        return score_magnitude
    else:
        return -score_magnitude

def average_lines(lines):
    if not lines:
        return
    lines_array = np.array(lines)
    # 3. 평균 구하기 (axis=0은 세로로 같은 인덱스끼리 평균을 냄)
    # 결과는 [avg_x1, avg_y1, avg_x2, avg_y2] 형태가 됨
    averaged_line = np.mean(lines_array, axis=0)
    averaged_line = averaged_line.astype(int)
    return averaged_line

def detect():
    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model  = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    t0 = time.time()

    # Save mp4 ---------------------------------------------------------------
    if save_mp4_toggle is True:
        output_writer = None
    # Save csv
    if save_csv_toggle is True:
        csv_file = open(str(save_dir / 'result_data.csv'), 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['Left_Mid', 'Right_Mid', 'Final_Midpoint'])
    # ------------------------------------------------------------------------

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll) # line mask
        mask_vis = (ll_seg_mask * 255).astype(np.uint8) # mask -> vis image
        h, w = im0s.shape[:2]
        mask_resized = cv2.resize(mask_vis, (w, h), interpolation=cv2.INTER_NEAREST)

        # erode (seg 된 결과는 너무 두꺼움) -----------------------------------
        erode_kernel = np.ones((3, 3), np.uint8) 
        mask_eroded = cv2.erode(mask_resized, erode_kernel, iterations=5)
        # ------------------------------------------------------------------

        # Make ROI & apply -------------------------------------------------
        roi_bottom_width = 1.0   # 바닥 너비
        roi_top_width = 0.9      # 윗변 너비
        roi_height = 0.45        # 높이

        cx = w // 2
        y_top = int(h * (1 - roi_height))
        y_mid = int(h * (1 - roi_height / 2))
        x_bottom_half = int(w * roi_bottom_width / 2)
        x_top_half = int(w * roi_top_width / 2)

        roi_verts = np.array([[
            (cx - x_bottom_half, h),
            (cx - x_top_half, y_top),
            (cx + x_top_half, y_top),
            (cx + x_bottom_half, h)
        ]], dtype=np.int32)

        roi_mask_img = np.zeros_like(mask_eroded)
        cv2.fillPoly(roi_mask_img, roi_verts, 255)
        mask_roi_applied = cv2.bitwise_and(mask_eroded, roi_mask_img)

        # Apply Hough Transform ------------------------------------------------------------------ 
        lines = cv2.HoughLinesP(
            mask_roi_applied,          
            rho=5, #거리 해상도. 직선을 탐지할 때 얼마나 촘촘한 간격 
            theta=np.pi/180, # 각도 해상도. 얼마나 촘촘한 각도
            threshold=50, # 몇개의 점이 일진선으로 되어야 직선?
            minLineLength=100, #이거보다 짧은 직선은 버리기
            maxLineGap=50# 직선이 중간에 끊겨도 하나의 선으로 잇기 
        )

        mask_bgr = cv2.cvtColor(mask_roi_applied, cv2.COLOR_GRAY2BGR)
        cv2.polylines(mask_bgr, roi_verts, isClosed=True, color=(0, 255, 0), thickness=2)
        #------------------------------------------------------------------------------------------
        
        # filtering Algorithm ------------------------------------------------
        filtering_slope = 0.3 # 이 기울기 이하의 직선은 가로선으로 가정하고 버림

        right_lines = []
        left_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 기울기 필터링
                dx = x2 - x1
                dy = y2 - y1
            
                if dx == 0: 
                    continue # 수평선 제외

                slope = dy / dx
                
                if abs(slope) <= filtering_slope: # 임곗값 이하 기울기 제외
                    continue

                # Scoreing Method --------------------------
                # Scoring range : -100 ~ 100 
                ls = calculate_line_score(x1, y1, x2, y2)
                lm = calculate_midpoint_score(x1, y1, w)

                if ls + lm <= 0:
                    left_lines.append([x1, y1, x2, y2])
                else:
                    right_lines.append([x1, y1, x2, y2])
                # ------------------------------------------
        
        left_result = average_lines(left_lines)
        right_result = average_lines(right_lines)

        left_mid_point = 0 # initialize
        right_mid_point = w

        # Final lane, Point Visualization and Compute  -----------------------------
        if left_result is not None:
            lx1, ly1, lx2, ly2 = left_result
            left_mid_point = int((lx1 + lx2) / 2)
            cv2.circle(mask_bgr, (left_mid_point, y_mid), 20, (0, 0, 255), -1)
        
        if right_result is not None:
            lx1, ly1, lx2, ly2 = right_result
            right_mid_point = int((lx1 + lx2) / 2)
            cv2.circle(mask_bgr, (right_mid_point, y_mid), 20, (255, 0, 0), -1)
        
        final_midpoint = int((left_mid_point + right_mid_point) / 2)
        cv2.circle(mask_bgr, (final_midpoint, y_mid), 20, (255, 255, 0), -1)
        
        for line in right_lines:
            x1, y1, x2 ,y2 = line
            color = (255, 0, 0)
            cv2.line(mask_bgr, (x1, y1), (x2, y2), color, 1)

        for line in left_lines:
            x1, y1, x2 ,y2 = line
            color = (0, 0, 255)
            cv2.line(mask_bgr, (x1, y1), (x2, y2), color, 1)
        # --------------------------------------------------------------------------


        # For Cv2 imshow
        combined = np.hstack((im0s, mask_bgr))
        target_width = 1200
        scale = target_width / combined.shape[1]
        new_w = int(combined.shape[1] * scale)
        new_h = int(combined.shape[0] * scale)
        combined_small = cv2.resize(combined, (new_w, new_h))
        cv2.imshow('Left: Original | Right: Erode+ROI+Hough', combined_small)
        
        # Save Data --------------------------------------------------------------------------------------------
        if save_mp4_toggle is True:
            if output_writer is None: # 처음 한 번만 실행됨
                save_path = str(save_dir / 'result_video.mp4')
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                output_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_w, new_h))
            output_writer.write(combined_small) # 프레임 저장
        if save_csv_toggle is True:
            writer.writerow([left_mid_point, right_mid_point, final_midpoint])
        # -------------------------------------------------------------------------------------------------------
        if cv2.waitKey(1) == ord('q'):
            break

    if save_mp4_toggle is True:
        if output_writer:
            output_writer.release()
            print("\n **Video Saved Complete** \n")

if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
            detect()
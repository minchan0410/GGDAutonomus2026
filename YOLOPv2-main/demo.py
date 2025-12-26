import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# 유틸리티 임포트
from utils.utils import \
    time_synchronized, select_device, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, LoadImages, LoadStreams

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source') # 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 아래 저장 관련 인자들은 이제 무시되거나 실시간용으로만 쓰입니다.
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    return parser

def detect():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    
    # 저장 기능을 완전히 끔 (폴더 생성 방지)
    save_img = False
    save_txt = False

    inf_time = AverageMeter()
    nms_time = AverageMeter()

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = model.to(device)

    cudnn.benchmark = True
    if half:
        model.half()
    model.eval()

    # 데이터로더 설정
    if source == '0':
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # 후처리 (Demo용 호환성 해결)
        pred = split_for_trace_model(pred, anchor_grid)

        # NMS 적용
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # 결과 처리
        for i, det in enumerate(pred):
            # im0s가 리스트일 수 있으므로(Stream), 안전하게 복사
            im0 = im0s[i].copy() if isinstance(im0s, list) else im0s.copy()
            
            # 박스 그리기
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, im0, line_thickness=3)

            # 세그멘테이션 마스크 크기 조정 및 오버레이
            h, w = im0.shape[:2]
            da_mask_resized = cv2.resize(da_seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            ll_mask_resized = cv2.resize(ll_seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            show_seg_result(im0, (da_mask_resized, ll_mask_resized), is_demo=True)

            # 실시간 화면 출력
            cv2.imshow('YOLOPv2 Real-time (Press Q to quit)', im0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        print(f'Inference: ({t2 - t1:.3f}s) | NMS: ({t4 - t3:.3f}s)')

    print(f'Total Done. ({time.time() - t0:.3f}s)')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = make_parser().parse_args()
    with torch.no_grad():
        detect()
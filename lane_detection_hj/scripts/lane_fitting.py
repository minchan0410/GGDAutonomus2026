# lane/seed_points.py
import numpy as np
import cv2
from collections import deque

class SeedPointFinder:
    def __init__(self, cfg):
        self.cfg = cfg

    def _scan_row(self, edge_img, y):
        line_vals = edge_img[y, :].astype(np.float32)
        xs = np.where(line_vals > 0)[0]
        return xs

    def find(self, bev_edges):
        h, w = bev_edges.shape[:2]
        c = self.cfg
        edges_vis = cv2.cvtColor(bev_edges, cv2.COLOR_GRAY2BGR)

        # scan y1
        scan_y = h - c.scan_y1_offset
        edge_xs = self._scan_row(bev_edges, scan_y)
        x_max = x2 = None

        if edge_xs.size > 0:
            x_max = int(edge_xs[-1])
            cv2.circle(edges_vis, (x_max, scan_y), 10, (0,255,0), -1)

            xL = x_max - c.lane_band_left
            xR = x_max - c.lane_band_right
            if xR >= 0 and xL < w:
                xL = max(0, xL)
                xR = min(w-1, xR)
                cand = edge_xs[(edge_xs >= xL) & (edge_xs <= xR)]
                if cand.size > 0:
                    x2 = int(cand[-1])
                    cv2.circle(edges_vis, (x2, scan_y), 10, (0,255,0), -1)

        # scan y2
        scan_y2 = h - c.scan_y2_offset
        edge_xs2 = self._scan_row(bev_edges, scan_y2)
        x_max2 = x2_2 = None

        if edge_xs2.size > 0:
            x_max2 = int(edge_xs2[-1])
            cv2.circle(edges_vis, (x_max2, scan_y2), 10, (0,255,255), -1)

            xL2 = x_max2 - c.lane_band_left
            xR2 = x_max2 - c.lane_band_right
            if xR2 >= 0 and xL2 < w:
                xL2 = max(0, xL2)
                xR2 = min(w-1, xR2)
                cand2 = edge_xs2[(edge_xs2 >= xL2) & (edge_xs2 <= xR2)]
                if cand2.size > 0:
                    x2_2 = int(cand2[-1])
                    cv2.circle(edges_vis, (x2_2, scan_y2), 10, (0,255,255), -1)

        have_right = (x_max is not None) and (x_max2 is not None)
        have_left  = (x2 is not None) and (x2_2 is not None)

        return {
            "edges_vis": edges_vis,
            "scan_y": scan_y,
            "scan_y2": scan_y2,
            "x_max": x_max, "x_max2": x_max2,
            "x2": x2, "x2_2": x2_2,
            "have_right": have_right,
            "have_left": have_left
        }






class HoughLaneFitter:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def lane_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        ang = np.degrees(np.arctan2(dx, -dy))
        if ang > 180: ang -= 360
        if ang <= -180: ang += 360
        return ang

    @staticmethod
    def seg_slope_and_len(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        length = (dx*dx + dy*dy) ** 0.5
        slope = np.degrees(np.arctan2(dx, -dy))
        return slope, length

    @staticmethod
    def weight_by_slope(m, m_target, sigma=0.5):
        if m is None:
            return 0.0
        diff = m - m_target
        return float(np.exp(-(diff*diff) / (2.0*sigma*sigma)))

    @staticmethod
    def is_seed_endpoint(x1, y1, x2, y2, sx, sy, r=25):
        r2 = r*r
        d1 = (x1 - sx)**2 + (y1 - sy)**2
        d2 = (x2 - sx)**2 + (y2 - sy)**2
        if d1 <= r2:
            return True, (x1, y1, x2, y2)
        if d2 <= r2:
            return True, (x2, y2, x1, y1)
        return False, (x1, y1, x2, y2)

    def detect_lines(self, bev_edges):
        c = self.cfg
        return cv2.HoughLinesP(
            bev_edges,
            rho=c.hough_rho,
            theta=c.hough_theta,
            threshold=c.hough_threshold,
            minLineLength=c.hough_min_line_len,
            maxLineGap=c.hough_max_line_gap
        )

    def fit(self, bev_edges, seeds):
        c = self.cfg
        lines = self.detect_lines(bev_edges)

        have_right = seeds["have_right"]
        have_left  = seeds["have_left"]
        x_max, x_max2 = seeds["x_max"], seeds["x_max2"]
        x2, x2_2 = seeds["x2"], seeds["x2_2"]
        scan_y, scan_y2 = seeds["scan_y"], seeds["scan_y2"]

        m_right = self.lane_angle(x_max, scan_y, x_max2, scan_y2) if have_right else None
        m_left  = self.lane_angle(x2, scan_y, x2_2, scan_y2)       if have_left  else None

        best_right = None
        best_left  = None

        if lines is not None:
            for l in lines:
                x1, y1, x2l, y2l = l[0]

                if have_right and (m_right is not None):
                    ok, (sx1, sy1, sx2, sy2) = self.is_seed_endpoint(
                        x1, y1, x2l, y2l, x_max2, scan_y2, r=c.seed_r
                    )
                    if ok:
                        m, length = self.seg_slope_and_len(sx1, sy1, sx2, sy2)
                        wgt = self.weight_by_slope(m, m_right, sigma=c.sigma)
                        score = (wgt ** c.power) * length
                        if (best_right is None) or (score > best_right[0]):
                            best_right = (score, (sx1, sy1, sx2, sy2), m, length)

                if have_left and (m_left is not None):
                    ok, (sx1, sy1, sx2, sy2) = self.is_seed_endpoint(
                        x1, y1, x2l, y2l, x2_2, scan_y2, r=c.seed_r
                    )
                    if ok:
                        m, length = self.seg_slope_and_len(sx1, sy1, sx2, sy2)
                        wgt = self.weight_by_slope(m, m_left, sigma=c.sigma)
                        score = (wgt ** c.power) * length
                        if (best_left is None) or (score > best_left[0]):
                            best_left = (score, (sx1, sy1, sx2, sy2), m, length)

        best_right = self._filter_best(best_right, m_right)
        best_left  = self._filter_best(best_left,  m_left)

        return {
            "best_right": best_right,
            "best_left": best_left,
            "m_right": m_right,
            "m_left": m_left
        }

    def _filter_best(self, best, m_target):
        if best is None:
            return None
        c = self.cfg
        score, (x1,y1,x2,y2), m, length = best

        if (m is None) or (abs(m) > c.min_abs_slope_deg):
            return None
        if (m_target is not None) and (abs(m - m_target) > c.slope_tol_deg):
            return None
        return best
    
class CenterlineEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_offset = None
        self.prev_center_pts = None
        self.prev_err_px = 0
        self.prev_m_center = 0.0
        self.miss_cnt = 0
        self.offset_hist = deque(maxlen=5)

    @staticmethod
    def x_at_y_from_line(x1, y1, x2, y2, yq):
        if y2 == y1:
            return None
        t = (yq - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    @staticmethod
    def lane_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        ang = np.degrees(np.arctan2(dx, -dy))
        if ang > 180: ang -= 360
        if ang <= -180: ang += 360
        return ang

    def _push_offset(self, offset: float):
        """확실한 판단이든 애매존 판단이든 최종 선택된 offset을 히스토리에 저장"""
        self.prev_offset = offset
        self.offset_hist.append(offset)
    
    def _majority_offset(self, half_w: float):
        """
        최근 5개 offset의 다수결로 +half_w / -half_w 중 하나 선택.
        동률이면 prev_offset, 그것도 없으면 +half_w로 fallback.
        """
        if len(self.offset_hist) == 0:
            return self.prev_offset if self.prev_offset is not None else +half_w

        pos = sum(1 for o in self.offset_hist if o > 0)  # +half_w 쪽
        neg = sum(1 for o in self.offset_hist if o < 0)  # -half_w 쪽

        if pos > neg:
            return +half_w
        if neg > pos:
            return -half_w

        # 동률이면 직전 offset을 우선(관성)
        if self.prev_offset is not None:
            return +half_w if self.prev_offset > 0 else -half_w

        return +half_w
    
    def update(self, w, h, best_right, best_left):
        c = self.cfg

        LANE_W = c.lane_w_px
        HALF_W = LANE_W / 2.0

        yA = h - c.yA_offset
        yB = h - c.yB_offset
        yC = h - c.yC_offset

        xrA = xrB = xrC = None
        xlA = xlB = xlC = None

        if best_right is not None:
            _, (rx1, ry1, rx2, ry2), _, _ = best_right
            xrA = self.x_at_y_from_line(rx1, ry1, rx2, ry2, yA)
            xrB = self.x_at_y_from_line(rx1, ry1, rx2, ry2, yB)
            xrC = self.x_at_y_from_line(rx1, ry1, rx2, ry2, yC)

        if best_left is not None:
            _, (lx1, ly1, lx2, ly2), _, _ = best_left
            xlA = self.x_at_y_from_line(lx1, ly1, lx2, ly2, yA)
            xlB = self.x_at_y_from_line(lx1, ly1, lx2, ly2, yB)
            xlC = self.x_at_y_from_line(lx1, ly1, lx2, ly2, yC)

        center_p1 = center_p2 = center_p3 = None

        if (xlA is not None) and (xrA is not None) and (xlB is not None) and (xrB is not None):
            center_p1 = (int((xlA + xrA) / 2.0), int(yA))
            center_p2 = (int((xlB + xrB) / 2.0), int(yB))
            center_p3 = (int((xlC + xrC) / 2.0), int(yC))

        else:
            xs = [float(v) for v in [xlA, xrA, xlB, xrB, xlC, xrC] if v is not None]
            if len(xs) == 0:
                center_p1 = center_p2 = center_p3 = None

            else:
                x_mean = float(np.mean(xs))
                left_th = 0.3 * w
                right_th = 0.7 * w

                # ========= helper: offset 히스토리에 저장 =========
                def push_offset(ofs: float):
                    self.prev_offset = ofs
                    # offset_hist가 __init__에 없을 수도 있으니 안전하게
                    if hasattr(self, "offset_hist") and (self.offset_hist is not None):
                        self.offset_hist.append(ofs)

                # ========= helper: 최근 5개 다수결 =========
                def majority_offset():
                    # 히스토리가 없거나 비었으면 기존 로직 fallback
                    if (not hasattr(self, "offset_hist")) or (self.offset_hist is None) or (len(self.offset_hist) == 0):
                        if self.prev_offset is None:
                            return +HALF_W if x_mean < (w/2.0) else -HALF_W
                        return self.prev_offset

                    pos = sum(1 for o in self.offset_hist if o > 0)
                    neg = sum(1 for o in self.offset_hist if o < 0)

                    if pos > neg:
                        return +HALF_W
                    if neg > pos:
                        return -HALF_W

                    # 동률이면 직전 offset 우선(관성)
                    if self.prev_offset is not None:
                        return +HALF_W if self.prev_offset > 0 else -HALF_W

                    return +HALF_W if x_mean < (w/2.0) else -HALF_W

                # ========= 여기부터 기존 분기 유지 + 애매존만 변경 =========
                if x_mean < left_th:
                    offset = +HALF_W
                    push_offset(offset)
                    print("확실한 왼쪽 차선")

                elif x_mean > right_th:
                    offset = -HALF_W
                    push_offset(offset)
                    print("확실한 오른쪽 차선")

                else:
                    # ✅ 애매존: 바로 전 1개가 아니라 최근 5개 다수결로 결정
                    offset = majority_offset()
                    push_offset(offset)

                    if offset > 0:
                        print("애매존!!!! 최근 5개 다수결 -> 왼쪽 차선 유지")
                    else:
                        print("애매존!!!! 최근 5개 다수결 -> 오른쪽 차선 유지")

                def avg2(a, b):
                    vals = []
                    if a is not None: vals.append(float(a))
                    if b is not None: vals.append(float(b))
                    return float(np.mean(vals)) if vals else None

                xA = avg2(xlA, xrA)
                xB = avg2(xlB, xrB)
                xC = avg2(xlC, xrC)

                center_p1 = (int(xA + offset), int(yA)) if xA is not None else None
                center_p2 = (int(xB + offset), int(yB)) if xB is not None else None
                center_p3 = (int(xC + offset), int(yC)) if xC is not None else None

        new_ok = (center_p1 is not None) and (center_p2 is not None) and (center_p3 is not None)

        if new_ok:
            self.miss_cnt = 0
            self.prev_center_pts = [center_p1, center_p2, center_p3]

            cx = w // 2
            y_ref = h - c.y_ref_offset
            p_ref = min(self.prev_center_pts, key=lambda p: abs(p[1] - y_ref))
            self.prev_err_px = int(p_ref[0] - cx)

            x1c, y1c = self.prev_center_pts[0]
            x2c, y2c = self.prev_center_pts[1]
            m_tmp = -self.lane_angle(x1c, -y1c, x2c, -y2c)
            if m_tmp is not None:
                self.prev_m_center = float(m_tmp)

        else:
            self.miss_cnt += 1
            if self.miss_cnt > c.miss_decay_after:
                self.prev_err_px = int(self.prev_err_px * c.decay)
                self.prev_m_center = float(self.prev_m_center * c.decay)

        return {
            "center_pts": self.prev_center_pts,
            "err_px": self.prev_err_px,
            "m_center": self.prev_m_center,
            "miss_cnt": self.miss_cnt
        }

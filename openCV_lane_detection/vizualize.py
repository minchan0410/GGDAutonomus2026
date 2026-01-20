# viz/overlay.py
import cv2
import matplotlib.pyplot as plt

class OverlayDrawer:
    def draw_center_error_overlay(self, img, w, h, center_pts, y_ref=None):
        if (center_pts is None) or (len(center_pts) == 0):
            return

        cx = w // 2
        cv2.line(img, (cx, h), (cx, h-120), (0, 0, 255), 2)
        cv2.putText(img, "Vehicle Center", (cx-60, h-130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        if y_ref is None:
            p_ref = max(center_pts, key=lambda p: p[1])
        else:
            p_ref = min(center_pts, key=lambda p: abs(p[1] - y_ref))

        x_ref, y_ref2 = int(p_ref[0]), int(p_ref[1])
        err_px = -x_ref + cx

        cv2.line(img, (cx, y_ref2), (x_ref, y_ref2), (0, 255, 255), 4)
        cv2.circle(img, (x_ref, y_ref2), 7, (0, 255, 255), -1)
        cv2.circle(img, (cx, y_ref2), 6, (0, 0, 255), -1)

        side = "RIGHT" if err_px > 0 else "LEFT" if err_px < 0 else "CENTER"
        cv2.putText(img, f"Center Error: {err_px:+d}px ({side})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(img, f"y={y_ref2}",
                    (x_ref+10, y_ref2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        return err_px
    
    def draw_centerline(self, img, center_pts):
        if center_pts is None:
            return
        p1, p2, p3 = center_pts
        cv2.line(img, p1, p2, (255, 255, 0), 12)
        cv2.line(img, p2, p3, (255, 255, 0), 12)
        for p in [p1,p2,p3]:
            cv2.circle(img, p, 4, (255, 255, 0), -1)

    def draw_best_lines(self, img, best_right, best_left):
        if best_right is not None:
            _, (x1,y1,x2,y2), _, _ = best_right
            cv2.line(img, (x1,y1), (x2,y2), (0,255,255), 6)
        if best_left is not None:
            _, (x1,y1,x2,y2), _, _ = best_left
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 6)

class Plotter:
    def plot(self, t_hist, err_hist, m_hist):
        plt.figure()
        plt.plot(t_hist, err_hist)
        plt.xlabel("time (s)")
        plt.ylabel("lateral error (px)")
        plt.title("lateral error")
        plt.grid(True)

        plt.figure()
        plt.plot(t_hist, m_hist)
        plt.xlabel("time (s)")
        plt.ylabel("heading error")
        plt.title("Heading Error")
        plt.grid(True)

        plt.show()
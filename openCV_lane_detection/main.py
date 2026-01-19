# main.py
import time
import cv2
from config import Config
from video_reader import VideoReader
from preprocess import Preprocess
from preprocess import BEVTransformer
from lane_fitting import SeedPointFinder
from lane_fitting import HoughLaneFitter
from lane_fitting import CenterlineEstimator
from vizualize import OverlayDrawer
from vizualize import Plotter
import rospy
from std_msgs.msg import Float32


def main():
    cfg = Config()

    reader = VideoReader(cfg.video_path)
    pre = Preprocess(cfg.kernel_size, cfg.low_threshold, cfg.high_threshold)
    bev = BEVTransformer(cfg)
    seeds_finder = SeedPointFinder(cfg)
    fitter = HoughLaneFitter(cfg)
    center_est = CenterlineEstimator(cfg)
    drawer = OverlayDrawer()
    plotter = Plotter()

    err_hist, m_hist, t_hist = [], [], []
    lateral_error = 0
    t0 = time.time()

    rospy.init_node("lane_pipeline_node", anonymous=True)
    pub_yaw = rospy.Publisher("/heading_error", Float32, queue_size=1)
    pub_lateral_err = rospy.Publisher("/lateral_error", Float32, queue_size=1)

    while True:
        if rospy.is_shutdown():
            break
        ret, frame = reader.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        src, dst = bev.build_src_dst(w, h)

        gray = pre.grayscale(frame)
        bev_gray = bev.warp(gray, src, dst, (w, h))

        blur = pre.gaussian_blur(bev_gray)
        bev_edges = pre.canny(blur)

        seeds = seeds_finder.find(bev_edges)
        edges_vis = seeds["edges_vis"]

        fit_res = fitter.fit(bev_edges, seeds)
        best_right = fit_res["best_right"]
        best_left  = fit_res["best_left"]

        ctrl = center_est.update(w, h, best_right, best_left)
        pub_yaw.publish(Float32(ctrl["m_center"]))

        # log
        err_hist.append(ctrl["err_px"])
        m_hist.append(ctrl["m_center"])
        t_hist.append(time.time() - t0)

        # viz
        if ctrl["center_pts"] is not None:
            drawer.draw_centerline(edges_vis, ctrl["center_pts"])
            lateral_error = drawer.draw_center_error_overlay(edges_vis, w, h, ctrl["center_pts"], y_ref=h-cfg.y_ref_offset)
        
        pub_lateral_err.publish(lateral_error)
        drawer.draw_best_lines(edges_vis, best_right, best_left)

        cv2.putText(edges_vis, f"m_center: {ctrl['m_center']:+.4f}  miss:{ctrl['miss_cnt']}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("edges_with_bottom_blue_line", edges_vis)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    reader.release()
    cv2.destroyAllWindows()

    # plotter.plot(t_hist, err_hist, m_hist)

if __name__ == "__main__":
    main()

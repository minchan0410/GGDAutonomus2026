#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading
from std_msgs.msg import Int16, Bool, String, Int32MultiArray, Float32
from geometry_msgs.msg import PointStamped, PoseArray
from collections import deque
import math

ROI_RADIUS = 0.5

SPEED_0    = 0
SPEED_MID  = 150
SPEED_HIGH = 255

LC_STEER      = 22.5
STEER_TIME1    = 1
STEER_TIME2    = 1
STEER_TIME3    = 1

STRAIGHT_TIME1 = 1
STRAIGHT_TIME2 = 1

DEFAULT_LANE_LINES_TOPIC = "/lane_lines_px"
DEFAULT_INVALID_VALUE = float("nan")
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_ROI_HEIGHT = 0.45
DEFAULT_CLAMP_TO_ROI = True


def _finite(val):
    return val is not None and not math.isnan(val) and not math.isinf(val)


# EMA-based helper removed — using raw ddx + rolling-distance gating now


class FinalPlanner:
    def __init__(self):
        rospy.init_node("final_planner", anonymous=False)

        self.rate_hz = rospy.get_param("~rate_hz", 20)
        self.default_motor = rospy.get_param("~default_motor", 0)

        # ---- params ----
        self.lane_lines_topic = rospy.get_param("~lane_lines_topic", DEFAULT_LANE_LINES_TOPIC)
        self.roi_offset_x = rospy.get_param("planner_common/roi/offset_x", 0.74)

        # ROI radius (meters)
        self.roi_radius = rospy.get_param("planner_common/roi/radius", ROI_RADIUS)
        # ROI sector parameters (degrees): min and max angle (inclusive), 0 = forward.
        # Defaults: -90..+90 (front 180°)
        self.roi_angle_min_deg = rospy.get_param("planner_common/roi/angle_min_deg", -90.0)
        self.roi_angle_max_deg = rospy.get_param("planner_common/roi/angle_max_deg", 90.0)

        self.SPEED_0 = rospy.get_param("~speed_0", SPEED_0)
        self.SPEED_MID = rospy.get_param("~speed_mid", SPEED_MID)
        self.SPEED_HIGH = rospy.get_param("~speed_high", SPEED_HIGH)

        self.LC_STEER = rospy.get_param("~lc_steer", LC_STEER)

        # delay after left lane change complete before transitioning to lane_driving2 (sec)
        self.left_lc_complete_delay = rospy.get_param("~left_lc_complete_delay_sec", 0.0)
        self.right_lc_complete_delay = rospy.get_param("~right_lc_complete_delay_sec", 0.0)

        # timeout in lane_driving1 before automatically entering lane_change_to_left (sec)
        self.lane_driving1_timeout = rospy.get_param("~lane_driving1_timeout_sec", 5.0)

        self.queues_maxlen = rospy.get_param("~queues_maxlen", 10)
        self.yolo_count_threshold = rospy.get_param("~yolo_count_threshold", 7)

        self.traffic_green_threshold = rospy.get_param("~traffic_green_threshold", 5)
        self.traffic_green_timeout = rospy.get_param("~traffic_green_timeout", 20.0)
        # startup guard: block state changes for a few seconds
        self.state_change_delay_sec = rospy.get_param("~state_change_delay_sec", 0.0)
        # start state ramp duration (sec)
        self.start_ramp_sec = rospy.get_param("~start_ramp_sec", 3.0)
        # serial readiness gate
        self.serial_timeout_sec = rospy.get_param("~serial_timeout_sec", 0.5)
        # ---- lane change params (run2.py aligned) ----
        self.invalid_value = float(rospy.get_param("~invalid_value", DEFAULT_INVALID_VALUE))
        self.image_width = int(rospy.get_param("~image_width", DEFAULT_IMAGE_WIDTH))
        self.image_height = int(rospy.get_param("~image_height", DEFAULT_IMAGE_HEIGHT))
        self.roi_height = float(rospy.get_param("~roi_height", DEFAULT_ROI_HEIGHT))
        self.clamp_to_roi = bool(rospy.get_param("~clamp_to_roi", DEFAULT_CLAMP_TO_ROI))

        # New gating params (raw ddx + rolling distance)
        # window size (number of samples) for rolling distance
        self.lc_window_size = int(rospy.get_param("~lc_window_size", 5))
        # minimum pixel displacement within window
        self.lc_dist_threshold = float(rospy.get_param("~lc_dist_threshold", 200.0))
        # ddx threshold (absolute)
        self.lc_ddx_threshold = float(rospy.get_param("~lc_ddx_threshold", 30000.0))

        # ---- freeze on serial loss (FINAL only) ----
        self.freeze_on_serial_loss = rospy.get_param("~freeze_on_serial_loss", True)
        self.frozen = False
        self.frozen_since = None
        self.last_serial_ready = None

        # ---- sync primitives (must exist before any callbacks fire) ----
        self.lock = threading.Lock()

        # ---- state ----
        self.mode = "DEFAULT"
        self.last_lane_steer = 0
        self.lane_steer_received = False

        self.state = "start"
        self.last_state = None
        self.start_time = None
        self.start_done = False
        self.lc_complete_time = None
        # timer for lane_driving2 duration
        self.lane_driving2_start_time = None
        # timer for automatic transition in lane_driving1
        self.lane_driving1_start_time = None

        # lane-change reason latch (for viz)
        # "none" | "yolo"
        self.lane_change_reason = "none"

        # ---- queues ----
        self.yolo_queue = deque(maxlen=self.queues_maxlen)
        self.yolo_crash = False

        self.cur_lane = 2

        self.lc_start_time = None

        self.traffic_light = 0
        self.traffic_queue = deque(maxlen=self.queues_maxlen)
        self.traffic_start_time = None

        self.left_lane_change_complete = False
        self.right_lane_change_complete = False
        self.crossline = False
        self.traffic_stop = False

        self.serial_ok = False
        self.serial_received = False
        self.serial_last_time = None

        # ---- lane change state ----
        self.latest_lines = None
        self.last_lane_timer_time = None
        # EMA states removed; raw-history buffers used for gating
        self.y_top = int(self.image_height * (1.0 - self.roi_height))
        self.y_mid = int(self.image_height * (1.0 - self.roi_height / 2.0))
        self.y_bottom = self.image_height
        self.roi_left_x_mid, self.roi_right_x_mid = self._roi_x_bounds_at_y(self.y_mid)

        # history buffers for raw-based gating
        self.left_hist = deque(maxlen=self.lc_window_size)  # stores tuples (t_sec, x)
        self.right_hist = deque(maxlen=self.lc_window_size)
        self.left_last_dx = None
        self.right_last_dx = None

        # ---- subs ----
        rospy.Subscriber("/lane_steer", Int16, self.lane_steer_callback, queue_size=1)
        rospy.Subscriber("/cur_lane", Int16, self.cur_lane_callback, queue_size=1)
        rospy.Subscriber("/car_projected", PoseArray, self.car_projected_callback, queue_size=1)
        rospy.Subscriber("/traffic", Int16, self.traffic_callback, queue_size=1)
        rospy.Subscriber("/crossline", Int16, self.crossline_callback, queue_size=1)
        rospy.Subscriber("/rosserial_check", Int16, self.serial_check_callback, queue_size=1)
        rospy.Subscriber(self.lane_lines_topic, Int32MultiArray, self.lane_lines_callback, queue_size=1)

        # ---- pubs (actuation) ----
        self.motor_cmd_steer_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.motor_long_pub = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)

        # ---- pubs (viz/status) ----
        # 매 루프 publish (20Hz)로 사용
        self.state_pub = rospy.Publisher("/final_planner/state", String, queue_size=1)
        self.yolo_crash_pub = rospy.Publisher("/final_planner/yolo_crash", Bool, queue_size=1)
        self.reason_pub = rospy.Publisher("/final_planner/lane_change_reason", String, queue_size=1)

        self.yolo_crash_point_pub = rospy.Publisher("/final_planner/yolo_crash_point", PointStamped, queue_size=1)
        self.last_yolo_true_point = None
        # ---- lane change debug pubs (plotting) ----
        # Publishers for raw dx/ddx values (for plotting/debug)
        self.pub_left_x = rospy.Publisher('lane_change/left/x', Float32, queue_size=10)
        self.pub_right_x = rospy.Publisher('lane_change/right/x', Float32, queue_size=10)
        self.pub_left_dx = rospy.Publisher('lane_change/left/dx', Float32, queue_size=10)
        self.pub_left_ddx = rospy.Publisher('lane_change/left/ddx', Float32, queue_size=10)
        self.pub_right_dx = rospy.Publisher('lane_change/right/dx', Float32, queue_size=10)
        self.pub_right_ddx = rospy.Publisher('lane_change/right/ddx', Float32, queue_size=10)
        # Publishers for lane change completion flags
        self.pub_left_complete = rospy.Publisher('lane_change/left/complete', Bool, queue_size=10)
        self.pub_right_complete = rospy.Publisher('lane_change/right/complete', Bool, queue_size=10)

        th = threading.Thread(target=self.keyboard_listener, daemon=True)
        th.start()
        self.rate = rospy.Rate(self.rate_hz)
        self.node_start_time = rospy.Time.now()

        period = 1.0 / float(self.rate_hz) if self.rate_hz > 0.0 else 0.05
        rospy.Timer(rospy.Duration(period), self.lane_change_timer_callback)
        self.run()

    # ---------------- callbacks ----------------
    def car_projected_callback(self, msg: PoseArray):
        # ✅ FIX: init 중 콜백이 먼저 들어오면 yolo_queue가 아직 없을 수 있음
        if not hasattr(self, "yolo_queue"):
            return
        if self._startup_blocked():
            return

        nearest_point = None
        nearest_dist = None
        inside_any = False

        # Process each detected point in the PoseArray
        for pose in msg.poses:
            x = pose.position.x - self.roi_offset_x
            y = pose.position.y

            distance = math.sqrt(x**2 + y**2)

            # sector ROI: inside if within radius AND within angular sector [roi_angle_min_deg, roi_angle_max_deg] (deg)
            if distance <= self.roi_radius:
                angle_deg = math.degrees(math.atan2(y, x))
                # normalize angles to [-180, 180)
                def _norm(a):
                    return ((a + 180.0) % 360.0) - 180.0
                a = _norm(angle_deg)
                amin = _norm(self.roi_angle_min_deg)
                amax = _norm(self.roi_angle_max_deg)
                if amin <= amax:
                    inside = (a >= amin) and (a <= amax)
                else:
                    # wrap-around interval (e.g., amin=150, amax=-150)
                    inside = (a >= amin) or (a <= amax)
            else:
                inside = False

            if inside:
                inside_any = True
                # Keep track of nearest point
                if nearest_dist is None or distance < nearest_dist:
                    nearest_dist = distance
                    nearest_point = (x, y)

        if inside_any and nearest_point is not None:
            ps = PointStamped()
            ps.header = msg.header
            ps.point.x = nearest_point[0]
            ps.point.y = nearest_point[1]
            ps.point.z = 0.0
            self.last_yolo_true_point = ps

        self.yolo_queue.append(inside_any)

        if len(self.yolo_queue) == self.yolo_queue.maxlen:
            prev_yolo_crash = bool(self.yolo_crash)
            count_over = sum(1 for v in self.yolo_queue if v)
            self.yolo_crash = count_over >= self.yolo_count_threshold

            if (not prev_yolo_crash) and self.yolo_crash and (self.last_yolo_true_point is not None):
                try:
                    self.yolo_crash_point_pub.publish(self.last_yolo_true_point)
                except Exception:
                    pass

    def cur_lane_callback(self, msg: Int16):
        self.cur_lane = msg.data

    def lane_steer_callback(self, msg: Int16):
        with self.lock:
            self.last_lane_steer = int(msg.data)
            self.lane_steer_received = True

    def traffic_callback(self, msg: Int16):
        val = int(msg.data)
        with self.lock:
            self.traffic_light = val
            if val in (1, 3):
                self.traffic_queue.append(val)

    def crossline_callback(self, msg: Int16):
        val = int(msg.data)
        with self.lock:
            if val == 1:
                self.crossline = True
            else:
                self.crossline = False

    def serial_check_callback(self, msg: Int16):
        val = int(msg.data)
        with self.lock:
            self.serial_received = True
            self.serial_ok = (val == 0)
            self.serial_last_time = rospy.Time.now()

    # ---------------- lane change metrics ----------------
    def lane_lines_callback(self, msg: Int32MultiArray):
        if not msg.data or len(msg.data) < 8:
            with self.lock:
                self.latest_lines = None
            return
        with self.lock:
            self.latest_lines = list(msg.data[:8])

    @staticmethod
    def _line_valid(line):
        return line and len(line) == 4 and all(v != -1 for v in line)

    @staticmethod
    def _interp_x_at_y(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return None
        return (float(y) - y1) * (float(x2) - x1) / (float(y2) - y1) + x1

    def _roi_x_bounds_at_y(self, y):
        cx = self.image_width // 2
        x_left_bottom = cx - int(self.image_width * 0.5)
        x_left_top = cx - int(self.image_width * 0.45)
        x_right_top = cx + int(self.image_width * 0.45)
        x_right_bottom = cx + int(self.image_width * 0.5)

        if self.y_bottom == self.y_top:
            return float(x_left_top), float(x_right_top)

        t = (float(y) - self.y_top) / float(self.y_bottom - self.y_top)
        x_left = x_left_top + t * (x_left_bottom - x_left_top)
        x_right = x_right_top + t * (x_right_bottom - x_right_top)
        return float(x_left), float(x_right)

    def lane_change_timer_callback(self, event):
        with self.lock:
            latest_lines = self.latest_lines

        if latest_lines is None:
            return

        if self.last_lane_timer_time is None:
            self.last_lane_timer_time = event.current_real
            return

        dt = (event.current_real - self.last_lane_timer_time).to_sec()
        self.last_lane_timer_time = event.current_real
        if dt <= 0.0:
            return

        left_line = latest_lines[0:4]
        right_line = latest_lines[4:8]

        left_x = None
        right_x = None

        if self._line_valid(left_line):
            left_x = self._interp_x_at_y(left_line, self.y_mid)
        if self._line_valid(right_line):
            right_x = self._interp_x_at_y(right_line, self.y_mid)

        if self.clamp_to_roi:
            if not _finite(left_x):
                left_x = self.roi_left_x_mid
            if not _finite(right_x):
                right_x = self.roi_right_x_mid

        # EMA metrics removed; using raw-based gating below

        # --- Raw-based gating (replace EMA-ddx gating) ---
        # Append current sample (time, x) to history buffers
        t_sec = event.current_real.to_sec()
        # prepare values for publishing
        left_dx_val = None
        left_ddx_val = None
        right_dx_val = None
        right_ddx_val = None

        # LEFT
        if _finite(left_x):
            # push sample
            self.left_hist.append((t_sec, float(left_x)))
            # compute dx and ddx using raw samples
            left_hit = False
            left_dx_val = None
            left_ddx_val = None
            if len(self.left_hist) >= 2:
                # last two samples
                t1, x1 = self.left_hist[-2]
                t2, x2 = self.left_hist[-1]
                dt_sample = t2 - t1 if (t2 - t1) != 0 else None
                left_dx = None
                left_ddx = None
                if dt_sample is not None and dt_sample > 0:
                    left_dx = (x2 - x1) / dt_sample
                    if self.left_last_dx is not None:
                        left_ddx = (left_dx - self.left_last_dx) / dt_sample
                    self.left_last_dx = left_dx
                    left_dx_val = left_dx
                    left_ddx_val = left_ddx

                # rolling distance over window
                if len(self.left_hist) >= self.lc_window_size:
                    xs = [s[1] for s in self.left_hist]
                    rolling_dist = max(xs) - min(xs)
                else:
                    rolling_dist = 0.0

                # gating: both abs(ddx) and rolling distance must exceed thresholds
                if left_ddx is not None and abs(left_ddx) > self.lc_ddx_threshold and rolling_dist > self.lc_dist_threshold:
                    left_hit = True
            else:
                left_hit = False
        else:
            left_hit = False

        # RIGHT (symmetric)
        if _finite(right_x):
            self.right_hist.append((t_sec, float(right_x)))
            right_hit = False
            if len(self.right_hist) >= 2:
                t1, x1 = self.right_hist[-2]
                t2, x2 = self.right_hist[-1]
                dt_sample = t2 - t1 if (t2 - t1) != 0 else None
                right_dx = None
                right_ddx = None
                if dt_sample is not None and dt_sample > 0:
                    right_dx = (x2 - x1) / dt_sample
                    if self.right_last_dx is not None:
                        right_ddx = (right_dx - self.right_last_dx) / dt_sample
                    self.right_last_dx = right_dx
                    right_dx_val = right_dx
                    right_ddx_val = right_ddx

                if len(self.right_hist) >= self.lc_window_size:
                    xs = [s[1] for s in self.right_hist]
                    rolling_dist_r = max(xs) - min(xs)
                else:
                    rolling_dist_r = 0.0

                if right_ddx is not None and abs(right_ddx) > self.lc_ddx_threshold and rolling_dist_r > self.lc_dist_threshold:
                    right_hit = True
            else:
                right_hit = False
        else:
            right_hit = False

        # Latch completion flags
        if left_hit or right_hit:
            with self.lock:
                if left_hit:
                    self.left_lane_change_complete = True
                    try:
                        self.pub_left_complete.publish(Bool(True))
                    except Exception:
                        pass
                if right_hit:
                    self.right_lane_change_complete = True
                    try:
                        self.pub_right_complete.publish(Bool(True))
                    except Exception:
                        pass

        # Publish raw dx/ddx for plotting (use invalid_value when not finite)
        try:
            self.pub_left_x.publish(Float32(left_x if _finite(left_x) else self.invalid_value))
            self.pub_right_x.publish(Float32(right_x if _finite(right_x) else self.invalid_value))
            self.pub_left_dx.publish(Float32(left_dx_val if _finite(left_dx_val) else self.invalid_value))
            self.pub_left_ddx.publish(Float32(left_ddx_val if _finite(left_ddx_val) else self.invalid_value))
            self.pub_right_dx.publish(Float32(right_dx_val if _finite(right_dx_val) else self.invalid_value))
            self.pub_right_ddx.publish(Float32(right_ddx_val if _finite(right_ddx_val) else self.invalid_value))
        except Exception:
            pass

    # ---------------- keyboard ----------------
    def _log(self, error=False):
        serial_txt = "SERIAL OK" if self._serial_ready() else "SERIAL ERROR"
        state_txt = "default" if self.mode == "DEFAULT" else self.state.lower()
        line = f"[FINAL_PLANNER] | {serial_txt} | State = {state_txt}"
        if error:
            rospy.logwarn(line)
        else:
            rospy.loginfo(line)

    def keyboard_listener(self):
        while not rospy.is_shutdown():
            key = input().strip().lower()
            with self.lock:
                if key == "d":
                    self.mode = "DEFAULT"
                    self._log()
                elif key == "f":
                    if self._serial_ready():
                        self.mode = "FINAL"
                        self._log()
                    else:
                        self._log(error=True)
                else:
                    self._log()

    # ---------------- helper: reason latch ----------------
    def _compute_reason(self) -> str:
        y = bool(self.yolo_crash)
        if y:
            return "yolo"
        return "none"

    def _startup_blocked(self) -> bool:
        if self.state_change_delay_sec <= 0.0:
            return False
        return (rospy.Time.now() - self.node_start_time).to_sec() < self.state_change_delay_sec

    def _serial_ready(self) -> bool:
        if not self.serial_received or not self.serial_ok:
            return False
        if self.serial_timeout_sec <= 0.0:
            return True
        if self.serial_last_time is None:
            return False
        return (rospy.Time.now() - self.serial_last_time).to_sec() <= self.serial_timeout_sec

    # ---------------- main loop ----------------
    def run(self):
        while not rospy.is_shutdown():
            # snapshot (thread-safe)
            with self.lock:
                mode = self.mode
                lane_steer = self.last_lane_steer if self.lane_steer_received else 0
                state_local = self.state
                reason_local = self.lane_change_reason

            serial_ready = self._serial_ready()
            if self.last_serial_ready is None or serial_ready != self.last_serial_ready:
                self._log(error=not serial_ready)
                self.last_serial_ready = serial_ready

            # log state transitions
            if state_local != self.last_state:
                if state_local == "lane_driving":
                    self._log()
                elif state_local == "lane_change":
                    self._log()
                elif state_local == "crossline":
                    self._log()
                elif state_local == "traffic":
                    self._log()
                else:
                    self._log()

                self.last_state = state_local

            # ---- FREEZE behavior (FINAL + serial lost) ----
            if mode == "FINAL" and self.freeze_on_serial_loss and (not serial_ready):
                with self.lock:
                    if not self.frozen:
                        self.frozen = True
                        self.frozen_since = rospy.Time.now()
                        rospy.logwarn("[FINAL_PLANNER] SERIAL LOST -> FREEZE (keep state, stop commands)")

                # ---- publish status for viz (EVERY LOOP) ----
                self.state_pub.publish(String(self.state))
                self.yolo_crash_pub.publish(Bool(bool(self.yolo_crash)))
                self.reason_pub.publish(String(str(self.lane_change_reason)))

                self.rate.sleep()
                continue

            # serial recovered -> unfreeze
            if self.frozen and serial_ready:
                with self.lock:
                    self.frozen = False
                    self.frozen_since = None
                rospy.logwarn("[FINAL_PLANNER] SERIAL RECOVERED -> RESUME")

            # ---- planner logic ----
            if mode == "DEFAULT":
                self.drive(0, self.default_motor)

            elif mode == "FINAL":
                # start state: ramp 0 -> 255 once, then go to lane_driving
                if self.state == "start":
                    if self.start_done:
                        self.state = "lane_driving1"
                    else:
                        if self.start_time is None:
                            self.start_time = rospy.Time.now()
                        ramp_sec = max(0.001, float(self.start_ramp_sec))
                        elapsed = (rospy.Time.now() - self.start_time).to_sec()
                        ratio = max(0.0, min(1.0, elapsed / ramp_sec))
                        speed_cmd = int(round(255 * ratio))
                        self.drive(lane_steer, speed_cmd)
                        if ratio >= 1.0:
                            self.start_done = True
                            self.state = "lane_driving1"
                            self.start_time = None

                # lane driving
                elif self.state == "lane_driving1":
                    # normal lane following: start a timeout on entry and then transition
                    # lane_change_reason stays 'none' unless other logic sets it
                    self.lane_change_reason = "none"
                    # publish는 아래에서 공통으로 한 번에 함
                    self.drive(lane_steer, self.SPEED_HIGH)

                    # start/monitor the lane driving timeout; transition after configured duration
                    with self.lock:
                        if self.lane_driving1_start_time is None:
                            self.lane_driving1_start_time = rospy.Time.now()
                        else:
                            ld1_elapsed = (rospy.Time.now() - self.lane_driving1_start_time).to_sec()
                            if ld1_elapsed >= self.lane_driving1_timeout:
                                self.state = "lane_change_to_left"
                                self.lane_driving1_start_time = None
                                self.left_lane_change_complete =  False
                # lane change to left
                elif self.state == "lane_change_to_left":

                    # step 1: 좌꺽
                    self.drive(self.LC_STEER, self.SPEED_HIGH)
                    with self.lock:
                        # clear any lane_driving timers on entry
                        self.lane_driving2_crash_start = None
                        self.lane_driving2_start_time = None
                        if self.left_lane_change_complete: #차선 변경이 되었다고 판단하는 flag
                            if self.lc_complete_time is None:
                                self.lc_complete_time = rospy.Time.now()
                            else:
                                lc_elapsed = (rospy.Time.now() - self.lc_complete_time).to_sec()
                                if lc_elapsed >= self.left_lc_complete_delay:
                                    # delay elapsed -> move to next state
                                    self.state = "lane_driving2"
                                    self.lc_complete_time = None
                                    self.left_lane_change_complete = False
                                    self.yolo_queue.clear()
                                    self.yolo_crash= False
                        else:
                            self.lc_complete_time = None

                elif self.state == "lane_driving2":
                    print(list(self.yolo_queue))
                    # STEP 2: 차선 인식 주행 (유지 시간 동안 주행을 계속하고, 만료되면 다음 상태로 전환)
                    self.drive(lane_steer, self.SPEED_HIGH)
                    with self.lock:
                        # YOLO-triggered delayed transition (similar to lane_driving1 behavior)

                        if self.yolo_crash:
                            self.state = "lane_change_to_right"
                            self.lane_change_reason = "yolo"
                            self.right_lane_change_complete =  False
                        
                elif self.state == "lane_change_to_right":

                    # step 3: 우꺽
                    with self.lock:
                        # clear any lane_driving timers on entry
                        self.lane_driving2_crash_start = None
                        self.lane_driving2_start_time = None
                        if self.right_lane_change_complete: #차선 변경이 되었다고 판단하는 flag
                            if self.lc_complete_time is None:
                                self.lc_complete_time = rospy.Time.now()
                            else:
                                lc_elapsed = (rospy.Time.now() - self.lc_complete_time).to_sec()
                                self.drive(0, self.SPEED_HIGH)
                                if lc_elapsed >= self.right_lc_complete_delay:
                                    # delay elapsed -> move to next state
                                    self.state = "crossline"
                                    self.lc_complete_time = None
                                    self.right_lane_change_complete = False
                        else:
                            self.drive(-self.LC_STEER, self.SPEED_HIGH)

                            self.lc_complete_time = None

                elif self.state == "crossline":
                    if self.crossline == 1:  # 횡단보도 정지.
                        self.drive(lane_steer, self.SPEED_0)
                        self.traffic_queue.clear()
                        self.traffic_start_time = rospy.Time.now()
                        self.state = "traffic"
                    else:
                        self.drive(lane_steer, self.SPEED_MID)

                # traffic
                elif self.state == "traffic":
                    with self.lock:
                        if self.traffic_start_time is None:
                            self.traffic_start_time = rospy.Time.now()
                        traffic_elapsed = (rospy.Time.now() - self.traffic_start_time).to_sec()
                        green_count = sum(1 for v in self.traffic_queue if v == 1)
                        if (green_count >= self.traffic_green_threshold) or (traffic_elapsed >= self.traffic_green_timeout):
                            self.state = "lane_driving2"
                            self.traffic_start_time = None
                        else:
                            self.drive(lane_steer, self.SPEED_0)

            # ---- publish status for viz (EVERY LOOP) ----
            # NOTE: crash bool은 "현재 상태"를 그대로 publish.
            # lane-change 동안 주황 유지가 필요하면 viz 노드에서 state 기반 latch하면 됨.
            self.state_pub.publish(String(self.state))
            self.yolo_crash_pub.publish(Bool(bool(self.yolo_crash)))
            self.reason_pub.publish(String(str(self.lane_change_reason)))
            # publish lane change completion flags every loop
            try:
                self.pub_left_complete.publish(Bool(bool(self.left_lane_change_complete)))
                self.pub_right_complete.publish(Bool(bool(self.right_lane_change_complete)))
            except Exception:
                pass

            self.rate.sleep()

    # ---------------- actuation ----------------
    def drive(self, des_steer, long_cmd):
        self.motor_cmd_steer_pub.publish(Int16(int(des_steer)))
        self.motor_long_pub.publish(Int16(int(long_cmd)))


if __name__ == "__main__":
    try:
        FinalPlanner()
    except rospy.ROSInterruptException:
        pass


# [ERROR] [1770254058.165328]: bad callback: <bound method FinalPlanner.car_projected_callback of <__main__.FinalPlanner object at 0x7f6cd66570d0>>
# Traceback (most recent call last):
#   File "/opt/ros/noetic/lib/python3/dist-packages/rospy/topics.py", line 750, in _invoke_callback
#     cb(msg)
#   File "/home/vic/kkdws/src/pre_final_planner/scripts/final_planner.py", line 208, in car_projected_callback
#     if self._startup_blocked():
#   File "/home/vic/kkdws/src/pre_final_planner/scripts/final_planner.py", line 508, in _startup_blocked
#     return (rospy.Time.now() - self.node_start_time).to_sec() < self.state_change_delay_sec
# AttributeError: 'FinalPlanner' object has no attribute 'node_start_time'

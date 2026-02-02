#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading
from std_msgs.msg import Int16, Bool, String
from geometry_msgs.msg import PointStamped
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


class FinalPlanner:
    def __init__(self):
        rospy.init_node("final_planner", anonymous=False)

        self.rate_hz = rospy.get_param("~rate_hz", 20)
        self.default_motor = rospy.get_param("~default_motor", 0)

        # ---- subs ----
        rospy.Subscriber("/lane_steer", Int16, self.lane_steer_callback, queue_size=1)
        rospy.Subscriber("/cur_lane", Int16, self.cur_lane_callback, queue_size=1)
        rospy.Subscriber("/car_projected", PointStamped, self.car_projected_callback, queue_size=1)
        rospy.Subscriber("/traffic", Int16, self.traffic_callback, queue_size=1)
        rospy.Subscriber("/crossline", Int16, self.crossline_callback, queue_size=1)
        rospy.Subscriber("/rosserial_check", Int16, self.serial_check_callback, queue_size=1)

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

        self.lock = threading.Lock()
        th = threading.Thread(target=self.keyboard_listener, daemon=True)
        th.start()
        self.rate = rospy.Rate(self.rate_hz)
        self.node_start_time = rospy.Time.now()

        # ---- params ----
        self.roi_offset_x = rospy.get_param("planner_common/roi/offset_x", 0.74)

        self.roi_radius = rospy.get_param("planner_common/roi/radius", ROI_RADIUS)

        self.SPEED_0 = rospy.get_param("~speed_0", SPEED_0)
        self.SPEED_MID = rospy.get_param("~speed_mid", SPEED_MID)
        self.SPEED_HIGH = rospy.get_param("~speed_high", SPEED_HIGH)

        self.LC_STEER = rospy.get_param("~lc_steer", LC_STEER)
        self.steer_time1 = rospy.get_param("~steer_time1", STEER_TIME1)
        self.steer_time2 = rospy.get_param("~steer_time2", STEER_TIME2)
        self.steer_time3 = rospy.get_param("~steer_time3", STEER_TIME3)

        self.straight_time1 = rospy.get_param("~straight_time1", STRAIGHT_TIME1)
        self.straight_time2 = rospy.get_param("~straight_time2", STRAIGHT_TIME2)

        # delay after left lane change complete before transitioning to lane_driving2 (sec)
        self.left_lc_complete_delay = rospy.get_param("~left_lc_complete_delay_sec", 0.0)
        self.right_lc_complete_delay = rospy.get_param("~right_lc_cmoplete_delay_sec", 0.0)

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

        # ---- freeze on serial loss (FINAL only) ----
        self.freeze_on_serial_loss = rospy.get_param("~freeze_on_serial_loss", True)
        self.frozen = False
        self.frozen_since = None
        self.last_serial_ready = None

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
        self.run()

    # ---------------- callbacks ----------------
    def car_projected_callback(self, msg: PointStamped):
        # ✅ FIX: init 중 콜백이 먼저 들어오면 yolo_queue가 아직 없을 수 있음
        if not hasattr(self, "yolo_queue"):
            return
        if self._startup_blocked():
            return

        x = msg.point.x - self.roi_offset_x
        y = msg.point.y

        distance = math.sqrt(x**2 + y**2)
        inside = (x>=0) and (distance <=self.roi_radius)

        if inside:
            ps = PointStamped()
            ps.header = msg.header
            ps.point.x = x
            ps.point.y = y
            ps.point.z = msg.point.z
            self.last_yolo_true_point = ps

        self.yolo_queue.append(inside)

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
                        else:
                            self.lc_complete_time = None

                elif self.state == "lane_driving2":

                    # STEP 2: 차선 인식 주행 (유지 시간 동안 주행을 계속하고, 만료되면 다음 상태로 전환)
                    self.drive(lane_steer, self.SPEED_HIGH)
                    with self.lock:
                        # YOLO-triggered delayed transition (similar to lane_driving1 behavior)
                        if self.yolo_crash and (not self._startup_blocked()):
                            self.state = "lane_change_to_right"
                            self.lane_change_reason = "yolo"

                elif self.state == "lane_change_to_right":

                    # step 3: 우꺽
                    self.drive(-self.LC_STEER, self.SPEED_HIGH)
                    with self.lock:
                        # clear any lane_driving timers on entry
                        self.lane_driving2_crash_start = None
                        self.lane_driving2_start_time = None
                        if self.right_lane_change_complete: #차선 변경이 되었다고 판단하는 flag
                            if self.lc_complete_time is None:
                                self.lc_complete_time = rospy.Time.now()
                            else:
                                lc_elapsed = (rospy.Time.now() - self.lc_complete_time).to_sec()
                                if lc_elapsed >= self.right_lc_complete_delay:
                                    # delay elapsed -> move to next state
                                    self.state = "crossline"
                                    self.lc_complete_time = None
                                    self.right_lane_change_complete = False
                        else:
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

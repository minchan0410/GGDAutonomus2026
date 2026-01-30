#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import threading
from std_msgs.msg import Int16, Bool, String
from geometry_msgs.msg import PointStamped
from collections import deque

ROI_MIN_X = 0
ROI_MAX_X = 0.5
ROI_MIN_Y = -0.4
ROI_MAX_Y = 0.4

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
        rospy.Subscriber("/ultrasonic", Int16, self.ultrasonic1_callback, queue_size=1)
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
        self.sonic_crash_pub = rospy.Publisher("/final_planner/sonic_crash", Bool, queue_size=1)
        self.reason_pub = rospy.Publisher("/final_planner/lane_change_reason", String, queue_size=1)

        self.lock = threading.Lock()
        th = threading.Thread(target=self.keyboard_listener, daemon=True)
        th.start()
        self.rate = rospy.Rate(self.rate_hz)
        self.node_start_time = rospy.Time.now()

        # ---- params ----
        self.roi_min_x = rospy.get_param("planner_common/roi/min_x", ROI_MIN_X)
        self.roi_max_x = rospy.get_param("planner_common/roi/max_x", ROI_MAX_X)
        self.roi_min_y = rospy.get_param("planner_common/roi/min_y", ROI_MIN_Y)
        self.roi_max_y = rospy.get_param("planner_common/roi/max_y", ROI_MAX_Y)
        self.roi_offset_x = rospy.get_param("planner_common/roi/offset_x", 0.74)

        self.SPEED_0 = rospy.get_param("~speed_0", SPEED_0)
        self.SPEED_MID = rospy.get_param("~speed_mid", SPEED_MID)
        self.SPEED_HIGH = rospy.get_param("~speed_high", SPEED_HIGH)

        self.LC_STEER = rospy.get_param("~lc_steer", LC_STEER)
        self.steer_time1 = rospy.get_param("~steer_time1", STEER_TIME1)
        self.steer_time2 = rospy.get_param("~steer_time2", STEER_TIME2)
        self.steer_time3 = rospy.get_param("~steer_time3", STEER_TIME3)

        self.straight_time1 = rospy.get_param("~straight_time1", STRAIGHT_TIME1)
        self.straight_time2 = rospy.get_param("~straight_time2", STRAIGHT_TIME2)

        self.ultrasonic_threshold = rospy.get_param("planner_common/ultrasonic/threshold", 300)
        self.queues_maxlen = rospy.get_param("~queues_maxlen", 10)
        self.yolo_count_threshold = rospy.get_param("~yolo_count_threshold", 7)
        self.ultrasonic_count_threshold = rospy.get_param("~ultrasonic_count_threshold", 7)

        self.traffic_green_threshold = rospy.get_param("~traffic_green_threshold", 5)
        self.traffic_green_timeout = rospy.get_param("~traffic_green_timeout", 20.0)
        # startup guard: block state changes for a few seconds
        self.state_change_delay_sec = rospy.get_param("~state_change_delay_sec", 0.0)
        # serial readiness gate
        self.serial_timeout_sec = rospy.get_param("~serial_timeout_sec", 0.5)

        # ---- state ----
        self.mode = "DEFAULT"
        self.last_lane_steer = 0
        self.lane_steer_received = False

        self.state = "lane_driving"
        self.last_state = None

        # lane-change reason latch (for viz)
        # "none" | "yolo" | "sonic" | "both"
        self.lane_change_reason = "none"

        # ---- queues ----
        self.ultrasonic_queue = deque(maxlen=self.queues_maxlen)
        self.ultrasonic_crash = False

        self.yolo_queue = deque(maxlen=self.queues_maxlen)
        self.yolo_crash = False

        self.cur_lane = 2

        self.lc_start_time = None

        self.traffic_light = 0
        self.traffic_queue = deque(maxlen=self.queues_maxlen)
        self.traffic_start_time = None

        self.crossline = False
        self.traffic_stop = False

        self.serial_ok = False
        self.serial_received = False
        self.serial_last_time = None
        self.run()

    # ---------------- callbacks ----------------
    def ultrasonic1_callback(self, msg: Int16):
        if self._startup_blocked():
            return
        if msg.data == -1:
            self.ultrasonic_queue.append(float('inf'))
        else:
            self.ultrasonic_queue.append(msg.data)

            if len(self.ultrasonic_queue) == self.ultrasonic_queue.maxlen:
                count_over = sum(1 for v in self.ultrasonic_queue if v <= self.ultrasonic_threshold)
                self.ultrasonic_crash = count_over >= self.ultrasonic_count_threshold

    def car_projected_callback(self, msg: PointStamped):
        # ✅ FIX: init 중 콜백이 먼저 들어오면 yolo_queue가 아직 없을 수 있음
        if not hasattr(self, "yolo_queue"):
            return
        if self._startup_blocked():
            return

        x = msg.point.x
        y = msg.point.y

        inside = ((self.roi_min_x + self.roi_offset_x) <= x <= (self.roi_max_x + self.roi_offset_x)) and \
                (self.roi_min_y <= y <= self.roi_max_y)

        self.yolo_queue.append(inside)

        if len(self.yolo_queue) == self.yolo_queue.maxlen:
            count_over = sum(1 for v in self.yolo_queue if v)
            self.yolo_crash = count_over >= self.yolo_count_threshold

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
    def _log(self, error=False, throttle=None):
        serial_txt = "Serial OK" if self.serial_ok else "Serial ERROR"
        state_txt = self.state
        tail = "ERROR" if error else ""
        line = f"[FINAL_PLANNER] | {serial_txt:<11} | State = {state_txt:<12} | {tail}"
        if throttle is None:
            if error:
                rospy.logwarn(line)
            else:
                rospy.loginfo(line)
        else:
            if error:
                rospy.logwarn_throttle(throttle, line)
            else:
                rospy.loginfo_throttle(throttle, line)

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
        s = bool(self.ultrasonic_crash)
        if y and s:
            return "both"
        if y:
            return "yolo"
        if s:
            return "sonic"
        return "none"

    def _enter_lane_change(self, target_state: str):
        """
        lane_driving -> lane_change_* 진입 시점에 호출:
        - state 설정
        - lane_change_reason latch
        """
        self.state = target_state
        self.lane_change_reason = self._compute_reason()

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

            # ---- planner logic ----
            if mode == "DEFAULT":
                if not self._serial_ready():
                    self._log(error=True, throttle=0.2)
                self.drive(0, self.default_motor)

            elif mode == "FINAL":
                # serial lost -> force DEFAULT
                if not self._serial_ready():
                    with self.lock:
                        self.mode = "DEFAULT"
                    self._log(error=True, throttle=0.2)
                    self.drive(0, self.default_motor)
                    self.rate.sleep()
                    continue
                # lane driving
                if self.state == "lane_driving":
                    self.lane_change_reason = "none"
                    # publish는 아래에서 공통으로 한 번에 함
                    self.drive(lane_steer, self.SPEED_HIGH)

                    # crash triggers -> lane change
                    if (self.ultrasonic_crash or self.yolo_crash) and (not self._startup_blocked()):
                        with self.lock:
                            self._enter_lane_change("lane_change")

                # lane change
                if self.state == "lane_change":
                    if self.lc_start_time is None:
                        self.lc_start_time = rospy.Time.now()
                        self.lc_step = 0

                    lc_elapsed = (rospy.Time.now() - self.lc_start_time).to_sec()

                    # STEP 0: 좌로 꺾기
                    if self.lc_step == 0:
                        self.drive(self.LC_STEER, self.SPEED_HIGH)
                        if lc_elapsed >= self.steer_time1:
                            self.lc_step = 1
                            self.lc_start_time = rospy.Time.now()

                    # STEP 1: 직진
                    elif self.lc_step == 1:
                        self.drive(0, self.SPEED_HIGH)
                        if lc_elapsed >= self.straight_time1:
                            self.lc_step = 2
                            self.lc_start_time = rospy.Time.now()

                    # STEP 2: 우로 꺾기
                    elif self.lc_step == 2:
                        self.drive(-self.LC_STEER, self.SPEED_HIGH)
                        if lc_elapsed >= self.steer_time2:
                            self.lc_step = 3
                            self.lc_start_time = rospy.Time.now()

                    # STEP 3: 직진
                    elif self.lc_step == 3:
                        self.drive(0, self.SPEED_HIGH)
                        if lc_elapsed >= self.straight_time2:
                            self.lc_step = 4
                            self.lc_start_time = rospy.Time.now()

                    # STEP 4: 좌꺽
                    elif self.lc_step == 4:
                        self.drive(0, self.SPEED_HIGH)
                        if lc_elapsed >= self.steer_time3:
                            self.lc_step = 5
                            self.lc_start_time = rospy.Time.now()

                    # STEP 3: 종료 -> lane_driving 복귀
                    elif self.lc_step == 5:
                        with self.lock:
                            self.state = "crossline"
                            self.lc_start_time = None


                if self.state == "crossline":
                    if self.crossline == 1: #횡단보도 정지. 
                        self.drive(lane_steer, self.SPEED_0)
                        self.traffic_queue.clear()
                        self.traffic_start_time = rospy.Time.now()
                        self.state = "traffic"
                    else:
                        self.drive(lane_steer, self.SPEED_MID)
                
                # traffic
                if self.state == "traffic":
                    with self.lock:
                        if self.traffic_start_time is None:
                            self.traffic_start_time = rospy.Time.now()
                        traffic_elapsed = (rospy.Time.now() - self.traffic_start_time).to_sec()
                        green_count = sum(1 for v in self.traffic_queue if v == 1)
                        if (green_count >= self.traffic_green_threshold) or (traffic_elapsed >= self.traffic_green_timeout):
                            self.state = "lane_driving"
                            self.traffic_start_time = None
                        else:
                            self.drive(lane_steer, self.SPEED_0)



            # ---- publish status for viz (EVERY LOOP) ----
            # NOTE: crash bool은 "현재 상태"를 그대로 publish.
            # lane-change 동안 주황 유지가 필요하면 viz 노드에서 state 기반 latch하면 됨.
            self.state_pub.publish(String(self.state))
            self.yolo_crash_pub.publish(Bool(bool(self.yolo_crash)))
            self.sonic_crash_pub.publish(Bool(bool(self.ultrasonic_crash)))
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

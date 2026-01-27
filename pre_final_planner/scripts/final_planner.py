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
        rospy.Subscriber("/ultrasonic1", Int16, self.ultrasonic1_callback, queue_size=1)
        rospy.Subscriber("/cur_lane", Int16, self.cur_lane_callback, queue_size=1)
        rospy.Subscriber("/car_projected", PointStamped, self.car_projected_callback, queue_size=1)
        rospy.Subscriber("/traffic", Int16, self.traffic_callback, queue_size=1)
        rospy.Subscriber("/crossline", Int16, self.crossline_callback, queue_size=1)

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

        self.traffic_red_threshold = rospy.get_param("~traffic_red_threshold", 5)

        # ---- state ----
        self.mode = "DEFAULT"
        self.last_lane_steer = 0
        self.lane_steer_received = False

        self.state = "lane_driving"
        self.last_state = None

        self.crossline = 0 # not crossline
        # lane-change reason latch (for viz)
        # "none" | "yolo" | "sonic" | "both"
        self.lane_change_reason = "none"

        # ---- queues ----
        self.ultrasonic_queue = deque(maxlen=self.queues_maxlen)
        self.ultrasonic_crash = False

        self.yolo_queue = deque(maxlen=self.queues_maxlen)
        self.yolo_crash = False

        self.cur_lane = 2

        self.start_time = None
        self.wait_for_traffic = False

        self.traffic_light = 0
        self.traffic_queue = deque(maxlen=self.queues_maxlen)

        self.crossline = False

        self.run()

    # ---------------- callbacks ----------------
    def ultrasonic1_callback(self, msg: Int16):
        if msg.data == -1:
            self.ultrasonic_queue.append(float('inf'))
        else:
            self.ultrasonic_queue.append(msg.data)

            if len(self.ultrasonic_queue) == self.ultrasonic_queue.maxlen:
                count_over = sum(1 for v in self.ultrasonic_queue if v <= self.ultrasonic_threshold)
                self.ultrasonic_crash = count_over >= self.ultrasonic_count_threshold

    def crossline_callback(self, msg: Int16):
        self.crossline = msg.data
    
    def car_projected_callback(self, msg: PointStamped):
        # ✅ FIX: init 중 콜백이 먼저 들어오면 yolo_queue가 아직 없을 수 있음
        if not hasattr(self, "yolo_queue"):
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

    # ---------------- keyboard ----------------
    def keyboard_listener(self):
        while not rospy.is_shutdown():
            key = input().strip().lower()
            with self.lock:
                if key == "d":
                    self.mode = "DEFAULT"
                    rospy.loginfo("-> DEFAULT")
                elif key == "f":
                    self.mode = "FINAL"
                    rospy.loginfo("-> FINAL")
                else:
                    rospy.logwarn("invalid key")

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

    def _exit_lane_change_to_lane_driving(self, set_wait_for_traffic: bool):
        """
        lane_change_* -> lane_driving 복귀 시점에 호출:
        - state 리셋
        - reason 리셋 (요구사항: lane_driving으로 돌아오면 초록 복귀)
        """
        self.state = "lane_driving"
        self.start_time = None
        self.wait_for_traffic = bool(set_wait_for_traffic)
        self.lane_change_reason = "none"

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
                    rospy.loginfo("[lane_driving]")
                elif state_local == "lane_change_to_left":
                    rospy.loginfo("[lane change to left]")
                elif state_local == "lane_change_to_right":
                    rospy.loginfo("[lane change to right]")
                elif state_local == "traffic":
                    rospy.loginfo("[traffic]")
                else:
                    rospy.loginfo("[state: %s]" % state_local)

                self.last_state = state_local

            # ---- planner logic ----
            if mode == "DEFAULT":
                self.drive(0, self.default_motor)

            elif mode == "FINAL":
                # lane driving
                if self.state == "lane_driving":
                    if self.wait_for_traffic:
                        with self.lock:
                            self.state = "traffic"
                            self.traffic_queue.clear()
                            # traffic 들어가도 reason은 lane change랑 무관하니 none 유지
                            self.lane_change_reason = "none"
                        # publish는 아래에서 공통으로 한 번에 함
                    else:
                        self.drive(lane_steer, self.SPEED_HIGH)

                        # crash triggers -> decide lane change
                        if self.ultrasonic_crash or self.yolo_crash:
                            if self.cur_lane == 1:
                                # lane 1이면 오른쪽으로 차선 변경
                                with self.lock:
                                    self._enter_lane_change("lane_change_to_right")
                            elif self.cur_lane == 2:
                                # lane 2이면 왼쪽으로 차선 변경
                                with self.lock:
                                    self._enter_lane_change("lane_change_to_left")

                # lane change to right
                if self.state == "lane_change_to_right":
                    if self.start_time is None:
                        self.start_time = rospy.Time.now()
                        self.lc_step = 0

                    elapsed = (rospy.Time.now() - self.start_time).to_sec()

                    # STEP 0: 우로 꺾기
                    if self.lc_step == 0:
                        self.drive(-self.LC_STEER, self.SPEED_HIGH)
                        if elapsed >= self.steer_time1:
                            self.lc_step = 1
                            self.start_time = rospy.Time.now()

                    # STEP 1: 직진
                    elif self.lc_step == 1:
                        self.drive(0, self.SPEED_HIGH)
                        if elapsed >= self.straight_time:
                            self.lc_step = 2
                            self.start_time = rospy.Time.now()

                    # STEP 2: 좌로 꺾기
                    elif self.lc_step == 2:
                        self.drive(self.LC_STEER, self.SPEED_HIGH)
                        if elapsed >= self.steer_time2:
                            self.lc_step = 3
                            self.start_time = rospy.Time.now()

                    # STEP 3: 종료 -> lane_driving 복귀 + traffic 대기
                    elif self.lc_step == 3:
                        with self.lock:
                            self._exit_lane_change_to_lane_driving(set_wait_for_traffic=True)

                # lane change to left
                if self.state == "lane_change_to_left":
                    if self.start_time is None:
                        self.start_time = rospy.Time.now()
                        self.lc_step = 0

                    elapsed = (rospy.Time.now() - self.start_time).to_sec()

                    # STEP 0: 좌로 꺾기
                    if self.lc_step == 0:
                        self.drive(self.LC_STEER, self.SPEED_HIGH)
                        if elapsed >= self.steer_time1:
                            self.lc_step = 1
                            self.start_time = rospy.Time.now()

                    # STEP 1: 직진
                    elif self.lc_step == 1:
                        self.drive(0, self.SPEED_HIGH)
                        if elapsed >= self.straight_time1:
                            self.lc_step = 2
                            self.start_time = rospy.Time.now()

                    # STEP 2: 우로 꺾기
                    elif self.lc_step == 2:
                        self.drive(-self.LC_STEER, self.SPEED_HIGH)
                        if elapsed >= self.steer_time2:
                            self.lc_step = 3
                            self.start_time = rospy.Time.now()

                    # STEP 3: 직진
                    elif self.lc_step == 3:
                        self.drive(0, self.SPEED_HIGH)
                        if elapsed >= self.straight_time2:
                            self.lc_step = 4
                            self.start_time = rospy.Time.now()

                    # STEP 4: 좌꺽
                    elif self.lc_step == 4:
                        self.drive(0, self.SPEED_HIGH)
                        if elapsed >= self.steer_time3:
                            self.lc_step = 5
                            self.start_time = rospy.Time.now()

                    # STEP 3: 종료 -> lane_driving 복귀
                    elif self.lc_step == 5:
                        with self.lock:
                            self._exit_lane_change_to_lane_driving(set_wait_for_traffic=False)

                # traffic
                if self.state == "traffic":
                    with self.lock:
                        red_count = sum(1 for v in self.traffic_queue if v == 1)

                    if red_count >= self.traffic_red_threshold:
                        self.drive(lane_steer, self.SPEED_0)
                    else:
                        self.drive(lane_steer, self.SPEED_MID)

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

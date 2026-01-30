#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16
import time

class Rosserial_checker:
    def __init__(self):
        self.timeout = 0.2   # 0.2초 이상 안 들어오면 경고

        # --- potentiometer ---
        self.last_time_pot = None
        self.msg_count_pot = 0
        self.start_time_pot = time.time()
        rospy.Subscriber("/potentiometer", Int16, self.cb_pot)

        # --- ultrasonic 1~5 ---
        self.ultra_topics = {
            "/ultrasonic1": {"last_time": None, "count": 0, "start": time.time()},
            "/ultrasonic2": {"last_time": None, "count": 0, "start": time.time()},
            "/ultrasonic3": {"last_time": None, "count": 0, "start": time.time()},
            "/ultrasonic4": {"last_time": None, "count": 0, "start": time.time()},
            "/ultrasonic5": {"last_time": None, "count": 0, "start": time.time()},
        }

        # 각 토픽별 subscriber 등록 (topic 이름을 콜백에 바인딩)
        for topic in self.ultra_topics.keys():
            rospy.Subscriber(topic, Int16, self.cb_ultra, callback_args=topic)

        # --- heartbeat ---
        self.heartbeat_pub = rospy.Publisher("/heart_beat", Int16, queue_size=1)
        self.heartbeat_msg = Int16()
        self.heartbeat_value = 1

        # ---rosserial check ---
        # rosserial 상태 퍼블리셔
        self.rosserial_pub = rospy.Publisher("/rosserial_check", Int16, queue_size=1)
        self.rosserial_msg = Int16()

    # potentiometer callback
    def cb_pot(self, msg):
        now = time.time()

        if self.last_time_pot is None:
            self.last_time_pot = now
            return

        dt = now - self.last_time_pot
        self.last_time_pot = now
        self.msg_count_pot += 1

        rospy.loginfo(f"[POT] value: {msg.data}, dt: {dt:.4f}s")

    # ultrasonic callback (topic별로 동일 콜백 사용)
    def cb_ultra(self, msg, topic):
        now = time.time()
        state = self.ultra_topics[topic]

        if state["last_time"] is None:
            state["last_time"] = now
            return

        dt = now - state["last_time"]
        state["last_time"] = now
        state["count"] += 1

        rospy.loginfo(f"[ULTRA {topic}] value: {msg.data}, dt: {dt:.4f}s")

    def monitor_loop(self):
        rate = rospy.Rate(50)  # 50Hz 감시 루프
        while not rospy.is_shutdown():

            # heartbeat publish
            self.heartbeat_msg.data = self.heartbeat_value
            self.heartbeat_pub.publish(self.heartbeat_msg)

            now = time.time()

            # ===== rosserial 전체 상태 체크 =====
            all_dead = True
            now = time.time()

            # potentiometer 체크
            if self.last_time_pot is not None:
                if (now - self.last_time_pot) <= self.timeout:
                    all_dead = False

            # ultrasonic 체크
            for state in self.ultra_topics.values():
                if state["last_time"] is not None:
                    if (now - state["last_time"]) <= self.timeout:
                        all_dead = False

            # 모두 안 들어왔을 때
            if all_dead:
                self.rosserial_msg.data = 1
                self.rosserial_pub.publish(self.rosserial_msg)
                rospy.logwarn_throttle(1, "[ROSERIAL ERROR] All sensors stopped!")
            else:
                self.rosserial_msg.data = 0
                self.rosserial_pub.publish(self.rosserial_msg)


            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("pot_monitor")
    monitor = Rosserial_checker()
    monitor.monitor_loop()

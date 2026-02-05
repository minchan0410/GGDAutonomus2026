#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16
import time

class Rosserial_checker:
    def __init__(self):
        self.timeout = 0.5   # 타임아웃 여유를 조금 늘림 (0.2 -> 0.5 권장)

        # --- potentiometer ---
        self.last_time_pot = None
        self.msg_count_pot = 0
        self.start_time_pot = time.time()
        rospy.Subscriber("/potentiometer", Int16, self.cb_pot)

        # --- heartbeat ---
        self.heartbeat_pub = rospy.Publisher("/heart_beat", Int16, queue_size=1)
        self.heartbeat_msg = Int16()
        self.heartbeat_value = 1

        # ---rosserial check ---
        self.rosserial_pub = rospy.Publisher("/rosserial_check", Int16, queue_size=1)
        self.rosserial_msg = Int16()

    # potentiometer callback
    def cb_pot(self, msg):
        now = time.time()
        if self.last_time_pot is None:
            self.last_time_pot = now
            return
        
        # dt 계산 및 갱신
        # dt = now - self.last_time_pot  # 필요시 주석 해제하여 사용
        self.last_time_pot = now
        self.msg_count_pot += 1
        
        # 개별 로그는 끔 (전체 상태 로그만 보기 위해)
        # rospy.loginfo_throttle(0.5, f"[POT] dt: {dt:.4f}s")


    def monitor_loop(self):
        # 감시 주기는 10Hz~20Hz면 충분합니다 (50Hz는 단순 감시용으로 좀 빠름)
        rate = rospy.Rate(50) 
        
        while not rospy.is_shutdown():

            # heartbeat publish
            self.heartbeat_msg.data = self.heartbeat_value
            self.heartbeat_pub.publish(self.heartbeat_msg)

            now = time.time()

            # ===== rosserial 전체 상태 체크 =====
            # 기본적으로 "모두 죽었다(True)"라고 가정하고, 하나라도 살아있으면 "False"로 바꿈
            all_dead = True
            
            # potentiometer 체크
            if self.last_time_pot is not None:
                if (now - self.last_time_pot) <= self.timeout:
                    all_dead = False

            # 결과에 따른 로그 출력 (1초마다 출력)
            if all_dead:
                self.rosserial_msg.data = 1
                self.rosserial_pub.publish(self.rosserial_msg)
                # 에러 상황: 빨간색/경고 로그
                rospy.logerr_throttle(1, "[ROS_SERIAL] ERROR") 
            else:
                self.rosserial_msg.data = 0
                self.rosserial_pub.publish(self.rosserial_msg)
                # 정상 상황: 일반 정보 로그
                rospy.loginfo_throttle(1, "[ROS_SERIAL] OK")

            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("rosserial_checker")
    monitor = Rosserial_checker()
    monitor.monitor_loop()
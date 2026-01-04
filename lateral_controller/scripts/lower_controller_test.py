#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Int16

class Test:
    def __init__(self):
        self.motor_cmd_steer_pub = rospy.Publisher(
            "/des_steer", Int16, queue_size=1
        )

        self.rate = rospy.Rate(20)  # 20 Hz
        self.start_time = rospy.Time.now().to_sec()

        self.run()

    def run(self):
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.start_time

            # sin 파 생성: -15 ~ +15
            steer_value = int(22 * math.sin(2 * math.pi * 0.2 * t))
            # ↑ 0.2 Hz = 5초에 한 번 왕복 (원하면 바꿔도 됨)

            msg = Int16()
            msg.data = steer_value

            self.motor_cmd_steer_pub.publish(msg)
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("test")
    Test()

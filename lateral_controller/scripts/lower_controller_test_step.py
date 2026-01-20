#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16

AMP = 20          # deg (원하면 15 등으로 바꿔)
PERIOD = 3.0      # seconds (5초 주기)
HZ = 20           # publish rate

class TestSquare:
    def __init__(self):
        self.pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.rate = rospy.Rate(HZ)
        self.t0 = rospy.Time.now().to_sec()

    def run(self):
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.t0

            # 5초 주기 사각파: 앞 2.5초는 +AMP, 뒤 2.5초는 -AMP
            phase = (t % PERIOD)
            steer_value = AMP if phase < (PERIOD / 2.0) else -AMP

            self.pub.publish(Int16(int(steer_value)))
            self.rate.sleep()

if __name__ == "__main__":
    rospy.init_node("test_square")
    TestSquare().run()

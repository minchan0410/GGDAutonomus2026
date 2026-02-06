#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32, Int16

POT_LEFT_MAX =              588
POT_RIGHT_MAX =             431  # potentiometer
POT_TOTAL_RANGE_DEGREE =    270
POT_CENTER = (POT_LEFT_MAX + POT_RIGHT_MAX) / 2.0

GT_LEFT_MAX  =  22.5
GT_RIGHT_MAX = -22.5  # degree

Kp, Ki, Kd, u_max = 14.0, 0.0, 1.0, 255
MARGIN = 1.5  # degree


class PID:
    def __init__(self, Kp, Ki, Kd, u_max):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_max = u_max
        self.integral = 0.0
        self.e_prev = 0.0
        self.d_filt = 0.0
        self.alpha = 0.9

    def update(self, e, dt):
        p = self.Kp * e

        self.integral += e * dt
        i = self.Ki * self.integral

        de = (e - self.e_prev) / dt if dt > 1e-6 else 0.0
        self.d_filt = self.alpha * self.d_filt + (1 - self.alpha) * de
        d = self.Kd * self.d_filt

        u_unsat = p + i + d
        u = max(min(u_unsat, self.u_max), -self.u_max)

        if u != u_unsat:
            self.integral -= e * dt

        self.e_prev = e
        return u


class LowerController:
    def __init__(self):
        self.des_steer = 0.0
        self.cur_steer = 0.0
        self.pid = PID(Kp, Ki, Kd, u_max)

        rospy.Subscriber("/des_steer", Int16, self.cb_desired_steer, queue_size=1)
        rospy.Subscriber("/potentiometer", Int16, self.cb_potentiometer, queue_size=1)

        # motor command
        self.motor_cmd_steer_pub = rospy.Publisher("/motor_cmd_steer", Int16, queue_size=1)

        # NEW: publish desired/cur steer + pwm
        self.des_steer_pub = rospy.Publisher("/des_steer_deg", Float32, queue_size=1)
        self.cur_steer_pub = rospy.Publisher("/cur_steer_deg", Float32, queue_size=1)
        self.pwm_pub       = rospy.Publisher("/motor_cmd_steer", Int16, queue_size=1)

        self.last = rospy.Time.now()
        hz = 20.0
        rospy.Timer(rospy.Duration(1.0 / hz), self.on_timer)

    def cb_desired_steer(self, msg):
        # 기존 코드 유지: 부호 반전 포함
        self.des_steer = max(min(float(msg.data), GT_LEFT_MAX), GT_RIGHT_MAX)

    def cb_potentiometer(self, msg):
        self.cur_steer = self.pot2deg(msg.data)

    def on_timer(self, _):
        now = rospy.Time.now()
        dt = (now - self.last).to_sec()
        self.last = now

        e = self.des_steer - self.cur_steer
        u = self.pid.update(e, dt)

        if self.out_of_range():
            u = 0

        u_i16 = Int16(int(u))

        # motor command publish
        self.motor_cmd_steer_pub.publish(u_i16)

        # NEW publish
        self.des_steer_pub.publish(Float32(self.des_steer))
        self.cur_steer_pub.publish(Float32(self.cur_steer))
        self.pwm_pub.publish(u_i16)

    def pot2deg(self, pot_val):
        return (float(pot_val) - POT_CENTER) * POT_TOTAL_RANGE_DEGREE / 1023.0

    def out_of_range(self):
        return (not (GT_RIGHT_MAX + MARGIN < self.cur_steer < GT_LEFT_MAX - MARGIN)) and \
               (not (GT_RIGHT_MAX < self.des_steer < GT_LEFT_MAX))

if __name__ == "__main__":
    rospy.init_node("lateral_lower_controller")
    LowerController()
    rospy.spin()

import rospy
from std_msgs.msg import Float32, Int16

POT_LEFT_MAX =              590
POT_RIGHT_MAX =             420   # potentiometer
POT_TOTAL_RANGE_DEGREE =    270
POT_CENTER = int((POT_LEFT_MAX + POT_RIGHT_MAX)/2)
GT_LEFT_MAX =   22.5
GT_RIGHT_MAX = -22.5  # degree
Kp, Ki, Kd, u_max = 2.0, 1.0, 0.1, 255
MARGIN = 1.5          # degree

class PID:
    def __init__(self, Kp, Ki, Kd, u_max):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_max = u_max
        self.integral = 0.0
        self.e_prev = 0.0
        self.d_filt = 0.0
        self.alpha = 0.9  # derivative low-pass

    def update(self, e, dt):
        p = self.Kp * e

        # conditional integrator (anti-windup)
        self.integral += e * dt
        i = self.Ki * self.integral

        de = (e - self.e_prev) / dt if dt > 1e-6 else 0.0
        self.d_filt = self.alpha * self.d_filt + (1 - self.alpha) * de
        d = self.Kd * self.d_filt

        u_unsat = p + i + d
        u = max(min(u_unsat, self.u_max), -self.u_max)

        # anti-windup clamp
        if u != u_unsat:
            self.integral -= e * dt

        self.e_prev = e
        return u

class LowerController:
    def __init__(self):
        self.des_steer = 0.0
        self.cur_steer = 0.0
        self.pid = PID(Kp, Ki, Kd, u_max)

        rospy.Subscriber("/des_steer", Int16, self.cb_desired_steer)
        rospy.Subscriber("/potentiometer", Int16, self.cb_potentiometer)
        
        self.motor_cmd_steer_pub = rospy.Publisher("/motor_cmd_steer", Int16, queue_size=1)

        self.last = rospy.Time.now()
        hz = 50.0
        rospy.Timer(rospy.Duration(1.0/hz), self.on_timer)

    def cb_desired_steer(self, msg): self.des_steer = max(min(msg.data, GT_LEFT_MAX), GT_RIGHT_MAX)
        
    def cb_potentiometer(self, msg): self.cur_steer = self.pot2deg(msg.data)

    def on_timer(self, _):
        now = rospy.Time.now()
        dt = (now - self.last).to_sec()
        self.last = now

        e = self.des_steer - self.cur_steer
        u = self.pid.update(e, dt)
        
        if self.out_of_range():
            u = 0
            
        self.motor_cmd_steer_pub.publish(Int16(u))

    def pot2deg(self, pot_val):
        return max(min(pot_val, GT_LEFT_MAX), GT_RIGHT_MAX)

    def out_of_range(self):
        return not GT_RIGHT_MAX + MARGIN < self.cur_steer < GT_LEFT_MAX - MARGIN and not GT_RIGHT_MAX < self.des_steer < GT_LEFT_MAX
    
if __name__ == "__main__":
    rospy.init_node("lateral_lower_controller")
    LowerController()
    rospy.spin()

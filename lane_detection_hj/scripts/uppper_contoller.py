import rospy
from std_msgs.msg import Float32, Int16
import numpy as np
import matplotlib.pyplot as plt

class Upper_controller:

    def __init__(self):
        rospy.init_node('upper_controller', anonymous=True)

        rospy.Subscriber("/heading_error", Float32, self.heading_callback)
        rospy.Subscriber("/lateral_error", Float32, self.lateral_callback)

        self.steering_pub = rospy.Publisher("/des_steer", Int16, queue_size=1)
        self.speed_pub    = rospy.Publisher("/motor_cmd_long", Int16, queue_size=1)

        self.velocity = 0
        self.heading_error = 0
        self.lateral_error = 0
        self.is_heading_error = False
        self.is_lateral_error = False

        # ====== logging for plot ======
        self.t0 = rospy.get_time()
        self.t_hist = []
        self.lat_hist = []
        self.head_hist = []
        self.steer_hist = []
        self.vel_hist = []

        rospy.on_shutdown(self.on_shutdown)

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            is_ready = self.is_heading_error and self.is_lateral_error

            if is_ready:
                self.velocity = 150
                steering = self.stanley_lane()

                self.steering_pub.publish(Int16(steering))
                self.speed_pub.publish(Int16(self.velocity))

                # ---- append logs ----
                t = rospy.get_time() - self.t0
                self.t_hist.append(t)
                self.lat_hist.append(self.lateral_error)
                self.head_hist.append(self.heading_error)
                self.steer_hist.append(steering)
                self.vel_hist.append(self.velocity)

                print(f"steering:   {steering}")
                print(f"velocity:   {self.velocity}")
                print("=================================")
            else:
                print("Error")

            rate.sleep()

    def heading_callback(self, msg):
        self.is_heading_error = True
        self.heading_error = -msg.data

    def lateral_callback(self, msg):
        self.is_lateral_error = True
        self.lateral_error = msg.data

    def stanley_lane(self):
        k_s = 0.15
        GAIN_pi_t = 1

        x_t = self.lateral_error
        pi_t = self.heading_error
        pi_t = (pi_t + 180) % 360 - 180  # normalize to [-180, 180)

        steering = GAIN_pi_t * pi_t + np.degrees(np.arctan(k_s * x_t / max(self.velocity, 1e-6)))

        rospy.logwarn(f'\n x_t : {x_t} \n pi_t : {pi_t} \n')

        return int(steering)

    def on_shutdown(self):
        if len(self.t_hist) < 2:
            return

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))

        axes[0].plot(self.t_hist, self.lat_hist, color="tab:red")
        axes[0].set_ylabel("lateral error")
        axes[0].set_title("Lateral Error")
        axes[0].grid(True)

        axes[1].plot(self.t_hist, self.head_hist, color="tab:blue")
        axes[1].set_ylabel("heading error")
        axes[1].set_title("Heading Error")
        axes[1].grid(True)

        axes[2].plot(self.t_hist, self.steer_hist, color="tab:green")
        axes[2].set_ylabel("steering cmd")
        axes[2].set_title("Steering (Stanley Output)")
        axes[2].grid(True)

        axes[3].plot(self.t_hist, self.vel_hist, color="tab:purple")
        axes[3].set_xlabel("time (s)")
        axes[3].set_ylabel("speed cmd")
        axes[3].set_title("Speed Command")
        axes[3].grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        test_track = Upper_controller()
    except rospy.ROSInterruptException:
        pass

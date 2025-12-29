#!/usr/bin/env python3
import time
import rospy
import serial


# 좌우측 모터 5초간 70으로 구동 후 정지 테스트 코드

def main():
    rospy.init_node("lr_pwm_test_sender", anonymous=True)

    port = rospy.get_param("~port", "/dev/ttyACM0")
    baud = int(rospy.get_param("~baud", 115200))
    rate_hz = float(rospy.get_param("~rate", 20.0))

    pwm = int(rospy.get_param("~pwm", 70))
    duration = float(rospy.get_param("~duration", 5.0))

    ser = serial.Serial(port, baud, timeout=0.05)
    time.sleep(2.0)  # 아두이노 리셋/부팅 대기

    rate = rospy.Rate(rate_hz)
    t_end = time.time() + duration

    try:
        # 5초 동안 좌/우 PWM=70 전송
        while (not rospy.is_shutdown()) and (time.time() < t_end):
            ser.write(f"U {pwm} {pwm}\n".encode())
            ser.flush()
            rate.sleep()
    finally:
        # 정지 명령 여러 번 보내서 확실히 멈춤
        for _ in range(5):
            ser.write(b"U 0 0\n")
            ser.flush()
            time.sleep(0.05)
        ser.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int16
from pynput import keyboard
import threading

# === 설정값 (튜닝 파라미터) ===
MAX_SPEED = 100           # 최대 속도 값
MAX_STEER = 30            # 최대 조향 값

# 가속/조향 증가량 (키 누를 때)
RAMP_STEP_SPEED = 50      # 가속 step
RAMP_STEP_STEER = 2.5       # 조향 step

# 감속/조향 복귀량 (키 뗐을 때) - 이 값을 조절하여 멈추는 속도를 제어하세요
DECAY_STEP_SPEED = 40     # 0으로 돌아오는 속도 (클수록 빨리 멈춤)
DECAY_STEP_STEER = 2.5      # 0으로 돌아오는 조향 속도 (클수록 빨리 중앙 정렬)

LOOP_RATE = 15            # 루프 주파수 (Hz)

class CustomTeleop:
    def __init__(self):
        rospy.init_node('custom_teleop_node', anonymous=True)

        # Publisher 설정
        self.pub_speed = rospy.Publisher('/motor_cmd_long', Int16, queue_size=1)
        self.pub_steer = rospy.Publisher('/des_steer', Int16, queue_size=1)

        # 현재 키 상태를 저장할 변수
        self.key_states = {
            'up': False,    # 직진
            'down': False,  # 후진
            'q': False,     # 좌회전
            'w': False      # 우회전
        }

        # 현재 값 (속도, 조향)
        self.current_speed = 0
        self.current_steer = 0

        # 주기적인 루프 실행을 위한 Rate 객체
        self.rate = rospy.Rate(LOOP_RATE)

        print("=== Custom Teleop Started ===")
        print(f"  Speed Accel: {RAMP_STEP_SPEED}, Decel: {DECAY_STEP_SPEED}")
        print(f"  Steer Accel: {RAMP_STEP_STEER}, Return: {DECAY_STEP_STEER}")
        print("-----------------------------")
        print(" Controls:")
        print("  - Arrow Up   : 직진")
        print("  - Arrow Down : 후진")
        print("  - q          : 좌회전")
        print("  - w          : 우회전")
        print("  - No Key     : 자연스러운 감속/중립 복귀")
        print("=============================")

    def on_press(self, key):
        try:
            if key == keyboard.Key.up:
                self.key_states['up'] = True
            elif key == keyboard.Key.down:
                self.key_states['down'] = True
            elif hasattr(key, 'char'):
                if key.char == 'q':
                    self.key_states['q'] = True
                elif key.char == 'w':
                    self.key_states['w'] = True
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.up:
                self.key_states['up'] = False
            elif key == keyboard.Key.down:
                self.key_states['down'] = False
            elif hasattr(key, 'char'):
                if key.char == 'q':
                    self.key_states['q'] = False
                elif key.char == 'w':
                    self.key_states['w'] = False
            
            if key == keyboard.Key.esc:
                return False
        except AttributeError:
            pass

    def update_values(self):
        """키 상태에 따라 값을 변경 (가속/감속 로직 적용)"""
        
        # --- Speed 계산 ---
        if self.key_states['up'] and not self.key_states['down']:
            # 가속 (전진)
            self.current_speed = min(self.current_speed + RAMP_STEP_SPEED, MAX_SPEED)
        elif self.key_states['down'] and not self.key_states['up']:
            # 가속 (후진)
            self.current_speed = max(self.current_speed - RAMP_STEP_SPEED, -MAX_SPEED)
        else:
            # 키를 뗐을 때: 0으로 자연스럽게 복귀 (Decay)
            if self.current_speed > 0:
                self.current_speed = max(0, self.current_speed - DECAY_STEP_SPEED)
            elif self.current_speed < 0:
                self.current_speed = min(0, self.current_speed + DECAY_STEP_SPEED)

        # --- Steering 계산 ---
        if self.key_states['w'] and not self.key_states['q']:
            # 우회전
            self.current_steer = min(self.current_steer + RAMP_STEP_STEER, MAX_STEER)
        elif self.key_states['q'] and not self.key_states['w']:
            # 좌회전
            self.current_steer = max(self.current_steer - RAMP_STEP_STEER, -MAX_STEER)
        else:
            # 키를 뗐을 때: 0으로 자연스럽게 복귀 (Decay)
            if self.current_steer > 0:
                self.current_steer = max(0, self.current_steer - DECAY_STEP_STEER)
            elif self.current_steer < 0:
                self.current_steer = min(0, self.current_steer + DECAY_STEP_STEER)

    def run(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        while not rospy.is_shutdown():
            self.update_values()

            speed_msg = Int16()
            speed_msg.data = int(self.current_speed)
            
            steer_msg = Int16()
            steer_msg.data = -int(self.current_steer)

            self.pub_speed.publish(speed_msg)
            self.pub_steer.publish(steer_msg)
            
            # 디버깅: 현재 값 출력 (필요시 주석 해제)
            # print(f"Speed: {self.current_speed}, Steer: {self.current_steer}")

            self.rate.sleep()
        
        listener.stop()

if __name__ == '__main__':
    try:
        teleop = CustomTeleop()
        teleop.run()
    except rospy.ROSInterruptException:
        pass
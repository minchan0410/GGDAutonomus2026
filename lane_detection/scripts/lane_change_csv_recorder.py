#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import os
from datetime import datetime
from std_msgs.msg import Float32

class LaneChangeCSVRecorder:
    def __init__(self):
        rospy.init_node('lane_change_csv_recorder', anonymous=True)
        
        # CSV 파일 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_dir = os.path.expanduser('~/lane_change_logs')
        os.makedirs(self.csv_dir, exist_ok=True)
        self.csv_file = os.path.join(self.csv_dir, f'lane_change_{timestamp}.csv')
        
        # CSV 파일 열기 (계속 열어두기)
        self.csv_f = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_f)
        self.csv_writer.writerow(['timestamp', 'left_x', 'right_x'])
        self.csv_f.flush()  # 헤더 즉시 저장
        
        rospy.loginfo(f'CSV file created: {self.csv_file}')
        
        # 데이터 저장소
        self.left_x = None
        self.right_x = None
        self.last_log_time = rospy.Time.now()
        self.log_interval = 0.05  # 50ms마다 저장
        
        # 토픽 구독
        rospy.Subscriber('lane_change/left/x', Float32, self.left_callback)
        rospy.Subscriber('lane_change/right/x', Float32, self.right_callback)
        
        # 타이머 (정기적으로 CSV에 기록)
        rospy.Timer(rospy.Duration(self.log_interval), self.timer_callback)
        
        rospy.loginfo('lane_change_csv_recorder started')
    
    def left_callback(self, msg):
        self.left_x = msg.data
    
    def right_callback(self, msg):
        self.right_x = msg.data
    
    def timer_callback(self, event):
        current_time = rospy.Time.now()
        
        # 데이터가 있으면 CSV에 기록
        if self.left_x is not None or self.right_x is not None:
            timestamp = current_time.to_sec()
            self.csv_writer.writerow([
                timestamp,
                self.left_x if self.left_x is not None else '',
                self.right_x if self.right_x is not None else ''
            ])
            self.csv_f.flush()  # 버퍼 즉시 디스크에 저장


if __name__ == '__main__':
    try:
        recorder = LaneChangeCSVRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        recorder.csv_f.close()
    except KeyboardInterrupt:
        rospy.loginfo('lane_change_csv_recorder stopped')
        recorder.csv_f.close()

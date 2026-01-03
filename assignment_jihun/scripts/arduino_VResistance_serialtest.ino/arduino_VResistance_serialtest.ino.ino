// Arduino + rosserial: Potentiometer publish

#include <ros.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Int32.h>

// ===== ROS =====
ros::NodeHandle nh;

std_msgs::Int16 pot_msg;
ros::Publisher pot_pub("potentiometer", &pot_msg);

// std_msgs::Int32 ultra_msg;
// ros::Publisher ultra_pub("ultrasonic_mm", &ultra_msg);

// potentiometer
const int analogPin = A1;

// // ultrasonic
// const int trig1 = 3;
// const int echo1 = 4;


void setup()
{
  // // 초음파 핀 설정
  // pinMode(trig, OUTPUT);
  // pinMode(echo, INPUT);

  nh.initNode();

  nh.advertise(pot_pub);
  // nh.advertise(ultra_pub);
}

void loop()
{
  int pot_val = analogRead(analogPin);
  pot_msg.data = (int16_t)pot_val;
  pot_pub.publish(&pot_msg);

  // 2) 초음파 거리 읽기 (mm)
  //long dist_mm = ultrasonic_distance(trig, echo);
  //ultra_msg.data = (int32_t)dist_mm;   // 실패 시 -1
  //ultra_pub.publish(&ultra_msg);

  // rosserial 처리
  nh.spinOnce();

  // 대략 20Hz
  delay(50);
}

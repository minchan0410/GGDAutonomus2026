#include <ros.h>
#include <std_msgs/Int16.h>

ros::NodeHandle nh;

/* ===================== Pins (placeholder) ===================== */
const int analogPin = A1; //potentiometer

// motors (PWM) placeholder
const int ML_IN1 = 13;  //motor left in1
const int ML_IN2 = 12;  //motor left in2
const int MR_IN1 = 9;   //motor right in1
const int MR_IN2 = 8;   //motor right in2
const int MS_IN1 = 11;  //motor steer in1
const int MS_IN2 = 10;  //motor steer in2

/* ===================== ROS: Potentiometer ===================== */
std_msgs::Int16 pot_val;
ros::Publisher pot_pub("potentiometer", &pot_val);


/* ===================== ROS: Motor Subscribers ===================== */
volatile int16_t motor_pwm_long  = 0;
volatile int16_t motor_pwm_steer = 0;

static inline int16_t clamp_i16(int16_t x, int16_t lo, int16_t hi)
{
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/* ===================== (ADD) Link watchdog (HB based) ===================== */
/*
  - 외부 "serial_checker" 노드가 heartbeat 토픽을 publish (예: 50Hz)
  - 보드는 heartbeat 수신 시간(last_hb_ms) 기반으로 끊김 판단
  - HB가 timeout 프레임 이상 안 오면 즉시 E-STOP latch
  - HB가 연속 STABLE 프레임 이상 정상으로 들어오면 복구(=latch 해제)
*/
volatile uint32_t last_hb_ms = 0;

const uint16_t HB_HZ = 50;
const uint8_t HB_TIMEOUT_FRAMES = 4;
const uint8_t HB_STABLE_FRAMES  = 20;

const uint32_t HB_FRAME_MS   = 1000UL / HB_HZ;
const uint32_t HB_TIMEOUT_MS = (uint32_t)HB_TIMEOUT_FRAMES * HB_FRAME_MS;
const uint32_t HB_STABLE_MS  = (uint32_t)HB_STABLE_FRAMES * HB_FRAME_MS;

bool estop_latched = true;          // 부팅 직후 안전: E-STOP 상태에서 시작
uint32_t hb_stable_since_ms = 0;

void callback_hb(const std_msgs::Int16& msg)
{
  (void)msg;
  last_hb_ms = millis();
}

ros::Subscriber<std_msgs::Int16> hb_sub("heart_beat", callback_hb);

/* ===================== Motor cmd callbacks (unchanged) ===================== */
void callback_long(const std_msgs::Int16& msg)
{
  motor_pwm_long = clamp_i16(msg.data, -255, 255);
}

void callback_steer(const std_msgs::Int16& msg)
{
  motor_pwm_steer = clamp_i16(msg.data, -255, 255);
}

ros::Subscriber<std_msgs::Int16> motor_long_sub("motor_cmd_long", callback_long);
ros::Subscriber<std_msgs::Int16> motor_steer_sub("motor_cmd_steer", callback_steer);


/* ===================== Motor Output ===================== */
void apply_motor_long(int16_t cmd)
{
  cmd = clamp_i16(cmd, -255, 255);

  if (cmd >= 0) {
    analogWrite(ML_IN1, (uint8_t)cmd);
    analogWrite(ML_IN2, 0);

    analogWrite(MR_IN2, (uint8_t)cmd);
    analogWrite(MR_IN1, 0);
  } else {
    analogWrite(ML_IN1, 0);
    analogWrite(ML_IN2, (uint8_t)(-cmd));

    analogWrite(MR_IN2, 0);
    analogWrite(MR_IN1, (uint8_t)(-cmd));
  }
}

void apply_motor_steer(int16_t cmd)
{
  cmd = clamp_i16(cmd, -255, 255);

  if (cmd >= 0) {
    analogWrite(MS_IN1, (uint8_t)cmd);
    analogWrite(MS_IN2, 0);
  } else {
    analogWrite(MS_IN1, 0);
    analogWrite(MS_IN2, (uint8_t)(-cmd));
  }
}

void potentiometer()
{
  pot_val.data = (int16_t)analogRead(analogPin);
  pot_pub.publish(&pot_val);
}

/* ===================== Timing ===================== */
unsigned long last_20hz_ms = 0;
const unsigned long PERIOD_20HZ = 50;

/* ===================== Setup ===================== */
void setup()
{
  // motor pins
  pinMode(ML_IN1, OUTPUT); pinMode(ML_IN2, OUTPUT);
  pinMode(MR_IN1, OUTPUT); pinMode(MR_IN2, OUTPUT);
  pinMode(MS_IN1, OUTPUT); pinMode(MS_IN2, OUTPUT);

  // ROS
  nh.initNode();

  nh.advertise(pot_pub);

  nh.subscribe(motor_long_sub);
  nh.subscribe(motor_steer_sub);

  // (ADD) heartbeat subscriber
  nh.subscribe(hb_sub);

  // (ADD) start in E-STOP until heartbeat is stable
  last_hb_ms = 0;
  estop_latched = true;
  hb_stable_since_ms = 0;
}

/* ===================== Loop ===================== */
void loop()
{
  // 1) ROS comm always
  nh.spinOnce();

  // (ADD) link watchdog / E-STOP latch
  uint32_t now_ms = millis();
  bool hb_alive = (now_ms - last_hb_ms) <= HB_TIMEOUT_MS;

  if (!hb_alive) {
    // 끊김: 즉시 latch
    estop_latched = true;
    hb_stable_since_ms = 0;
  } else {
    // 살아있음: 연속 안정 시간 누적
    if (hb_stable_since_ms == 0) hb_stable_since_ms = now_ms;

    // 연속 200ms 안정이면 복구
    if (estop_latched && (now_ms - hb_stable_since_ms) >= HB_STABLE_MS) {
      estop_latched = false;
    }
  }

  // 2) motor output always fast (but gated by E-STOP)
  if (estop_latched) {
    apply_motor_long(0);
    apply_motor_steer(0);
  } else {
    apply_motor_long(motor_pwm_long);
    apply_motor_steer(motor_pwm_steer);
  }

  // 3) 20 Hz frame
  unsigned long now = millis();
  if (now - last_20hz_ms >= PERIOD_20HZ) {
    last_20hz_ms = now;

    potentiometer();
  }

  delay(1);
}

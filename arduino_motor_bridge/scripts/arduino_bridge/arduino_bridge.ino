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

// ultrasonic count
const uint8_t US_N = 6;

// trig / echo placeholder
const uint8_t trigPins[US_N] = {22, 23, 24, 25, 26, 27};
const uint8_t echoPins[US_N] = {2, 3, 18, 19, 20, 21}; // Mega external interrupt 가능 핀 가정


/* ===================== ROS: Potentiometer ===================== */
std_msgs::Int16 pot_val;
ros::Publisher pot_pub("potentiometer", &pot_val);

/* ===================== ROS: Ultrasonic  ===================== */
std_msgs::Int16 sonic1_dist;
std_msgs::Int16 sonic2_dist;
std_msgs::Int16 sonic3_dist;
std_msgs::Int16 sonic4_dist;
std_msgs::Int16 sonic5_dist;
std_msgs::Int16 sonic6_dist;

ros::Publisher sonic1_pub("ultrasonic1", &sonic1_dist);
ros::Publisher sonic2_pub("ultrasonic2", &sonic2_dist);
ros::Publisher sonic3_pub("ultrasonic3", &sonic3_dist);
ros::Publisher sonic4_pub("ultrasonic4", &sonic4_dist);
ros::Publisher sonic5_pub("ultrasonic5", &sonic5_dist);
ros::Publisher sonic6_pub("ultrasonic6", &sonic6_dist);


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

ros::Subscriber<std_msgs::Int16> hb_sub("heartbeat", callback_hb);

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

/* ===================== Ultrasonic ISR Data ===================== */

volatile uint32_t echo_rise_us[US_N];  // Low -> High로 올라가는 시각 배열(초음파 6개)
volatile uint32_t echo_fall_us[US_N];  // High -> Low로 내려가는 시각 배열(초음파 6개)
volatile bool     echo_done[US_N];     // 12ms 안에(2m 내에 존재하는 물체에 대해) 감지했는지 여부

int16_t ultrasonic_mm[US_N];           // 초음파 센서 6개 mm단위 거리 배열

/* ===================== ISR Functions ===================== */

void echo0_isr() { if (digitalRead(echoPins[0])) echo_rise_us[0] = micros();
                   else { echo_fall_us[0] = micros(); echo_done[0] = true; } }

void echo1_isr() { if (digitalRead(echoPins[1])) echo_rise_us[1] = micros();
                   else { echo_fall_us[1] = micros(); echo_done[1] = true; } }

void echo2_isr() { if (digitalRead(echoPins[2])) echo_rise_us[2] = micros();
                   else { echo_fall_us[2] = micros(); echo_done[2] = true; } }

void echo3_isr() { if (digitalRead(echoPins[3])) echo_rise_us[3] = micros();
                   else { echo_fall_us[3] = micros(); echo_done[3] = true; } }

void echo4_isr() { if (digitalRead(echoPins[4])) echo_rise_us[4] = micros();
                   else { echo_fall_us[4] = micros(); echo_done[4] = true; } }

void echo5_isr() { if (digitalRead(echoPins[5])) echo_rise_us[5] = micros();
                   else { echo_fall_us[5] = micros(); echo_done[5] = true; } }

/* ===================== Ultrasonic Groups ===================== */

// 두 그룹로 나눠서 12ms + 12ms = 24ms 내에 6개 측정 (20Hz 프레임 50ms 안에 충분)
const uint8_t GROUP_A[] = {0, 1, 2}; // ultrasonic1~3
const uint8_t GROUP_A_N = 3;

const uint8_t GROUP_B[] = {3, 4, 5}; // ultrasonic4~6
const uint8_t GROUP_B_N = 3;

const uint32_t US_WAIT_US = 12000;  // 12 ms (최대 측정거리 기준)

/* ===================== Ultrasonic Helpers ===================== */

void trigger_group(const uint8_t* idxs, uint8_t n)
{
  for (uint8_t i = 0; i < n; i++) {
    echo_done[idxs[i]] = false;
  }

  for (uint8_t i = 0; i < n; i++) {
    digitalWrite(trigPins[idxs[i]], HIGH);
  }

  delayMicroseconds(10);

  for (uint8_t i = 0; i < n; i++) {
    digitalWrite(trigPins[idxs[i]], LOW);
  }
}

void compute_group(const uint8_t* idxs, uint8_t n)
{
  for (uint8_t i = 0; i < n; i++) {
    uint8_t id = idxs[i];
    if (echo_done[id]) {
      uint32_t dt = echo_fall_us[id] - echo_rise_us[id];
      ultrasonic_mm[id] = (int16_t)(0.17f * dt); // mm
    } else {
      ultrasonic_mm[id] = -1; // timeout
    }
  }
}

/* ===================== (요청) Ultrasonic Frame Function ===================== */
void ultrasonic()
{
  // Group A
  trigger_group(GROUP_A, GROUP_A_N);
  delayMicroseconds(US_WAIT_US);
  compute_group(GROUP_A, GROUP_A_N);

  // Group B
  trigger_group(GROUP_B, GROUP_B_N);
  delayMicroseconds(US_WAIT_US);
  compute_group(GROUP_B, GROUP_B_N);

  // 각 토픽으로 publish (ultrasonic1~6)
  sonic1_dist.data = ultrasonic_mm[0]; sonic1_pub.publish(&sonic1_dist);
  sonic2_dist.data = ultrasonic_mm[1]; sonic2_pub.publish(&sonic2_dist);
  sonic3_dist.data = ultrasonic_mm[2]; sonic3_pub.publish(&sonic3_dist);
  sonic4_dist.data = ultrasonic_mm[3]; sonic4_pub.publish(&sonic4_dist);
  sonic5_dist.data = ultrasonic_mm[4]; sonic5_pub.publish(&sonic5_dist);
  sonic6_dist.data = ultrasonic_mm[5]; sonic6_pub.publish(&sonic6_dist);
}

/* ===================== Timing ===================== */
unsigned long last_20hz_ms = 0;
const unsigned long PERIOD_20HZ = 50;

/* ===================== Setup ===================== */
void setup()
{
  // ultrasonic pins
  for (uint8_t i = 0; i < US_N; i++) {
    pinMode(trigPins[i], OUTPUT);
    digitalWrite(trigPins[i], LOW);
    pinMode(echoPins[i], INPUT);

    echo_done[i] = false;
    ultrasonic_mm[i] = -1;
  }

  // attach interrupts
  attachInterrupt(digitalPinToInterrupt(echoPins[0]), echo0_isr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(echoPins[1]), echo1_isr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(echoPins[2]), echo2_isr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(echoPins[3]), echo3_isr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(echoPins[4]), echo4_isr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(echoPins[5]), echo5_isr, CHANGE);

  // motor pins
  pinMode(ML_IN1, OUTPUT); pinMode(ML_IN2, OUTPUT);
  pinMode(MR_IN1, OUTPUT); pinMode(MR_IN2, OUTPUT);
  pinMode(MS_IN1, OUTPUT); pinMode(MS_IN2, OUTPUT);

  // ROS
  nh.initNode();

  nh.advertise(pot_pub);

  nh.advertise(sonic1_pub);
  nh.advertise(sonic2_pub);
  nh.advertise(sonic3_pub);
  nh.advertise(sonic4_pub);
  nh.advertise(sonic5_pub);
  nh.advertise(sonic6_pub);

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
    ultrasonic();
  }

  delay(1);
}

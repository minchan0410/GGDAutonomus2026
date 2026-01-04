// ===== Arduino: motor_bridge_M70.ino =====
// 프로토콜(PC/ROS -> Arduino):  "M <V>\n"
// 예) "M 70\n"  -> 좌/우 모터 모두 PWM=70 (부호로 방향)
// 응답(Arduino -> PC): "M <V> Recived\n" 또는 "ERR\n"

#include <Arduino.h>

const int L_IN1 = 7;
const int L_IN2 = 8;
const int L_PWM = 5;

const int R_IN1 = 9;
const int R_IN2 = 10;
const int R_PWM = 6;

static char buf[64];
static uint8_t idx = 0;

unsigned long last_rx_ms = 0;
const unsigned long DEADMAN_MS = 200;

int clamp255(int v){
  if (v > 255) return 255;
  if (v < -255) return -255;
  return v;
}

void setMotor(int in1, int in2, int pwmPin, int cmd){
  cmd = clamp255(cmd);

  if (cmd > 0){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(pwmPin, cmd);
  } else if (cmd < 0){
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(pwmPin, -cmd);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(pwmPin, 0);
  }
}

void allStop(){
  setMotor(L_IN1, L_IN2, L_PWM, 0);
  setMotor(R_IN1, R_IN2, R_PWM, 0);
}

void setup(){
  pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT); pinMode(L_PWM, OUTPUT);
  pinMode(R_IN1, OUTPUT); pinMode(R_IN2, OUTPUT); pinMode(R_PWM, OUTPUT);

  Serial.begin(115200);
  allStop();
  last_rx_ms = millis();
}

void handleLine(const char* line){
  int V;

  // "M 70" 형태만 허용(완전 통일)
  if (sscanf(line, "M %d", &V) == 1){
    setMotor(L_IN1, L_IN2, L_PWM, V);
    setMotor(R_IN1, R_IN2, R_PWM, V);

    last_rx_ms = millis();

    // 수신 에코(디버깅)
    Serial.print("M ");
    Serial.print(V);
    Serial.println(" Recived");
  } else {
    Serial.println("ERR");
  }
}

void loop(){
  while (Serial.available() > 0){
    char c = (char)Serial.read();
    if (c == '\r') continue;

    if (c == '\n'){
      buf[idx] = '\0';
      if (idx > 0) handleLine(buf);
      idx = 0;
    } else {
      if (idx < sizeof(buf) - 1) buf[idx++] = c;
      else idx = 0;
    }
  }

  if (millis() - last_rx_ms > DEADMAN_MS){
    allStop();
  }
}

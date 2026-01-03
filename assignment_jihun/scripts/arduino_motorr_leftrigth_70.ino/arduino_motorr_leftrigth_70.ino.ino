#include "Arduino.h"

// ====== 핀 설정(예시) ======
// Mega PWM 가능한 핀: 2~13, 44~46
const int L_IN1 = 5;   // LEFT speed(PWM)
const int L_IN2 = 6;   // LEFT dir/other (가능하면 이것도 PWM 핀 권장)

const int R_IN1 = 9;   // RIGHT speed(PWM)
const int R_IN2 = 10;  // RIGHT dir/other (가능하면 이것도 PWM 핀 권장)


void motor_forward(int IN1, int IN2, int speed)
{
    analogWrite(IN1, speed);
    analogWrite(IN2, LOW);
}

void motor_backward(int IN1, int IN2, int speed)
{
    analogWrite(IN1, LOW);
    analogWrite(IN2, speed);
}

void motor_hold(int IN1, int IN2)
{
    analogWrite(IN1, LOW);
    analogWrite(IN2, LOW);
}

// ====== 편의 함수: 좌/우 모터 제어 ======
void left_forward(int spd)  { motor_forward(L_IN1, L_IN2, spd); }
void left_backward(int spd) { motor_backward(L_IN1, L_IN2, spd); }
void left_stop()            { motor_hold(L_IN1, L_IN2); }

void right_forward(int spd)  { motor_forward(R_IN1, R_IN2, spd); }
void right_backward(int spd) { motor_backward(R_IN1, R_IN2, spd); }
void right_stop()            { motor_hold(R_IN1, R_IN2); }

void setup()
{
    pinMode(L_IN1, OUTPUT);
    pinMode(L_IN2, OUTPUT);
    pinMode(R_IN1, OUTPUT);
    pinMode(R_IN2, OUTPUT);

    pinMode(POT_PIN, INPUT);

    Serial.begin(115200);
    left_stop();
    right_stop();
}

void loop()
{

    Serial.print("speed = ");
    Serial.println(spd);

    // 1) 둘 다 전진 2초
    left_forward(spd);
    right_forward(spd);
    delay(2000);

    // 2) 제자리 좌회전(왼쪽 후진, 오른쪽 전진) 1초
    left_backward(spd);
    right_forward(spd);
    delay(1000);

    // 3) 제자리 우회전(왼쪽 전진, 오른쪽 후진) 1초
    left_forward(spd);
    right_backward(spd);
    delay(1000);

    // 4) 정지 1초
    left_stop();
    right_stop();
    delay(1000);

    // 5) 둘 다 후진 2초
    left_backward(spd);
    right_backward(spd);
    delay(2000);

    // 6) 정지 1초
    left_stop();
    right_stop();
    delay(1000);
}

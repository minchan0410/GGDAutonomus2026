#include <Arduino.h>  // Arduino 기본 함수/타입(digitalWrite, analogWrite, millis 등) 사용을 위한 헤더

// =========================
// 1) 핀 매핑(배선 규약)
// =========================
// L = Left drive motor(왼쪽 구동 DC 모터)
// R = Right drive motor(오른쪽 구동 DC 모터)
// 각 모터는 H-Bridge(또는 유사 드라이버)로 제어한다고 가정하고,
//   - IN1, IN2 : 방향 제어용 디지털 신호(정/역방향)
//   - PWM      : 속도/출력 세기 제어용 PWM 신호(0~255)
// 를 사용한다.

const int L_IN1 = 7;   // 왼쪽 모터 방향 입력1(IN1) 연결된 Arduino 핀 번호
const int L_IN2 = 8;   // 왼쪽 모터 방향 입력2(IN2) 연결된 Arduino 핀 번호
const int L_PWM = 5;   // 왼쪽 모터 PWM 입력(또는 ENA)에 연결된 Arduino 핀 번호(PWM 가능한 핀)

const int R_IN1 = 9;   // 오른쪽 모터 IN1 핀
const int R_IN2 = 10;  // 오른쪽 모터 IN2 핀
const int R_PWM = 6;   // 오른쪽 모터 PWM 핀(PWM 가능한 핀)

// =========================
// 2) 시리얼 수신 버퍼(라인 기반 프로토콜 처리)
// =========================
// ROS(PC)에서 아두이노로 문자열을 보낼 때, 예를 들어 아래처럼 보낸다고 가정:
//   "U 70 70\n"
// 여기서 '\n' (줄바꿈)까지가 "명령 1개"의 경계다.
// 시리얼 데이터는 한 번에 통째로 들어오지 않고, 여러 번 나뉘어서 들어올 수 있다.
// 그래서 문자들을 버퍼(buf)에 하나씩 누적하고,
// '\n'을 만났을 때 버퍼에 모인 한 줄을 파싱(handleLine)한다.

static char buf[64];    // '\n' 만나기 전까지 문자를 쌓아둘 버퍼(최대 63글자 + 마지막 '\0')
static uint8_t idx = 0; // buf에 현재 몇 글자까지 쌓였는지 가리키는 인덱스

// =========================
// 3) 데드맨(Deadman) 타이머용 변수
// =========================
// 데드맨은 "통신이 끊겼을 때 자동으로 모터를 정지시키는 안전장치"다.
// last_rx_ms : 마지막으로 "정상 명령"을 수신한 시간을 millis() 기준으로 저장
// DEADMAN_MS : 이 시간(ms) 동안 명령이 안 오면 allStop() 호출해서 모터를 모두 정지
unsigned long last_rx_ms = 0;
const unsigned long DEADMAN_MS = 200; // 200ms 동안 명령이 없으면 자동 정지

// =========================
// 4) clamp 함수: 입력값 범위 제한(-255~255)
// =========================
// 아두이노 PWM은 analogWrite로 0~255 범위를 쓴다.
// 여기서는 부호를 이용해 방향(정/역)을 표현하기 때문에,
//   cmd > 0 : 정방향
//   cmd < 0 : 역방향
//   cmd = 0 : 정지
// 그리고 크기는 |cmd| (0~255)로 제한한다.
int clamp255(int v){
  if (v > 255) return 255;    // 255 초과면 255로 잘라냄
  if (v < -255) return -255;  // -255 미만이면 -255로 잘라냄
  return v;                   // 범위 안이면 그대로 반환
}

// =========================
// 5) setMotor: 모터 1개를 cmd 값대로 구동하는 함수
// =========================
// in1, in2 : 방향 핀
// pwmPin   : PWM 출력 핀
// cmd      : -255~255 (부호=방향, 절대값=세기)
// 동작 규칙:
//   cmd > 0 : IN1=HIGH, IN2=LOW  -> 정방향, PWM=cmd
//   cmd < 0 : IN1=LOW,  IN2=HIGH -> 역방향, PWM=|cmd|
//   cmd = 0 : IN1=LOW,  IN2=LOW  -> 정지(코스트 성격), PWM=0
// 주의: 드라이버에 따라 IN1/IN2 둘 다 LOW가 코스트/브레이크 중 무엇인지 다를 수 있다.
void setMotor(int in1, int in2, int pwmPin, int cmd){
  cmd = clamp255(cmd);  // 혹시 모를 범위 밖 입력 방지

  if (cmd > 0){
    // 정방향: IN1=1, IN2=0
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);

    // PWM 듀티를 cmd로 설정(0~255)
    analogWrite(pwmPin, cmd);

  } else if (cmd < 0){
    // 역방향: IN1=0, IN2=1
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);

    // cmd는 음수이므로 -cmd로 절대값을 PWM에 넣음
    analogWrite(pwmPin, -cmd);

  } else {
    // 정지: 방향핀 둘 다 LOW, PWM 0
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(pwmPin, 0);
  }
}

// =========================
// 6) allStop: 모든 모터를 정지시키는 함수(L/R만)
// =========================
// 데드맨 타임아웃 또는 시작 시 안전을 위해 호출.
// L/R 두 모터 모두 cmd=0으로 만들어서 정지시킨다.
void allStop(){
  setMotor(L_IN1, L_IN2, L_PWM, 0); // 왼쪽 구동 모터 정지
  setMotor(R_IN1, R_IN2, R_PWM, 0); // 오른쪽 구동 모터 정지
}

// =========================
// 7) setup: 전원 켰을 때 1회 실행되는 초기화
// =========================
void setup(){
  // 각 핀을 출력 모드로 설정 (모터드라이버에 신호를 "내보내는" 핀이므로 OUTPUT)
  pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT); pinMode(L_PWM, OUTPUT);
  pinMode(R_IN1, OUTPUT); pinMode(R_IN2, OUTPUT); pinMode(R_PWM, OUTPUT);

  // 시리얼 통신 시작 (PC/ROS 쪽도 115200으로 맞춰야 함)
  Serial.begin(115200);

  // 혹시 부팅 중 쓰레기 신호로 모터가 움직이는 걸 막기 위해 즉시 정지
  allStop();

  // "마지막 정상 명령 수신 시각"을 현재로 초기화
  last_rx_ms = millis();
}

// =========================
// 8) handleLine: 한 줄(명령 1개)을 파싱해서 모터에 적용 (L/R만)
// =========================
// line에는 예를 들어 "U 70 70" 같은 문자열이 들어온다고 가정.
// sscanf를 이용해 포맷이 맞는지 검사하고 2개 정수(L,R)를 뽑는다.
// 성공하면:
//   - 2개의 모터 명령 적용
//   - last_rx_ms 갱신(데드맨 타이머 리셋)
//   - "U L R Recived"를 시리얼로 응답(디버깅용)
// 실패하면:
//   - "ERR" 응답
void handleLine(const char* line){
  int L, R;

  // line이 "U %d %d" 형태로 들어왔는지 확인
  // 반환값이 2이면 정수 2개를 정상적으로 파싱했다는 뜻
  if (sscanf(line, "U %d %d", &L, &R) == 2){

    // 파싱된 명령을 각각의 모터에 적용
    setMotor(L_IN1, L_IN2, L_PWM, L); // 왼쪽 구동
    setMotor(R_IN1, R_IN2, R_PWM, R); // 오른쪽 구동

    // 정상 명령을 받았으니 데드맨 타이머 리셋
    last_rx_ms = millis();

    // 수신 에코(디버깅용)
    Serial.printf("U %d %d Recived\n", L, R);

  } else {
    // 포맷이 맞지 않거나 숫자 2개를 못 뽑으면 에러
    Serial.println("ERR");
  }
}

// =========================
// 9) loop: 계속 반복 실행되는 메인 루프
// =========================
// 역할 1) 시리얼에서 들어오는 문자를 읽어서 '\n' 단위로 한 줄을 완성
// 역할 2) 완성된 줄을 handleLine로 파싱/적용
// 역할 3) 일정 시간 명령이 없으면 데드맨으로 allStop
void loop(){

  // ---------- 9-1) 시리얼 수신 처리 ----------
  while (Serial.available() > 0){
    char c = (char)Serial.read();  // 시리얼에서 문자 1개 읽기

    // Windows 계열에서는 줄바꿈이 "\r\n"일 수 있음.
    // 여기서는 '\r'은 무시하고 '\n'만 줄의 끝으로 사용한다.
    if (c == '\r') continue;

    // 줄 끝('\n')을 만나면 지금까지 모은 buf를 한 줄로 확정
    if (c == '\n'){

      // C문자열로 만들기 위해 마지막에 널 종료문자 추가
      buf[idx] = '\0';

      // 빈 줄이 아니라면 파싱 시도
      if (idx > 0) handleLine(buf);

      // 다음 줄을 받기 위해 인덱스 초기화
      idx = 0;

    } else {
      // 줄 중간 문자라면 버퍼에 계속 누적
      if (idx < sizeof(buf) - 1) {
        buf[idx++] = c;
      } else {
        // 버퍼 오버플로우 방지: 너무 길면 해당 줄 포기
        idx = 0;
      }
    }
  }

  // ---------- 9-2) 데드맨(안전정지) ----------
  if (millis() - last_rx_ms > DEADMAN_MS){
    allStop();
  }
}

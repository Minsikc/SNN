/*  last edit date: 230918
 *   input format = F,0,0,PD,N5,15,200,20,F,1,1,1,1,10,1,1,0,0,0,0,0,0>
 *  
    
    int DFF1 = 31; //PA7 = D
    int DS = 42; //PA19
    int VH = 43; //PA20 SODR: VDDHALF2 (PCB board power pin), CODR: VDDHALF1 (PCB board power pin)

    int FB = 22; //PB26
    int CR = 52; //PB21
    int CON = 53; //PB14
    int HL_CHOP = 2; //PB25

    int WL_0 = 25; //PD0
    int WL_1 = 26; //PD1
    int WL_2 = 27; //PD2
    int WL_3 = 28; //PD3
    int WL_4 = 29; //

    int N1_0 = 51; //PC12
    int N1_1 = 50; //PC13
    int N1_2 = 49; //PC14
    int N1_3 = 48; //PC15
    int N1_4 = 47; //PC16
    /
    int N3_0 = 46; //PC17
    int N3_1 = 45; //PC18
    int N3_2 = 44; //PC19
    int N3_3 = 41; //PC9
    int N3_4 = 40; //PC8

    int N2_0 = 39; //PC7  
    int N2_1 = 38; //PC6
    int N2_2 = 37; //PC5
    int N2_3 = 36; //PC4
    int N2_3 = 35; //PC3

    int N4_0 = 34; //PC2
    int N4_1 = 33; //PC1
    int N4_2 = 7; //PC23
    int N4_3 = 6; //PC24
    int N4_4 = 5; //PC25

    int ADC_0 = A0;
    int ADC_1 = A1;
    int ADC_2 = A2;
    int ADC_3 = A3;
    int ADC_4 = A4;
*/
#include <ctype.h>

#define VAR_NUM 30 //Input 변수 개수
#define MAX_BIT_LENGTH 20
#define NUM_ROWS 5
#define NUM_COLS 5

// --- 1. 전역 변수 선언 ---
int row_num, col_num, set_num, update_num, read_period;
int pulse_width, pre_enable_time, post_enable_time, zero_time;
int read_time, read_set_time, read_delay;
String Direction, read_function, remainder_string, update_function;
//int ADC_VALUE_MONITOR[MAX];

int bit_length;
bool is_potentiation;

int N1_pulses[NUM_ROWS][MAX_BIT_LENGTH], N2_pulses[NUM_COLS][MAX_BIT_LENGTH];
int N3_pulses[NUM_ROWS][MAX_BIT_LENGTH], N4_pulses[NUM_COLS][MAX_BIT_LENGTH];

const int n1_pins[] = {12, 13, 14, 15, 16};
const int n2_pins[] = {7, 6, 5, 4, 3};
const int n3_pins[] = {17, 18, 19, 9, 8};
const int n4_pins[] = {2, 1, 23, 24, 25};
const int wl_pins[] = {0, 1, 2, 3, 6};

void setup() {
  pinMode(53, OUTPUT);
  pinMode(52, OUTPUT);
  pinMode(51, OUTPUT);
  pinMode(50, OUTPUT);
  pinMode(49, OUTPUT);
  pinMode(48, OUTPUT);
  pinMode(47, OUTPUT);
  pinMode(46, OUTPUT);
  pinMode(45, OUTPUT);
  pinMode(44, OUTPUT);
  pinMode(43, OUTPUT);
  pinMode(42, OUTPUT);
  pinMode(41, OUTPUT);
  pinMode(40, OUTPUT);
  pinMode(39, OUTPUT);
  pinMode(38, OUTPUT);
  pinMode(37, OUTPUT);
  pinMode(36, OUTPUT);
  pinMode(35, OUTPUT);
  pinMode(34, OUTPUT);
  pinMode(33, OUTPUT);
  pinMode(31, OUTPUT);
  pinMode(29, OUTPUT);
  pinMode(28, OUTPUT);
  pinMode(27, OUTPUT);
  pinMode(26, OUTPUT);
  pinMode(25, OUTPUT);
  pinMode(22, OUTPUT);
  pinMode(2, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(7, OUTPUT);
  ADC->ADC_MR |= 0x80; // Mode FREERUN
  ADC->ADC_CR = 2; // Start converter
  ADC->ADC_CHER = 0xF8; // Enabling All channels
  Serial.begin(115200);
  Serial.setTimeout(5000);
}

// --- 3. ✅ 간결해진 loop() 함수 ---
void loop() {
  if (Serial.available() > 0) {
    String input_string = Serial.readStringUntil('\n');
    parse_command(input_string); // 먼저 모든 것을 파싱하여 전역 변수 세팅
    execute_command();           // 세팅된 값에 따라 실행
  }
}


void parse_command(String input_str) {

  /* for debug
  Serial.println("--- Inside parse_command ---");
  Serial.print("Received Raw String: ");
  Serial.println(input_str);
  */
  // String 객체를 C언어 스타일의 char 배열로 변환 (메모리 효율적)
  char char_buf[input_str.length() + 1];
  input_str.toCharArray(char_buf, sizeof(char_buf));

  // strtok으로 첫 번째 토큰(Direction)을 가져옴
  Direction = strtok(char_buf, ",");
  row_num = atoi(strtok(NULL, ","));
  col_num = atoi(strtok(NULL, ","));
  update_function = strtok(NULL, ",");

  if (update_function == "STOCHASTIC_POTENTIATION" || update_function == "STOCHASTIC_DEPRESSION" || update_function == "STOCHASTIC_DNO_POTENTIATION" || update_function == "STOCHASTIC_DNO_DEPRESSION" || update_function == "DEBUG_STOCHASTIC" || update_function == "STOCHASTIC_POTENTIATION_NR" || update_function == "STOCHASTIC_DEPRESSION_NR" || update_function == "STOCHASTIC_DNO_POTENTIATION_NR" || update_function == "STOCHASTIC_DNO_DEPRESSION_NR") {
    read_function = strtok(NULL, ",");
    bit_length = atoi(strtok(NULL, ","));
    if (bit_length > MAX_BIT_LENGTH) bit_length = MAX_BIT_LENGTH;
    
    pulse_width = atoi(strtok(NULL, ","));
    pre_enable_time = atoi(strtok(NULL, ","));
    post_enable_time = atoi(strtok(NULL, ","));
    zero_time = atoi(strtok(NULL, ","));
    read_time = atoi(strtok(NULL, ","));
    read_delay = atoi(strtok(NULL, ","));

    is_potentiation = (update_function != "STOCHASTIC_DEPRESSION");
    int (*row_pulses)[MAX_BIT_LENGTH] = is_potentiation ? N1_pulses : N3_pulses;
    int (*col_pulses)[MAX_BIT_LENGTH] = is_potentiation ? N2_pulses : N4_pulses;
    
    for (int i = 0; i < NUM_ROWS; i++) {
      char* token = strtok(NULL, ",");
      /*
      Serial.print("Received Token for N1[");
      Serial.print(i);
      Serial.print("]: ");
      Serial.println(token ? token : "NULL"); // token이 NULL이면 "NULL"이라고 출력
      */
      for (int b = 0; b < bit_length; b++) row_pulses[i][b] = (token[b] == '1') ? 1 : 0;
    }
    
    for (int i = 0; i < NUM_COLS; i++) {
      char* token = strtok(NULL, ",");
      for (int b = 0; b < bit_length; b++) { col_pulses[i][b] = (token[b] == '1') ? 1 : 0;
      }

    }
    
    
  } else { // 기존 명령어 파싱
    read_function = strtok(NULL, ",");
    set_num = atoi(strtok(NULL, ","));
    update_num = atoi(strtok(NULL, ","));
    read_period = atoi(strtok(NULL, ","));
    remainder_string = strtok(NULL, ",");
    pulse_width = atoi(strtok(NULL, ","));
    pre_enable_time = atoi(strtok(NULL, ","));
    post_enable_time = atoi(strtok(NULL, ","));
    zero_time = atoi(strtok(NULL, ","));
    read_time = atoi(strtok(NULL, ","));
    read_set_time = atoi(strtok(NULL, ","));
    read_delay = atoi(strtok(NULL, ","));
  }
}


void execute_command() {
  /*
    Serial.println("--- PARSER DEBUG START ---");
    Serial.print("Parsed Direction: "); Serial.println(Direction);
    Serial.print("Parsed row_num: "); Serial.println(row_num);
    Serial.print("Parsed col_num: "); Serial.println(col_num);
    Serial.print("Parsed update_function: "); Serial.println(update_function);
    Serial.print("Parsed read_function: "); Serial.println(read_function);
    Serial.print("Parsed set_num: "); Serial.println(set_num);
    Serial.print("Parsed update_num: "); Serial.println(update_num);
    Serial.print("Parsed read_period: "); Serial.println(read_period);
    Serial.print("Parsed pulse_width: "); Serial.println(pulse_width);
    Serial.print("Parsed pre_enable_time: "); Serial.println(pre_enable_time);
    Serial.print("Parsed post_enable_time: "); Serial.println(post_enable_time);
    Serial.print("Parsed zero_time: "); Serial.println(zero_time);
    Serial.print("Parsed read_time: "); Serial.println(read_time);
    Serial.print("Parsed read_set_time: "); Serial.println(read_set_time);
    Serial.print("Parsed read_delay: "); Serial.println(read_delay);
    Serial.print("Parsed read_delay: "); Serial.println(read_delay);
    Serial.print("Parsed read_delay: "); Serial.println(read_delay);
  
    Serial.println("--- Parsed Pulse Streams (as integer arrays) ---");

    // N2_pulses 배열 출력
    Serial.println("N2 Pulses:");
    for (int c = 0; c < NUM_COLS; c++) {
      Serial.print("  N2["); Serial.print(c); Serial.print("]: ");
      for (int b = 0; b < bit_length; b++) {
        Serial.print(N2_pulses[c][b]);
      }
      Serial.println();
    }
    Serial.println("--- PARSER DEBUG END ---");
  */

  int cycle_num;
  if (remainder_string == "T") {
    if (update_num % read_period == 0) {
      cycle_num = read_period;
    }
    else {
      cycle_num = 1;
    }
  }
  else {
    cycle_num = 1;
  }

  //Program
  if (Direction == "F") { //Forward
    if (update_function == "P") { // Potentiation Only
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read 
      delay(10);
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
      delay(50);
      Serial.println("EOD>");
    }

    else if (update_function == "D") { // Depression Only
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
      delay(50);
      Serial.println("EOD>");
    }

    else if (update_function == "PD") { // Potentiation first and then Depression
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            //delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          for (int k = 0; k < update_num; k++) {
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            //delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
      delay(50);
      Serial.println("EOD>");
    }

    else if (update_function == "DP") { // Depression first and then Potentiation
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          for (int k = 0; k < update_num; k++) {
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
      delay(50);
      Serial.println("EOD>");
    }

    // PROG_NR: potentiation/depression with NO reads at all.
    // Every existing P/D opcode performs at least one array read, and
    // STOCHASTIC_POTENTIATION does two per call -- so priming a start state
    // with 12 commands costs ~5 s of reading against ~0.03 s of pulsing,
    // and charges the array 24 read-disturbances it did not need.
    // row_num/col_num=5 drives the whole array; direction is chosen by
    // read_function ("P" or "D") since that field is unused when nothing
    // is read.
    else if (update_function == "PROG_NR") {
      row_num = 5;
      col_num = 5;
      for (int k = 0; k < update_num; k++) {
        if (read_function == "D")
          Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
        else
          Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
      }
      Serial.println("EOD>");
    }

    else if (update_function == "HS_N24"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          for (int k = 0; k < update_num; k++) {
            N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }

    }

    // HS_SEQ: apply an ordered PAIR of half-selects, alternating, N times.
    // row_num carries the pair as two digits: tens = first, units = second
    // (1..4 = N1..N4, 0 = skip that slot).
    //   row_num=14 -> N1,N4,N1,N4,...   row_num=40 -> N4 alone
    //   row_num=0  -> no pulses at all (read-only control)
    // update_num = number of cycles; one cycle = first then second.
    // Emits ONE read at entry and nothing else, so the read cost is the same
    // for every pair and for the control.
    else if (update_function == "HS_SEQ") {
      int first  = row_num / 10;
      int second = row_num % 10;
      row_num = 5;   // N1/N3 select all rows
      col_num = 5;   // N2/N4 select all columns

      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
      Serial.println(">");

      for (int k = 0; k < update_num; k++) {
        if (first == 1)      N1_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
        else if (first == 2) N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
        else if (first == 3) N3_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
        else if (first == 4) N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);

        if (second == 1)      N1_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
        else if (second == 2) N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
        else if (second == 3) N3_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
        else if (second == 4) N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
      }

      Serial.println("EOD>");
    }


    else if (update_function == "HS_N1"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N1_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
    }

      else if (update_function == "HS_N2"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
    }


    else if (update_function == "HS_N2_Only"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
              }
        }
      }
      Serial.print("operation end");
    }

    else if (update_function == "HS_N4_Only"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
              }
        }
      }
      Serial.print("operation end");
    }

    else if (update_function == "Gsym_DPPD"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            
            //Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
              }  
          }
        }
      }

      
      else if (update_function == "Gsym_PD"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
                          
            //Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
              }  
          }
        }
      }

      else if (update_function == "Gsym_PPDD"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            
            //Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
              }  
          }
        }
      }


      
    
    else if (update_function == "HS_N23_Only"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N23_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
              }
        }
      }
      Serial.print("operation end");
    }

    else if (update_function == "HS_N14_Only"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N14_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
              }
        }
      }
      Serial.print("operation end");
    }

    // Reverse-order counterparts of HS_N14_Only / HS_N23_Only. Same pulse
    // budget and same overlap timing; only the rise/fall order is flipped.
    // HS_SEQ alternates a pair, so it fires N1->N4 and N4->N1 equally often
    // and an ordering asymmetry cancels out; these do not alternate, so they
    // isolate one ordering. Terminated with EOD> like HS_SEQ, so the host
    // reader stops immediately instead of waiting out the quiet window.
    else if (update_function == "HS_N41_Only"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N41_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
              }
        }
      }
      Serial.println("EOD>");
    }

    else if (update_function == "HS_N32_Only"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N32_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
              }
        }
      }
      Serial.println("EOD>");
    }


    else if (update_function == "HS_N4"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
    }

    else if (update_function == "HS_N3"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N3_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }
    }
    
    else if (update_function == "HS_N13"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N1_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          for (int k = 0; k < update_num; k++) {
            N3_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }

    }


    else if (update_function == "HS_N42"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          for (int k = 0; k < update_num; k++) {
            N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }

    }

    else if (update_function == "HS_relaxation"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N23_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          for (int k = 0; k < update_num; k++) {
            N14_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }

    }


    else if (update_function == "HS_relaxation_N23"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N23_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }

    }

    else if (update_function == "HS_relaxation_N14"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N14_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }

    }


    else if (update_function == "Reset"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            Reset_update(pulse_width, pre_enable_time, post_enable_time, zero_time);
            //delay(2); // For time difference between Update phase and Read phase
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
        }
      }

    }


    else if (update_function == "N13_cross"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N1_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            N3_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          
          }
        }
      }

      else if (update_function == "N24_cross"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          
          }
        }
      }

      else if (update_function == "N12_cross"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N1_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          
          }
        }
      }


      else if (update_function == "N34_cross"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N3_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          
          }
        }
      }

      else if (update_function == "N23_14_cross"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N23_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            N14_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          
          }
        }
      }

      else if (update_function == "N1_23_cross"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N1_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            N23_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          
          }
        }
      }

      else if (update_function == "N3_14_cross"){
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); // 첫 시작 점 read
      Serial.println(">");
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N3_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            N14_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            if ((k + 1) % read_period == j) {
              Serial.print(i + 1);
              Serial.print(",");
              Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
              Serial.println(">");
            }
          }
          
          }
        }
      }


      else if (update_function == "N2_HS_only"){
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N2_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            }
          
          }
        }
      }

      else if (update_function == "N4_HS_only"){
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N4_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            }
          
          }
        }
      }

      else if (update_function == "N23_HS_only"){
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N23_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            }
          
          }
        }
      }

      else if (update_function == "N14_HS_only"){
      for (int i = 0; i < set_num; i++) {
        for (int j = 0; j < cycle_num; j++) {
          for (int k = 0; k < update_num; k++) {
            N14_half_selected(pulse_width, pre_enable_time, post_enable_time, zero_time);
            }
          
          }
        }
      }

    
    else if (update_function == "R") { // Read operation Only
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
      Serial.println(">");
      Serial.println("EOD>");
    }
    else if (update_function == "GLOBAL_PD_SEQ_READ") {
      Serial.println("--- Global Update & Sequential Read Start ---");
      
      // 1. & 2. 모든 셀 동시 업데이트 (P/D)
      
      for (int k = 0; k < update_num; k++) {
        row_num = 5; col_num = 5;
        Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
        
        // read_period 마다 전체 어레이(5줄)를 순차적으로 측정
        if ((k + 1) % read_period == 0) {
          Read_all_rows_sequentially(read_function, read_time, read_set_time, read_delay);
        }
      }

      // --- Depression 사이클 + 주기적 측정 ---
      for (int k = 0; k < update_num; k++) {
        row_num = 5; col_num = 5;
        Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);

        if ((k + 1) % read_period == 0) {
          Read_all_rows_sequentially(read_function, read_time, read_set_time, read_delay);
        }
      }

      Serial.println("EOD>");
    }
    
    else if (update_function == "READ_ROW") {
      if (row_num >= 0 && row_num <= 4) {
        Serial.print("Row_");
        Serial.print(row_num);
        Serial.print(",");
        Read_operation_forward_6T(read_function, read_time, 0, read_delay);
        Serial.println(">");
      } else {
        Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      }
      Serial.println("EOD>");
    }
    
    else if (update_function == "STOCHASTIC_POTENTIATION") {
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      apply_potentiation_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      Serial.println("EOD>");
      }
    else if (update_function == "STOCHASTIC_DEPRESSION") {
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      apply_depression_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      Serial.println("EOD>");
     }
    else if (update_function == "STOCHASTIC_DNO_POTENTIATION") {
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      apply_DNO_potentiation_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      Serial.println("EOD>");
     }    
    else if (update_function == "STOCHASTIC_DNO_DEPRESSION") {
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      apply_DNO_depression_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Read_all_rows_sequentially(read_function, read_time, 0, read_delay);
      Serial.println("EOD>");
     }

    // ---- No-Read (_NR) stochastic updates -------------------------------
    // Identical pulse delivery to the four opcodes above -- same
    // apply_*_pulses call, same arguments -- but with NO pre/post read.
    //
    // Why: each read costs ~0.79% of the distance to the read attractor
    // (tau ~ 128 reads). A training epoch issues ~10.5 update commands, so
    // the built-in pre/post reads cost ~21 array reads per epoch and decay
    // the stored gradient by ~15% WITHIN one epoch. The host discards those
    // per-command reads anyway: it reads the accumulated gradient once, after
    // all updates are flushed. Dropping them leaves 2 reads per epoch (one
    // reference read after reset, one at the end) and ~1.6% decay.
    //
    // The host must therefore NOT expect Row_ data back from these -- only
    // the EOD> terminator. Use READ_ROW when a measurement is wanted.
    else if (update_function == "STOCHASTIC_POTENTIATION_NR") {
      apply_potentiation_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Serial.println("EOD>");
     }
    else if (update_function == "STOCHASTIC_DEPRESSION_NR") {
      apply_depression_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Serial.println("EOD>");
     }
    else if (update_function == "STOCHASTIC_DNO_POTENTIATION_NR") {
      apply_DNO_potentiation_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Serial.println("EOD>");
     }
    else if (update_function == "STOCHASTIC_DNO_DEPRESSION_NR") {
      apply_DNO_depression_pulses(bit_length, pulse_width, pre_enable_time, post_enable_time, zero_time);
      Serial.println("EOD>");
     }
    /*else if (update_function == "GND") {  // Set Capacitor Voltage to Neutral
        int n1;
        int n2; 
        
        if (row_num == 0) {
          n1 = (1 << 12);
        }
        else if (row_num == 1) {
          n1 = (1 << 13);
        }
        else if (row_num == 2) {
          n1 = (1 << 14);
        }
        else if (row_num == 3) {
          n1 = (1 << 15);
        }
        else if (row_num == 4) {
          n1 = (1 << 16);
        }
      
        if (col_num == 0) {
          n2 = (1 << 7);
        }
        else if (col_num == 1) {
          n2 = (1 << 6);
        }
        else if (col_num == 2) {
          n2 = (1 << 5);
        }
        else if (col_num == 3) {
          n2 = (1 << 4);
        }
        else if (col_num == 4) {
          n2 = (1 << 3);
        }

      PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
      PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
      PIOA->PIO_CODR = 1 << 20; //VH VDDHAFL1
      PIOC->PIO_SODR = n1; // N1 ON
      PIOC->PIO_SODR = n2; // N2 ON
      delay(extra1); //Ground Capacitor for extra1 milisec
      PIOC->PIO_CODR = 1 << 2; // N1 clear
      PIOC->PIO_CODR = 1 << 7; //N2 clear
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
      Serial.println(">");

      delay(50);
      Serial.println("EOD>");
    }*/

    /*else if (update_function == "PR") { // Positive Direction Retention Test
      int retentionIntv = extra1 * minutes;
      PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
      PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
      
      PIOC->PIO_SODR = 1 << 2; // N1 ON
      PIOC->PIO_SODR = 1 << 7; // N2 ON
      delay(20); //Ground Capacitor for extra1 milisec
      PIOC->PIO_CODR = 1 << 2; // N1 clear
      PIOC->PIO_CODR = 1 << 7; //N2 clear
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); //Read GND ADC points
      Serial.println(">");

      Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
      for (int j = 0; j < read_period; j++) {
        Serial.print("0,");
        Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); //t0 measure point
        Serial.println(">");
      }
      for (int i = 0; i < set_num; i++) {
        delay(retentionIntv);
        for (int k = 0; k < read_period; k++) {
          Serial.print(i + 1);
          Serial.print(",");
          Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
          Serial.println(">");
        }
      }
      delay(50);
      Serial.println("EOD>");
    }*/
/*      else if (update_function == "NR") { // Negative Direction Retention Test
      int retentionIntv = extra1 * minutes;
      PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
      PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
      PIOC->PIO_SODR = 1 << 2; // N1 ON
      PIOC->PIO_SODR = 1 << 7; // N2 ON
      delay(20); //Ground Capacitor for extra1 milisec
      PIOC->PIO_CODR = 1 << 2; // N1 clear
      PIOC->PIO_CODR = 1 << 7; //N2 clear
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); //Read GND ADC points
      Serial.println(">");

      Depression_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
      for (int j = 0; j < read_period; j++) {
        Serial.print("0,");
        Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); //t0 measure point
        Serial.println(">");
      }
      for (int i = 0; i < set_num; i++) {
        delay(retentionIntv);
        for (int k = 0; k < read_period; k++) {
          Serial.print(i + 1);
          Serial.print(",");
          Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
          Serial.println(">");
        }
      }
      delay(50);
      Serial.println("EOD>");
    }
    */
    /*else if (update_function == "PRC") { // Positive Direction N1 ON Retention Test
      int retentionIntv = extra1 * minutes;
      PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
      PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
      PIOC->PIO_SODR = 1 << 2; // N1 ON
      PIOC->PIO_SODR = 1 << 7; // N2 ON
      delay(20); //Ground Capacitor for extra1 milisec
      PIOC->PIO_CODR = 1 << 2; // N1 clear
      PIOC->PIO_CODR = 1 << 7; //N2 clear
      Serial.print("0,");
      Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); //Read GND ADC points
      Serial.println(">");

      Potentiation_6T(pulse_width, pre_enable_time, post_enable_time, zero_time);
      for (int j = 0; j < read_period; j++) {
        Serial.print("0,");
        Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay); //t0 measure point
        Serial.println(">");
      }
      for (int i = 0; i < set_num; i++) {
        delay(retentionIntv);
        for (int k = 0; k < read_period; k++) {
          Serial.print(i + 1);
          Serial.print(",");
          Read_operation_forward_6T(read_function, read_time, read_set_time, read_delay);
          Serial.println(">");
        }
      }
      delay(50);
      Serial.println("EOD>");
    }*/

    }

    else if (Direction == "B") { //Backward (Backpropagation)
    Serial.println("Not implemented yet!");
    //Read_operation_backward(read_function, WL_0, WL_1, WL_2, WL_3, WL_4)
    }
}
////////////////////////////////////////// UPDATE & READ function ///////////////////////////////////

void Potentiation_6T(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time) { // only 0 index, 1번만
  int n1; // "지역변수는 선언된 지역 내에서만 유효하기 때문에 선언된 지역이 다르면 이름이 같아도 문제가 되지 않는다." 
  int n2; 
  
  if (row_num == 0) {
    n1 = (1 << 12);
  }
  else if (row_num == 1) {
    n1 = (1 << 13);
  }
  else if (row_num == 2) {
    n1 = (1 << 14);
  }
  else if (row_num == 3) {
    n1 = (1 << 15);
  }
  else if (row_num == 4) {
    n1 = (1 << 16);
  }
  else if (row_num == 5){
    n1 = (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16);
  }

  if (col_num == 0) {
    n2 = (1 << 7);
  }
  else if (col_num == 1) {
    n2 = (1 << 6);
  }
  else if (col_num == 2) {
    n2 = (1 << 5);
  }
  else if (col_num == 3) {
    n2 = (1 << 4);
  }
  else if (col_num == 4) {
    n2 = (1 << 3);
  }
  else if (col_num == 5){
    n2 = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 3);
  }
  
  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n1; // N1 clear
  PIOC->PIO_CODR = n2; //// N2 clear --> for 010 like pulse input
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n1; // N1 set
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n2; // N2 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n2; // N2 clear
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n1; // N1 clear
  delayMicroseconds(zero_time);
}


void Reset_update(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time) { // only 0 index, 1번만
  int n1; // "지역변수는 선언된 지역 내에서만 유효하기 때문에 선언된 지역이 다르면 이름이 같아도 문제가 되지 않는다." 
  int n3; 
  
  if (row_num == 0) {
    n1 = (1 << 12);
    n3 = (1 << 17);
  }
  else if (row_num == 1) {
    n1 = (1 << 13);
    n3 = (1 << 18);
  }
  else if (row_num == 2) {
    n1 = (1 << 14);
    n3 = (1 << 19);
  }
  else if (row_num == 3) {
    n1 = (1 << 15);
    n3 = (1 << 9);
  }
  else if (row_num == 4) {
    n1 = (1 << 16);
    n3 = (1 << 8);
  }
  else if (row_num == 5){
    n1 = (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16);
    n3 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 9) | (1 << 8);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n1; // N1 clear
  PIOC->PIO_CODR = n3; //// N3 clear --> for 010 like pulse input
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n1; // N1 set
  PIOC->PIO_SODR = n3; // N3 set
  delayMicroseconds(pre_enable_time);
  delayMicroseconds(pulse_width);
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n1; // N1 clear
  PIOC->PIO_CODR = n3; // N3 clear
  delayMicroseconds(zero_time);
}

void Depression_6T(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time) {
  int n3;
  int n4; 
  
  if (row_num == 0) {
    n3 = (1 << 17);
  }
  else if (row_num == 1) {
    n3 = (1 << 18);
  }
  else if (row_num == 2) {
    n3 = (1 << 19);
  }
  else if (row_num == 3) {
    n3 = (1 << 9);
  }
  else if (row_num == 4) {
    n3 = (1 << 8);
  }
  else if ( row_num == 5){
    n3 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 9) | (1 << 8); // col_num -> row_num  
  }

  if (col_num == 0) {
    n4 = (1 << 2);
  }
  else if (col_num == 1) {
    n4 = (1 << 1);
  }
  else if (col_num == 2) {
    n4 = (1 << 23);
  }
  else if (col_num == 3) {
    n4 = (1 << 24);
  }
  else if (col_num == 4) {
    n4 = (1 << 25);
  }
  else if (col_num == 5){
    n4 = (1 << 2) | (1 << 1) | (1 << 23) | (1 << 24) | (1 << 25);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n3; // N3 clear
  PIOC->PIO_CODR = n4; //// N4 clear
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n3; // N3 set
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n4; // N4 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n4; // N4 clear
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n3; // N3 clear
  delayMicroseconds(zero_time);
}

void N1_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time){
  int n1;
  if (row_num == 0) {
    n1 = (1 << 12);
  }
  else if (row_num == 1) {
    n1 = (1 << 13);
  }
  else if (row_num == 2) {
    n1 = (1 << 14);
  }
  else if (row_num == 3) {
    n1 = (1 << 15);
  }
  else if (row_num == 4) {
    n1 = (1 << 16);
  }
  else if (row_num == 5){
    n1 = (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n1; // N1 clear
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n1; // N1 set
  delayMicroseconds(pre_enable_time);
  delayMicroseconds(pulse_width);
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n1; // N1 clear
  delayMicroseconds(zero_time);
}

void N2_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time){
  int n2; 
  
  if (col_num == 0) {
  n2 = (1 << 7);
  }
  else if (col_num == 1) {
    n2 = (1 << 6);
  }
  else if (col_num == 2) {
    n2 = (1 << 5);
  }
  else if (col_num == 3) {
    n2 = (1 << 4);
  }
  else if (col_num == 4) {
    n2 = (1 << 3);
  }
  else if (col_num == 5){
    n2 = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 3);
  }


  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n2; //// N2 clear --> for 010 like pulse input
  delayMicroseconds(zero_time);
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n2; // N2 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n2; // N2 clear
  delayMicroseconds(post_enable_time);
  delayMicroseconds(zero_time);
}

void N3_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time){
  int n3;
    
  if (row_num == 0) {
    n3 = (1 << 17);
  }
  else if (row_num == 1) {
    n3 = (1 << 18);
  }
  else if (row_num == 2) {
    n3 = (1 << 19);
  }
  else if (row_num == 3) {
    n3 = (1 << 9);
  }
  else if (row_num == 4) {
    n3 = (1 << 8);
  }
  else if (row_num == 5){
    n3 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 9) | (1 << 8);
  }
 
  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n3; // N3 clear
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n3; // N3 set
  delayMicroseconds(pre_enable_time);
  delayMicroseconds(pulse_width);
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n3; // N3 clear
  delayMicroseconds(zero_time); 
}

 void N4_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time){
  int n4;

  if (col_num == 0) {
    n4 = (1 << 2);
  }
  else if (col_num == 1) {
    n4 = (1 << 1);
  }
  else if (col_num == 2) {
    n4 = (1 << 23);
  }
  else if (col_num == 3) {
    n4 = (1 << 24);
  }
  else if (col_num == 4) {
    n4 = (1 << 25);
 }
  else if (col_num == 5){
    n4 = (1 << 2) | (1 << 1) | (1 << 23) | (1 << 24) | (1 << 25);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n4; //// N4 clear
  delayMicroseconds(zero_time);
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n4; // N4 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n4; // N4 clear
  delayMicroseconds(post_enable_time);
  delayMicroseconds(zero_time);
 }

void N23_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time) {
  int n3;
  int n2; 
  
  if (row_num == 0) {
    n3 = (1 << 17);
  }
  else if (row_num == 1) {
    n3 = (1 << 18);
  }
  else if (row_num == 2) {
    n3 = (1 << 19);
  }
  else if (row_num == 3) {
    n3 = (1 << 9);
  }
  else if (row_num == 4) {
    n3 = (1 << 8);
  }
  else if (row_num == 5){
    n3 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 9) | (1 << 8);
  }

  if (col_num == 0) {
  n2 = (1 << 7);
  }
  else if (col_num == 1) {
    n2 = (1 << 6);
  }
  else if (col_num == 2) {
    n2 = (1 << 5);
  }
  else if (col_num == 3) {
    n2 = (1 << 4);
  }
  else if (col_num == 4) {
    n2 = (1 << 3);
  }
  else if (col_num == 5){
    n2 = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 3);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n3; // N3 clear
  PIOC->PIO_CODR = n2; //// N2 clear
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n3; // N3 set
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n2; // N2 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n2; // N2 clear
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n3; // N3 clear
  delayMicroseconds(zero_time);
}

void N14_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time) {
  int n1;
  int n4;
  
  if (row_num == 0) {
    n1 = (1 << 12);
  }
  else if (row_num == 1) {
    n1 = (1 << 13);
  }
  else if (row_num == 2) {
    n1 = (1 << 14);
  }
  else if (row_num == 3) {
    n1 = (1 << 15);
  }
  else if (row_num == 4) {
    n1 = (1 << 16);
  }
  else if (row_num == 5){
    n1 = (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16);
  }

  if (col_num == 0) {
    n4 = (1 << 2);
  }
  else if (col_num == 1) {
    n4 = (1 << 1);
  }
  else if (col_num == 2) {
    n4 = (1 << 23);
  }
  else if (col_num == 3) {
    n4 = (1 << 24);
  }
  else if (col_num == 4) {
    n4 = (1 << 25);
 }
  else if (col_num == 5){
    n4 = (1 << 2) | (1 << 1) | (1 << 23) | (1 << 24) | (1 << 25);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)
  
  PIOC->PIO_CODR = n1; // N1 clear
  PIOC->PIO_CODR = n4; //// N4 clear
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n1; // N1 set
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n4; // N4 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n4; // N4 clear
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n1; // N1 clear
  delayMicroseconds(zero_time);
}

// Reverse of N14_half_selected: N4 rises first and falls last, so the overlap
// is entered N4->N1 instead of N1->N4. Same pin maps, same timing; only the
// SODR/CODR order differs, which is the whole point of the pair.
void N41_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time) {
  int n1;
  int n4;

  if (row_num == 0) {
    n1 = (1 << 12);
  }
  else if (row_num == 1) {
    n1 = (1 << 13);
  }
  else if (row_num == 2) {
    n1 = (1 << 14);
  }
  else if (row_num == 3) {
    n1 = (1 << 15);
  }
  else if (row_num == 4) {
    n1 = (1 << 16);
  }
  else if (row_num == 5){
    n1 = (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16);
  }

  if (col_num == 0) {
    n4 = (1 << 2);
  }
  else if (col_num == 1) {
    n4 = (1 << 1);
  }
  else if (col_num == 2) {
    n4 = (1 << 23);
  }
  else if (col_num == 3) {
    n4 = (1 << 24);
  }
  else if (col_num == 4) {
    n4 = (1 << 25);
  }
  else if (col_num == 5){
    n4 = (1 << 2) | (1 << 1) | (1 << 23) | (1 << 24) | (1 << 25);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)

  PIOC->PIO_CODR = n4; // N4 clear
  PIOC->PIO_CODR = n1; //// N1 clear
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n4; // N4 set
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n1; // N1 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n1; // N1 clear
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n4; // N4 clear
  delayMicroseconds(zero_time);
}

// Reverse of N23_half_selected: N2 rises first and falls last.
void N32_half_selected(int pulse_width, int pre_enable_time, int post_enable_time, int zero_time) {
  int n3;
  int n2;

  if (row_num == 0) {
    n3 = (1 << 17);
  }
  else if (row_num == 1) {
    n3 = (1 << 18);
  }
  else if (row_num == 2) {
    n3 = (1 << 19);
  }
  else if (row_num == 3) {
    n3 = (1 << 9);
  }
  else if (row_num == 4) {
    n3 = (1 << 8);
  }
  else if (row_num == 5){
    n3 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 9) | (1 << 8);
  }

  if (col_num == 0) {
  n2 = (1 << 7);
  }
  else if (col_num == 1) {
    n2 = (1 << 6);
  }
  else if (col_num == 2) {
    n2 = (1 << 5);
  }
  else if (col_num == 3) {
    n2 = (1 << 4);
  }
  else if (col_num == 4) {
    n2 = (1 << 3);
  }
  else if (col_num == 5){
    n2 = (1 << 7) | (1 << 6) | (1 << 5) | (1 << 4) | (1 << 3);
  }

  PIOB->PIO_CODR = 1 << 26; // digitalWrite(FB,LOW)
  PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)

  PIOC->PIO_CODR = n2; // N2 clear
  PIOC->PIO_CODR = n3; //// N3 clear
  delayMicroseconds(zero_time);
  PIOC->PIO_SODR = n2; // N2 set
  delayMicroseconds(pre_enable_time);
  PIOC->PIO_SODR = n3; // N3 set
  delayMicroseconds(pulse_width);
  PIOC->PIO_CODR = n3; // N3 clear
  delayMicroseconds(post_enable_time);
  PIOC->PIO_CODR = n2; // N2 clear
  delayMicroseconds(zero_time);
}


void Read_operation_forward_6T(String read_function, int read_time, int read_set_time, int read_delay) {
  //Serial.println("Read operation start");
  int n1;
  int n3; 
  
  if (row_num == 0) {
    n1 = (1 << 12);
    n3 = (1 << 17);
  }
  else if (row_num == 1) {
    n1 = (1 << 13);
    n3 = (1 << 18);
  }
  else if (row_num == 2) {
    n1 = (1 << 14);
    n3 = (1 << 19);
  }
  else if (row_num == 3) {
    n1 = (1 << 15);
    n3 = (1 << 9);
  }
  else if (row_num == 4) {
    n1 = (1 << 16);
    n3 = (1 << 8);
  }
  else if (row_num == 5){ // 5 means line all
    n1 = (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15) | (1 << 16);
    n3 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 9) | (1 << 8);
  }

  if (read_function == "N5") { // N5 only
    //Serial.print("N5 - ");
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)

    PIOC->PIO_SODR = n3; //N3 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n3; //N3 Clear
  }
  else if (read_function == "N6") { // N6 only
    //Serial.print("N6 - ");
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_SODR = 1 << 25; // digitalWrite(HL_CHOP,HIGH)

    PIOC->PIO_SODR = n1; //N1 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n1; //N1 Clear
  }
  else if (read_function == "N56") { // N5 and then N6
    //Serial.print("N5 - ");
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)

    PIOC->PIO_SODR = n3; //N3 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n3; //N3 Clear
    Serial.print(",");

    delayMicroseconds(20);  //to separate N5 / N6 reads

    //Serial.print("N6 - ");
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_SODR = 1 << 25; // digitalWrite(HL_CHOP,HIGH)

    PIOC->PIO_SODR = n1; //N1 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n1; //N1 Clear
  }
  else if (read_function == "N65") { // N6 and then N5
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_SODR = 1 << 25; // digitalWrite(HL_CHOP,HIGH)

    PIOC->PIO_SODR = n1; //N1 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n1; //N1 Clear
    Serial.print(",");

    delayMicroseconds(20);  //to separate N5 / N6 reads

    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)

    PIOC->PIO_SODR = n3; //N3 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n3; //N3 Clear
    }

    else if (read_function == "N5_ref"){// N1을 길게 켠 상태에서 N5를 읽어보는 실험
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)

    PIOC->PIO_SODR = n1; //N1 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n1; //N1 Clear
    Serial.print(",");
    }

    else if (read_function == "N6_ref"){// N3을 길게 켠 상태에서 N6를 읽어보는 실험
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_SODR = 1 << 25; // digitalWrite(HL_CHOP,HIGH)

    PIOC->PIO_SODR = n3; //N3 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n3; //N3 Clear
    }

    else if (read_function == "N56_ref"){
    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_CODR = 1 << 25; // digitalWrite(HL_CHOP,LOW)  

    
    PIOC->PIO_SODR = n1; //N1 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n1; //N1 Clear

    PIOB->PIO_CODR = 1 << 26;
    PIOB->PIO_SODR = 1 << 25; // digitalWrite(HL_CHOP,HIGH)
    PIOC->PIO_SODR = n3; //N3 SET
    delayMicroseconds(read_set_time);
    Read_scaling(read_time, read_delay, row_num);
    PIOC->PIO_CODR = n3; //N3 Clear
    Serial.print(",");
    }
    
     
}
    
void Read_scaling(int read_time, int read_delay, int row_num) {
   int wl_0, wl_1, wl_2, wl_3, wl_4;   
   int ADC_0, ADC_1, ADC_2, ADC_3, ADC_4;
   if (row_num == 0) {
      wl_0 = 1;
    }
    else {
      wl_0 = 0;
    }
    if (row_num == 1) {
      wl_1 = 1 << 1;
    }
    else {
      wl_1 = 0;
    }
    if (row_num == 2) {
      wl_2 = 1 << 2;
    }
    else {
      wl_2 = 0;
    }
    if (row_num == 3) {
      wl_3 = 1 << 3;
    }
    else {
      wl_3 = 0;
    }
    if (row_num == 4) {
      wl_4 = 1 << 6;
    }
    else {
      wl_4 = 0;
    }
    if(row_num == 5){
      wl_0 = 1;
      wl_1 = 1 << 1;
      wl_2 = 1 << 2;
      wl_3 = 1 << 3;
      wl_4 = 1 << 6;
    }
    /*
  int wl_0 = 1;
  int wl_1 = 1 << 1;
  int wl_2 = 1 << 2;
  int wl_3 = 1 << 3;
  int wl_4 = 1 << 6;
  */
  int wl = wl_0 | wl_1 | wl_2 | wl_3 | wl_4;

  PIOB->PIO_CODR = 1 << 14; //CON
  PIOA->PIO_SODR = 1 << 19; //DS
  PIOB->PIO_SODR = 1 << 21; //CR

  for (int i = 0; i < read_time; i++) {
    PIOD->PIO_SODR = wl ; //WL SET
    PIOA->PIO_SODR = 1 << 7 ; //DFF1 CLK HIGH
    PIOA->PIO_CODR = 1 << 7 ; //DFF1 CLK LOW
    PIOD->PIO_CODR = wl ; //WL clear
    delayMicroseconds(1000);  // WL이 닫힌 후에 얼마만큼의 delay를 더 줄 것인가? 그게 아니고 본 for문을 한 바퀴 도는 단위로 생각되어짐.
  }
  for (int i = 0; i < read_delay ; i++) {
    PIOD->PIO_SODR = 0; //
    PIOA->PIO_SODR = 1 << 7 ; //DFF1 CLK HIGH
    PIOA->PIO_CODR = 1 << 7 ; //DFF1 CLK LOW
    PIOD->PIO_CODR = wl ; //WL clear
    delayMicroseconds(10);  //
  }

  PIOA->PIO_SODR = 1 << 7 ; //DFF1 CLK HIGH
  PIOA->PIO_CODR = 1 << 7 ; //DFF1 CLK LOW
  PIOA->PIO_CODR = 1 << 19; //DS ON
  PIOB->PIO_SODR = 1 << 14 ; //CON OFF

  ADC_0 = ADC->ADC_CDR[7];// read data on A0
  ADC_1 = ADC->ADC_CDR[6];// read data on A1
  ADC_2 = ADC->ADC_CDR[5];// read data on A2
  ADC_3 = ADC->ADC_CDR[4];// read data on A3
  ADC_4 = ADC->ADC_CDR[3];// read data on A4
  PIOB->PIO_CODR = 1 << 21; //CR

  Serial.print(ADC_0 / 4);
  Serial.print(",");
  Serial.print(ADC_1 / 4);
  Serial.print(",");
  Serial.print(ADC_2 / 4);
  Serial.print(",");
  Serial.print(ADC_3 / 4);
  Serial.print(",");
  Serial.print(ADC_4 / 4);

  //  return ADC_0;
}

void Read_all_rows_sequentially(String read_function_seq, int read_time, int read_set_time, int read_delay) {
  for (int i = 0; i < 5; i++) {
    row_num = i;
    
    Serial.print("Row_");
    Serial.print(i);
    Serial.print(",");
    
    Read_operation_forward_6T(read_function_seq, read_time, read_set_time, read_delay);
    
    Serial.println(">");
  }
}

void apply_potentiation_pulses(int bit_length, int pulse_width, int pre_time, int post_time, int zero_time) {
  // 모든 N1, N2 핀을 한 번에 끄기 위한 마스크를 미리 계산
  int n1_clear_mask = 0, n2_clear_mask = 0;
  for(int i=0; i<NUM_ROWS; ++i) n1_clear_mask |= (1 << n1_pins[i]);
  for(int i=0; i<NUM_COLS; ++i) n2_clear_mask |= (1 << n2_pins[i]);
  
  PIOB->PIO_CODR = (1 << 26) | (1 << 25); // FB, HL_CHOP LOW

  // bit_length 만큼의 시간 동안 순차적으로 펄스 인가
  for (int i = 0; i < bit_length; i++) {
    int n1_set_mask = 0, n2_set_mask = 0;

    // 이번 사이클(i)에서 켤 N1, N2 핀들의 비트마스크를 계산
    for (int row = 0; row < NUM_ROWS; row++) {
      if (N1_pulses[row][i] == 1) n1_set_mask |= (1 << n1_pins[row]);
    }
    for (int col = 0; col < NUM_COLS; col++) {
      if (N2_pulses[col][i] == 1) n2_set_mask |= (1 << n2_pins[col]);
    }
    
    // 계산된 마스크를 이용해 펄스 시퀀스 실행
    PIOC->PIO_CODR = n1_clear_mask | n2_clear_mask; // 모두 끄기
    delayMicroseconds(zero_time);
    
    PIOC->PIO_SODR = n1_set_mask; // 계획된 N1 핀들만 켜기
    delayMicroseconds(pre_time);
    
    PIOC->PIO_SODR = n2_set_mask; // 계획된 N2 핀들만 켜기
    delayMicroseconds(pulse_width);
    
    PIOC->PIO_CODR = n2_clear_mask; // 모든 N2 끄기
    delayMicroseconds(post_time);
    
    PIOC->PIO_CODR = n1_clear_mask; // 모든 N1 끄기
    delayMicroseconds(zero_time);

    
  }
}

void apply_DNO_potentiation_pulses(int bit_length, int pulse_width, int pre_time, int post_time, int zero_time) {
  // 모든 N1, N2, N3 핀을 한 번에 끄기 위한 마스크를 미리 계산
  int n1_clear_mask = 0, n2_clear_mask = 0, n3_clear_mask = 0;
  for(int i=0; i<NUM_ROWS; ++i) {
    n1_clear_mask |= (1 << n1_pins[i]);
    n3_clear_mask |= (1 << n3_pins[i]);
  }
  for(int i=0; i<NUM_COLS; ++i) n2_clear_mask |= (1 << n2_pins[i]);
  
  PIOB->PIO_CODR = (1 << 26) | (1 << 25); // FB, HL_CHOP LOW

  // bit_length 만큼의 시간 동안 순차적으로 펄스 인가
  for (int i = 0; i < bit_length; i++) {
    int n1_set_mask = 0;
    int n2_set_mask = 0;
    int n3_set_mask = 0; // ✅ N3 마스크 변수 추가

    // 이번 사이클(i)에서 켤 N1, N2, N3 핀들의 비트마스크를 계산
    for (int row = 0; row < NUM_ROWS; row++) {
      if (N1_pulses[row][i] == 1) {
        n1_set_mask |= (1 << n1_pins[row]);
      } else {
        // ✅ N1 펄스가 0이면, 대신 N3 핀을 켜도록 마스크에 추가
        n3_set_mask |= (1 << n3_pins[row]);
      }
    }
    for (int col = 0; col < NUM_COLS; col++) {
      if (N2_pulses[col][i] == 1) n2_set_mask |= (1 << n2_pins[col]);
    }
    
    // 계산된 마스크를 이용해 펄스 시퀀스 실행
    PIOC->PIO_CODR = n1_clear_mask | n2_clear_mask | n3_clear_mask; // N3도 끄기
    delayMicroseconds(zero_time);
    
    // ✅ N1과 N3 펄스를 동시에 인가
    PIOC->PIO_SODR = n1_set_mask | n3_set_mask; 
    delayMicroseconds(pre_time);
    
    PIOC->PIO_SODR = n2_set_mask;
    delayMicroseconds(pulse_width);
    
    PIOC->PIO_CODR = n2_clear_mask;
    delayMicroseconds(post_time);
    
    // ✅ N1과 N3 핀들을 모두 끔
    PIOC->PIO_CODR = n1_clear_mask | n3_clear_mask;
    delayMicroseconds(zero_time);
  }
}


void apply_depression_pulses(int bit_length, int pulse_width, int pre_time, int post_time, int zero_time) {
  int n3_clear_mask = 0, n4_clear_mask = 0;
  for(int i=0; i<NUM_ROWS; ++i) n3_clear_mask |= (1 << n3_pins[i]);
  for(int i=0; i<NUM_COLS; ++i) n4_clear_mask |= (1 << n4_pins[i]);
  
  PIOB->PIO_CODR = (1 << 26) | (1 << 25);

  for (int i = 0; i < bit_length; i++) {
    int n3_set_mask = 0, n4_set_mask = 0;
    for (int row = 0; row < NUM_ROWS; row++) {
      if (N3_pulses[row][i] == 1) n3_set_mask |= (1 << n3_pins[row]);
    }
    for (int col = 0; col < NUM_COLS; col++) {
      if (N4_pulses[col][i] == 1) n4_set_mask |= (1 << n4_pins[col]);
    }
    
    // N3, N4 핀 제어
    PIOC->PIO_CODR = n3_clear_mask | n4_clear_mask;
    delayMicroseconds(zero_time);
    PIOC->PIO_SODR = n3_set_mask;
    delayMicroseconds(pre_time);
    PIOC->PIO_SODR = n4_set_mask;
    delayMicroseconds(pulse_width);
    PIOC->PIO_CODR = n4_clear_mask;
    delayMicroseconds(post_time);
    PIOC->PIO_CODR = n3_clear_mask;
    delayMicroseconds(zero_time);
  }
}

void apply_DNO_depression_pulses(int bit_length, int pulse_width, int pre_time, int post_time, int zero_time) {
  // 모든 N1, N3, N4 핀을 한 번에 끄기 위한 마스크를 미리 계산
  int n1_clear_mask = 0, n3_clear_mask = 0, n4_clear_mask = 0;
  for(int i=0; i<NUM_ROWS; ++i) {
    n1_clear_mask |= (1 << n1_pins[i]);
    n3_clear_mask |= (1 << n3_pins[i]);
  }
  for(int i=0; i<NUM_COLS; ++i) n4_clear_mask |= (1 << n4_pins[i]);
  
  PIOB->PIO_CODR = (1 << 26) | (1 << 25); // FB, HL_CHOP LOW

  for (int i = 0; i < bit_length; i++) {
    int n1_set_mask = 0; // ✅ N1 마스크 변수 추가
    int n3_set_mask = 0;
    int n4_set_mask = 0;

    // 이번 사이클(i)에서 켤 N1, N3, N4 핀들의 비트마스크를 계산
    for (int row = 0; row < NUM_ROWS; row++) {
      if (N3_pulses[row][i] == 1) {
        n3_set_mask |= (1 << n3_pins[row]);
      } else {
        // ✅ N3 펄스가 0이면, 대신 N1 핀을 켜도록 마스크에 추가
        n1_set_mask |= (1 << n1_pins[row]);
      }
    }
    for (int col = 0; col < NUM_COLS; col++) {
      if (N4_pulses[col][i] == 1) n4_set_mask |= (1 << n4_pins[col]);
    }
    
    // N1, N3, N4 핀 제어
    PIOC->PIO_CODR = n1_clear_mask | n3_clear_mask | n4_clear_mask; // N1도 끄기
    delayMicroseconds(zero_time);
    
    // ✅ N1과 N3 펄스를 동시에 인가
    PIOC->PIO_SODR = n1_set_mask | n3_set_mask; 
    delayMicroseconds(pre_time);
    
    PIOC->PIO_SODR = n4_set_mask;
    delayMicroseconds(pulse_width);
    
    PIOC->PIO_CODR = n4_clear_mask;
    delayMicroseconds(post_time);
    
    // ✅ N1과 N3 핀들을 모두 끔
    PIOC->PIO_CODR = n1_clear_mask | n3_clear_mask;
    delayMicroseconds(zero_time);
  }
}

String getNextToken(String& str, int& startIndex) {
  if (startIndex >= str.length()) return "";
  int endIndex = str.indexOf(',', startIndex);
  String token;
  if (endIndex == -1) { // 마지막 토큰
    token = str.substring(startIndex);
    startIndex = str.length();
  } else {
    token = str.substring(startIndex, endIndex);
    startIndex = endIndex + 1;
  }
  return token;
}

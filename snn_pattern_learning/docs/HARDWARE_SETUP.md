# E-prop Hardware Experiment Setup Guide

이 문서는 memristor crossbar array를 사용한 E-prop 학습 실험을 위한 하드웨어 연동 가이드입니다.

## 개요

`Basic_RSNN_eprop_HW_forward` 모델은 5x5 memristor crossbar array를 사용하여 출력층 gradient를 실제 하드웨어에서 계산합니다. 이는 다음과 같은 방식으로 동작합니다:

1. **Stochastic Pulse 기반 업데이트**: Error vector와 trace vector를 확률값으로 변환
2. **Arduino 통신**: Serial 포트(COM7)를 통해 아두이노와 통신
3. **Potentiation/Depression**: 부호에 따라 potentiation 또는 depression 수행
4. **Gradient 누적**: 여러 timestep에 걸쳐 하드웨어에서 gradient 누적

## 파일 구조

```
SNN/snn_pattern_learning/
├── hardware/
│   ├── __init__.py                  # Hardware 모듈 초기화
│   └── hw_interface.py              # MemristorInterface 및 MockMemristorInterface 구현
├── configs/
│   ├── eprop_hardware.yaml          # 실제 하드웨어용 설정
│   └── eprop_hardware_mock.yaml     # Mock 하드웨어용 테스트 설정
├── models/
│   └── models.py                    # Basic_RSNN_eprop_HW_forward 모델
└── test_hardware.py                 # 하드웨어 인터페이스 테스트 스크립트
```

## Hardware Interface

### MemristorInterface

실제 아두이노 하드웨어와 통신하는 인터페이스입니다.

**주요 메서드:**
- `connect()`: 아두이노에 연결
- `disconnect()`: 아두이노 연결 해제
- `reset()`: 누적된 gradient를 0으로 초기화
- `accumulate_outer_product(err_probs, trace_probs, err_signs, trace_signs)`: Outer product gradient 누적
- `send_outer_product_update(u_probs, v_probs, direction)`: 단일 업데이트 전송 (POTENTIATION 또는 DEPRESSION)
- `read_accumulated_gradient()`: 누적된 gradient 읽기 (5x5 행렬)

**Arduino 통신 프로토콜:**
```
명령어 형식:
F,0,0,STOCHASTIC_<MODE>,N56,<bit_length>,<width>,<pre>,<post>,<zero>,<time>,<delay>,<5 u_streams>,<5 v_streams>

예시:
F,0,0,STOCHASTIC_POTENTIATION,N56,10,1,100,100,10,10,10,0110100101,1101001011,0011001110,1100110011,0101010101,1010101010,1111000011,0000111100,1100001111,0011110000

응답 형식:
PRE-UPDATE-READ
<5줄의 ADC 데이터, 각 10개 값>
POST-UPDATE-READ
<5줄의 ADC 데이터, 각 10개 값>
EOD
```

### MockMemristorInterface

테스트용 소프트웨어 기반 인터페이스입니다. 실제 하드웨어 없이 동일한 API로 동작합니다.

## 사용 방법

### 1. Mock Hardware로 테스트

먼저 mock hardware로 전체 시스템을 테스트하세요:

```bash
cd SNN/snn_pattern_learning

# Mock interface만 테스트
python test_hardware.py --mock

# E-prop 실험 실행 (mock hardware)
python main_unified.py --config eprop_hardware_mock.yaml --epochs 10 --verbose
```

### 2. 실제 Hardware 테스트

아두이노가 COM7에 연결되어 있는지 확인한 후:

```bash
# Hardware interface 테스트
python test_hardware.py --real

# E-prop 실험 실행 (real hardware)
python main_unified.py --config eprop_hardware.yaml --epochs 50 --verbose
```

## 설정 파일

### eprop_hardware.yaml (실제 하드웨어용)

```yaml
model:
  type: "RSNN_eprop_HW_forward"
  n_in: 50
  n_hidden: 5    # MUST be 5 for 5x5 hardware
  n_out: 5       # MUST be 5 for 5x5 hardware

  hardware:
    enabled: true
    use_mock_hw: false      # 실제 하드웨어 사용
    serial_port: "COM7"     # 아두이노 포트
    baud_rate: 115200
    bit_length: 10          # Stochastic pulse 스트림 길이
```

### eprop_hardware_mock.yaml (테스트용)

```yaml
model:
  hardware:
    enabled: true
    use_mock_hw: true       # Mock 하드웨어 사용
```

## 하드웨어 제약사항

1. **5x5 크기 고정**: 현재 하드웨어는 5x5 crossbar array만 지원합니다.
   - `n_hidden`과 `n_out`이 모두 5여야 합니다.
   - 다른 크기를 사용하면 초기화 시 ValueError 발생

2. **Serial 통신**: COM7 포트를 사용합니다.
   - Windows에서만 테스트됨
   - 다른 OS에서는 `/dev/ttyUSB0` 등으로 변경 필요

3. **Timing 파라미터**: 하드웨어 특성에 맞게 조정 필요
   - `pulse_width`, `pulse_pre`, `pulse_post` 등
   - 기본값은 quick_measure.ipynb의 값을 사용

## 문제 해결

### "Hardware not connected" 경고

이 경고는 정상적인 동작일 수 있습니다:
- Mock 하드웨어를 사용하는 경우, 실제 연결 없이도 동작합니다
- 실제 하드웨어 사용 시 이 경고가 나오면 COM7 연결을 확인하세요

### Serial 포트 오류

```
FileNotFoundError: could not open port 'COM7'
```

해결 방법:
1. 아두이노가 제대로 연결되어 있는지 확인
2. 장치 관리자에서 COM 포트 번호 확인
3. 설정 파일의 `serial_port` 값 수정

### 모델 크기 오류

```
ValueError: n_hidden and n_out must be 5 for 5x5 hardware
```

해결 방법:
- 설정 파일에서 `n_hidden`과 `n_out`을 5로 변경

## 다음 단계

1. **Mock Hardware 테스트**: `test_hardware.py --mock`로 시스템 검증
2. **실제 Hardware 연결**: 아두이노 연결 및 통신 테스트
3. **E-prop 실험 실행**: 하드웨어 gradient 누적으로 학습 수행
4. **결과 분석**: `results/eprop_hardware/` 폴더에서 결과 확인

## 참고 자료

- quick_measure.ipynb: 아두이노 통신 프로토콜 예제
- models/models.py: Basic_RSNN_eprop_HW_forward 구현
- hardware/hw_interface.py: Hardware interface 구현

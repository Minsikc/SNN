# SNN Pattern Learning - Unified Experiment System

이 프로젝트는 Spiking Neural Networks (SNN)를 위한 통합 실험 시스템입니다. 다양한 실험 타입을 YAML 설정 파일로 관리하고, 뉴런 타입을 동적으로 선택할 수 있습니다.

## 주요 기능

### 🔧 설정 기반 실험 관리
- **YAML 설정 파일**: 모든 실험 파라미터를 중앙에서 관리
- **뉴런 타입 선택**: Triangular/Boxcar 뉴런을 설정으로 선택 가능
- **실험 타입 통합**: Basic, Teacher-Student, Weight Initialization 실험 지원

### 🧪 실험 타입
1. **Basic Experiment**: 기본적인 SNN 학습 실험
2. **Teacher-Student Experiment**: 교사 모델에서 학생 모델로의 지식 전이
3. **Weight Initialization Experiment**: 다양한 가중치 초기화 방법 비교

### 🧠 뉴런 타입
- **Triangular**: 삼각형 서로게이트 그라디언트
- **Boxcar**: 박스카 서로게이트 그라디언트

## 디렉토리 구조

```
snn_pattern_learning/
├── configs/                    # 설정 파일들
│   ├── default.yaml           # 기본 설정
│   ├── teacher_student.yaml   # 교사-학생 실험 설정
│   ├── weight_init.yaml       # 가중치 초기화 실험 설정
│   └── config_loader.py       # 설정 로더
├── experiment_types/          # 실험 타입 모듈들
│   ├── base_experiment.py     # 기본 실험 클래스
│   ├── basic_experiment.py    # 기본 실험
│   ├── teacher_student_experiment.py  # 교사-학생 실험
│   └── weight_init_experiment.py      # 가중치 초기화 실험
├── models/                    # 모델 관련
│   ├── models.py             # 기존 모델들
│   └── model_factory.py      # 모델 팩토리 (뉴런 타입 지원)
├── utils/                     # 유틸리티
│   ├── experiment_logger.py   # 실험 로깅
│   ├── pattern_analyzer.py    # 패턴 분석
│   └── ...                   # 기타 유틸리티들
├── main_unified.py           # 통합 메인 스크립트
└── run_demos.py              # 데모 실행 스크립트
```

## 사용법

### 1. 기본 실행

```bash
# 기본 설정으로 실행
python main_unified.py

# 특정 설정 파일로 실행
python main_unified.py --config teacher_student.yaml

# 설정 파일 목록 확인
python main_unified.py --list_configs
```

### 2. 커맨드라인 오버라이드

```bash
# 에포크 수 변경
python main_unified.py --epochs 100

# 뉴런 타입 변경
python main_unified.py --neuron_type boxcar

# 모델 타입 변경
python main_unified.py --model_type Basic_RSNN_spike

# 여러 설정 동시 변경
python main_unified.py --config default.yaml --epochs 50 --learning_rate 0.01 --neuron_type triangular
```

### 3. 데모 실행

```bash
# 모든 실험 타입 데모 실행
python run_demos.py
```

## 설정 파일 예시

### 기본 설정 (default.yaml)
```yaml
experiment:
  type: "basic"
  name: "default_experiment"
  
model:
  type: "RSNN_eprop_analog_forward"
  n_in: 50
  n_hidden: 40
  n_out: 10
  neuron_type: "triangular"
  
neuron:
  triangular:
    thresh: 0.6
    subthresh: 0.25
    gamma: 0.3
    width: 1
  boxcar:
    thresh: 0.4
    subthresh: 0.1
    alpha: 1.0
    
training:
  epochs: 200
  learning_rate: 0.1
  batch_size: 1
```

### 교사-학생 실험 (teacher_student.yaml)
```yaml
experiment:
  type: "teacher_student"
  name: "teacher_student_experiment"
  
teacher_student:
  teacher_model_type: "Basic_RSNN_spike"
  student_model_type: "Basic_RSNN_spike"
  teacher_epochs: 100
  student_epochs: 200
  track_weight_diff: true
  
logging:
  enable_logger: true
  enable_pattern_analyzer: true
```

## 실험 결과

실험 결과는 다음과 같이 저장됩니다:

```
results/
├── experiment_name_timestamp/
│   ├── metadata.json          # 실험 메타데이터
│   ├── results.json           # 실험 결과
│   ├── training_curves.png    # 학습 곡선
│   └── best_model.pth         # 최고 성능 모델
```

## 기존 실험 파일들과의 호환성

기존의 `experiment_*.py` 파일들은 여전히 사용 가능하지만, 새로운 통합 시스템을 사용하는 것을 권장합니다:

- `experiment_teacher_student.py` → `main_unified.py --config teacher_student.yaml`
- `experiment_weight_init_batch.py` → `main_unified.py --config weight_init.yaml`

## 확장 가능성

### 새로운 실험 타입 추가
1. `experiment_types/` 폴더에 새로운 실험 클래스 생성
2. `BaseExperiment`를 상속받아 `run()` 메서드 구현
3. `configs/` 폴더에 해당 실험용 설정 파일 추가
4. `main_unified.py`의 `create_experiment()` 함수에 추가

### 새로운 뉴런 타입 추가
1. `neurons.py`에 새로운 뉴런 클래스 구현
2. `models/model_factory.py`의 `create_neuron_function()`에 추가
3. 설정 파일에 해당 뉴런 타입 설정 추가

## 문제 해결

### 일반적인 문제들
1. **모듈 import 오류**: Python path 설정 확인
2. **설정 파일 오류**: YAML 문법 검사
3. **CUDA 메모리 오류**: 배치 크기 감소

### 디버깅
```bash
# 자세한 로그 확인
python main_unified.py --verbose

# 설정 파일 확인
python main_unified.py --list_configs
```

## 개발자 정보

이 통합 시스템은 기존의 분산된 실험 스크립트들을 통합하여 관리하기 쉽도록 리팩토링한 것입니다. 
모든 실험 타입과 뉴런 타입을 설정 파일로 관리할 수 있어 실험의 재현성과 관리가 용이합니다.
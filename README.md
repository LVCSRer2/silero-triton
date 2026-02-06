# Silero VAD Triton Inference Server

Silero VAD 모델을 NVIDIA Triton Inference Server에서 서빙하고, 10개의 마이크 입력을 동시에 처리하는 클라이언트입니다.

## 구조

```
silero_triton/
├── model_repository/
│   └── silero_vad/
│       ├── 1/
│       │   └── model.onnx          # Silero VAD ONNX 모델
│       └── config.pbtxt            # Triton 모델 설정
├── client/
│   ├── multi_mic_client.py         # 동기식 멀티 마이크 클라이언트
│   └── async_multi_mic_client.py   # 비동기식 멀티 마이크 클라이언트
├── download_model.py               # 모델 다운로드 스크립트
├── docker-compose.yml              # Triton 서버 Docker 설정
├── requirements.txt                # Python 의존성
└── README.md
```

## 설치 및 실행

### 1. 환경 설정

```bash
# Python 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Silero VAD 모델 다운로드

```bash
python download_model.py
```

### 3. Triton 서버 시작

```bash
# Docker Compose로 Triton 서버 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 4. 클라이언트 실행

```bash
# 동기식 클라이언트 (스레드 기반)
python client/multi_mic_client.py --num-mics 10

# 비동기식 클라이언트 (asyncio 기반, 더 높은 성능)
python client/async_multi_mic_client.py --num-mics 10

# 사용 가능한 오디오 장치 확인
python client/multi_mic_client.py --list-devices
```

## 클라이언트 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--url` | localhost:8001 | Triton gRPC 서버 URL |
| `--num-mics` | 10 | 동시 처리할 마이크 수 |
| `--sample-rate` | 16000 | 오디오 샘플링 레이트 |
| `--list-devices` | - | 사용 가능한 오디오 장치 목록 출력 |

## Triton 모델 설정

`config.pbtxt` 주요 설정:

- **platform**: `onnxruntime_onnx` (ONNX Runtime 사용)
- **instance_group**: GPU에서 2개 인스턴스 실행
- **dynamic_batching**: 동적 배칭으로 처리량 최적화
- **TensorRT 최적화**: FP16 정밀도로 추론 가속

## 성능 최적화

1. **동적 배칭**: 여러 요청을 배치로 묶어 처리량 향상
2. **TensorRT**: ONNX 모델을 TensorRT로 최적화
3. **비동기 클라이언트**: asyncio로 I/O 대기 시간 최소화
4. **멀티 인스턴스**: GPU에서 여러 모델 인스턴스 실행

## 실제 환경 적용 시 주의사항

1. **다중 마이크 장치**: 실제 환경에서는 각 마이크에 다른 device_index 지정 필요
2. **네트워크 지연**: 원격 서버 사용 시 gRPC 스트리밍 고려
3. **메모리 관리**: 장시간 운영 시 메모리 모니터링 필요

## 라이선스

- Silero VAD: MIT License
- 이 코드: MIT License

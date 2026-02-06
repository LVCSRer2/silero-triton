# Silero VAD Triton Inference Server

Silero VAD 모델을 ONNX Runtime 기반 서버에서 서빙하고, 다수의 마이크 입력을 동시에 처리하는 클라이언트입니다.

## 구조

```
silero_triton/
├── model_repository/
│   └── silero_vad/
│       ├── 1/
│       │   └── model.onnx                # Silero VAD ONNX 모델
│       └── config.pbtxt                  # Triton 모델 설정
├── server/
│   └── onnx_vad_server.py                # ONNX Runtime TCP 서버
├── client/
│   ├── real_mic_30_clients.py            # 실제 마이크 + 다중 클라이언트
│   ├── simulate_multi_mic_test.py        # 시뮬레이션 부하 테스트
│   ├── tcp_multi_mic_client.py           # TCP 마이크 클라이언트
│   ├── multi_mic_client.py               # Triton gRPC 멀티 마이크 클라이언트
│   └── async_multi_mic_client.py         # Triton gRPC 비동기 클라이언트
├── download_model.py                     # 모델 다운로드 (torch 불필요)
├── docker-compose.yml                    # Triton 서버 Docker 설정
├── requirements.txt
└── README.md
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt

# 마이크 사용 시 PortAudio 필요
sudo apt-get install -y libportaudio2 portaudio19-dev
```

필요 패키지: `onnxruntime`, `numpy`, `sounddevice`, `tritonclient[grpc]`, `scipy`

> torch 의존성 없음

### 2. 모델 다운로드

```bash
python download_model.py
```

GitHub에서 Silero VAD ONNX 모델을 직접 다운로드합니다.

### 3. 서버 실행

```bash
# ONNX Runtime TCP 서버 (Docker 불필요)
python server/onnx_vad_server.py --port 8001

# 또는 Docker로 Triton 서버 실행
docker-compose up -d
```

### 4. 클라이언트 실행

```bash
# 실제 마이크 + 30개 동시 클라이언트
python client/real_mic_30_clients.py --num-clients 30

# 시뮬레이션 부하 테스트
python client/simulate_multi_mic_test.py --num-clients 100 --duration 10

# 오디오 장치 확인
python client/real_mic_30_clients.py --list-devices
```

## 클라이언트 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` | localhost | 서버 호스트 |
| `--port` | 8001 | 서버 포트 |
| `--num-clients` | 30 | 동시 처리할 클라이언트 수 |
| `--device` | 기본 장치 | 마이크 장치 인덱스 |
| `--duration` | 무제한 | 테스트 시간(초) |
| `--list-devices` | - | 오디오 장치 목록 출력 |

## 성능 테스트 결과 (CPU)

### 실제 마이크 입력 테스트

| 동시 클라이언트 | 평균 레이턴시 | 상태 |
|---------------|-------------|------|
| 30 | 1.72ms | ✅ |
| 50 | 1.86ms | ✅ |
| 100 | 1.96ms | ✅ |
| 200 | 2.03ms | ✅ |
| 500 | 1.90ms | ✅ |
| 1000 | 2.30ms | ✅ |

### 시뮬레이션 부하 테스트

| 동시 클라이언트 | 평균 레이턴시 | 처리량 |
|---------------|-------------|--------|
| 10 | 1.36ms | 300/s |
| 50 | 14.55ms | 1,401/s |
| 100 | 61.37ms | 1,385/s |
| 200 | 123.31ms | 1,253/s |

## Triton 모델 설정

`config.pbtxt` 주요 설정:

- **platform**: `onnxruntime_onnx` (ONNX Runtime)
- **instance_group**: CPU에서 4개 인스턴스 실행
- **dynamic_batching**: 동적 배칭으로 처리량 최적화

## 라이선스

- Silero VAD: MIT License
- 이 코드: MIT License

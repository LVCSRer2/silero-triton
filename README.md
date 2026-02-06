# Silero VAD Inference Server

Silero VAD 모델을 ONNX Runtime 기반 서버에서 서빙하고, 다수의 마이크 입력을 동시에 처리하는 시스템입니다.

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
│   ├── real_mic_30_clients.py            # 실제 마이크 + N개 동시 부하 테스트
│   ├── triton_mic_test.py                # Triton gRPC 마이크 부하 테스트
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
# ONNX Runtime TCP 서버 (권장, Docker 불필요)
python server/onnx_vad_server.py --port 8001

# 또는 Docker로 Triton 서버 실행
docker-compose up -d
```

### 4. 클라이언트 실행

```bash
# 실제 마이크 + 30개 동시 요청 (TCP 서버)
python client/real_mic_30_clients.py --num-clients 30

# 실제 마이크 + Triton 서버 테스트
python client/triton_mic_test.py --num-clients 30

# 시뮬레이션 부하 테스트
python client/simulate_multi_mic_test.py --num-clients 100 --duration 10

# 오디오 장치 확인
python client/real_mic_30_clients.py --list-devices
```

## 클라이언트 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` / `--url` | localhost | 서버 호스트 |
| `--port` | 8001 | 서버 포트 |
| `--num-clients` | 30 | 동시 처리할 클라이언트 수 |
| `--device` | 기본 장치 | 마이크 장치 인덱스 |
| `--duration` | 무제한 | 테스트 시간(초) |
| `--list-devices` | - | 오디오 장치 목록 출력 |

## 성능 테스트 결과 (CPU)

### 실제 마이크 동시 부하 테스트: TCP 서버 vs Triton Server

마이크 1개의 오디오를 N개 클라이언트 모두에게 복제하여 동시에 서버로 전송하는 방식입니다.
실시간 처리 기준은 32ms 미만 (512 samples @ 16kHz).

| 동시 클라이언트 | TCP (ONNX Runtime) | Triton Server | TCP 실시간 | Triton 실시간 |
|---------------|-------------------|---------------|-----------|-------------|
| 30 | **11.49ms** | 28.86ms | ✅ | ✅ |
| 50 | **25.29ms** | 47.23ms | ✅ | ❌ |
| 100 | **67.26ms** | 91.51ms | ❌ | ❌ |

### 시뮬레이션 부하 테스트 (TCP 서버)

각 클라이언트가 독립적으로 32ms마다 오디오를 생성하여 전송합니다.

| 동시 클라이언트 | 평균 레이턴시 | 처리량 |
|---------------|-------------|--------|
| 10 | 1.36ms | 300/s |
| 20 | 2.03ms | 599/s |
| 50 | 14.55ms | 1,401/s |
| 100 | 61.37ms | 1,385/s |
| 200 | 123.31ms | 1,253/s |

### 분석

- **실시간 처리 한계**: TCP 서버 기준 약 **50개** 동시 마이크 입력
- **TCP 서버가 Triton보다 빠른 이유**:
  - Silero VAD는 매우 가벼운 모델 (~2ms 추론)
  - gRPC 직렬화/역직렬화 오버헤드가 추론 시간보다 큼
  - Docker 컨테이너 네트워크 계층의 추가 지연
- **Triton이 유리한 경우**: 추론 시간이 긴 모델 (ASR, LLM 등)에서 배칭/멀티인스턴스 효과가 프로토콜 오버헤드를 상회

## 라이선스

- Silero VAD: MIT License
- 이 코드: MIT License

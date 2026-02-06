#!/usr/bin/env python3
"""
10개의 마이크 입력을 동시에 받아 Silero VAD Triton 서버로 테스트하는 클라이언트
"""
import numpy as np
import tritonclient.grpc as grpcclient
import sounddevice as sd
import threading
import queue
import time
import argparse
from dataclasses import dataclass
from typing import Optional
import signal
import sys


@dataclass
class VADState:
    """각 마이크 스트림의 VAD 상태를 저장"""
    h: np.ndarray  # LSTM hidden state
    c: np.ndarray  # LSTM cell state
    is_speaking: bool = False
    speech_prob: float = 0.0

    @classmethod
    def create_initial(cls, batch_size: int = 1):
        return cls(
            h=np.zeros((2, batch_size, 64), dtype=np.float32),
            c=np.zeros((2, batch_size, 64), dtype=np.float32)
        )


class MicrophoneStream:
    """단일 마이크 스트림 처리"""

    def __init__(self, mic_id: int, device_index: Optional[int],
                 sample_rate: int = 16000, chunk_size: int = 512):
        self.mic_id = mic_id
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # 512 samples = 32ms at 16kHz
        self.audio_queue = queue.Queue()
        self.running = False
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        """sounddevice 콜백: 오디오 데이터를 큐에 추가"""
        if status:
            print(f"[Mic {self.mic_id}] Status: {status}")
        # 모노로 변환하고 큐에 추가
        audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.audio_queue.put(audio_data.flatten())

    def start(self):
        """스트림 시작"""
        self.running = True
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32,
            callback=self.audio_callback
        )
        self.stream.start()
        print(f"[Mic {self.mic_id}] 스트림 시작 (device: {self.device_index})")

    def stop(self):
        """스트림 정지"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print(f"[Mic {self.mic_id}] 스트림 정지")

    def get_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """큐에서 오디오 데이터 가져오기"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class TritonVADClient:
    """Triton 서버와 통신하는 VAD 클라이언트"""

    def __init__(self, url: str = "localhost:8001", model_name: str = "silero_vad"):
        self.url = url
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=url)

        # 서버 연결 확인
        if not self.client.is_server_live():
            raise ConnectionError(f"Triton 서버에 연결할 수 없습니다: {url}")

        if not self.client.is_model_ready(model_name):
            raise RuntimeError(f"모델이 준비되지 않았습니다: {model_name}")

        print(f"Triton 서버 연결 완료: {url}, 모델: {model_name}")

    def infer(self, audio: np.ndarray, state: VADState) -> tuple[float, VADState]:
        """
        VAD 추론 수행

        Args:
            audio: 오디오 샘플 [samples]
            state: 현재 VAD 상태

        Returns:
            (speech_probability, new_state)
        """
        batch_size = 1

        # 입력 준비
        audio_input = audio.reshape(batch_size, -1).astype(np.float32)
        sr_input = np.array([16000], dtype=np.int64)

        # Triton 입력 생성
        inputs = [
            grpcclient.InferInput("input", audio_input.shape, "FP32"),
            grpcclient.InferInput("sr", sr_input.shape, "INT64"),
            grpcclient.InferInput("h", state.h.shape, "FP32"),
            grpcclient.InferInput("c", state.c.shape, "FP32"),
        ]

        inputs[0].set_data_from_numpy(audio_input)
        inputs[1].set_data_from_numpy(sr_input)
        inputs[2].set_data_from_numpy(state.h)
        inputs[3].set_data_from_numpy(state.c)

        # 출력 요청
        outputs = [
            grpcclient.InferRequestedOutput("output"),
            grpcclient.InferRequestedOutput("hn"),
            grpcclient.InferRequestedOutput("cn"),
        ]

        # 추론 실행
        result = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )

        # 결과 파싱
        speech_prob = result.as_numpy("output")[0, 0]
        new_h = result.as_numpy("hn")
        new_c = result.as_numpy("cn")

        # 새 상태 생성
        new_state = VADState(
            h=new_h,
            c=new_c,
            speech_prob=float(speech_prob),
            is_speaking=speech_prob > 0.5
        )

        return float(speech_prob), new_state


class MultiMicVADProcessor:
    """여러 마이크를 동시에 처리하는 VAD 프로세서"""

    def __init__(self, triton_url: str = "localhost:8001",
                 num_mics: int = 10, sample_rate: int = 16000):
        self.triton_client = TritonVADClient(triton_url)
        self.num_mics = num_mics
        self.sample_rate = sample_rate
        self.mic_streams: list[MicrophoneStream] = []
        self.vad_states: list[VADState] = []
        self.running = False
        self.threads: list[threading.Thread] = []
        self.results_lock = threading.Lock()
        self.results: dict[int, dict] = {}

    def _get_available_devices(self) -> list[int]:
        """사용 가능한 입력 장치 목록"""
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                input_devices.append(i)
                print(f"  입력 장치 {i}: {d['name']} (채널: {d['max_input_channels']})")
        return input_devices

    def setup_microphones(self, device_indices: Optional[list[int]] = None):
        """마이크 스트림 설정"""
        print("\n사용 가능한 입력 장치:")
        available = self._get_available_devices()

        if not available:
            raise RuntimeError("사용 가능한 입력 장치가 없습니다!")

        # 장치 인덱스가 지정되지 않으면 기본 장치를 반복 사용
        if device_indices is None:
            default_device = sd.default.device[0]  # 기본 입력 장치
            device_indices = [default_device] * self.num_mics
            print(f"\n기본 입력 장치({default_device})를 {self.num_mics}개 스트림에 사용합니다.")
            print("(실제 환경에서는 각각 다른 마이크 장치 인덱스를 지정하세요)")

        # 마이크 스트림 및 VAD 상태 초기화
        for i in range(self.num_mics):
            device_idx = device_indices[i] if i < len(device_indices) else device_indices[-1]
            stream = MicrophoneStream(
                mic_id=i,
                device_index=device_idx,
                sample_rate=self.sample_rate
            )
            self.mic_streams.append(stream)
            self.vad_states.append(VADState.create_initial())
            self.results[i] = {"prob": 0.0, "speaking": False, "count": 0}

    def _process_mic(self, mic_id: int):
        """개별 마이크 처리 스레드"""
        stream = self.mic_streams[mic_id]
        state = self.vad_states[mic_id]

        while self.running:
            audio = stream.get_audio(timeout=0.1)
            if audio is None:
                continue

            try:
                prob, new_state = self.triton_client.infer(audio, state)
                state = new_state
                self.vad_states[mic_id] = state

                with self.results_lock:
                    self.results[mic_id] = {
                        "prob": prob,
                        "speaking": prob > 0.5,
                        "count": self.results[mic_id]["count"] + 1
                    }

            except Exception as e:
                print(f"[Mic {mic_id}] 추론 오류: {e}")

    def start(self):
        """모든 마이크 처리 시작"""
        self.running = True

        # 마이크 스트림 시작
        for stream in self.mic_streams:
            stream.start()

        # 처리 스레드 시작
        for i in range(self.num_mics):
            thread = threading.Thread(target=self._process_mic, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        print(f"\n{self.num_mics}개 마이크 처리 시작!")

    def stop(self):
        """모든 마이크 처리 정지"""
        print("\n처리 정지 중...")
        self.running = False

        for stream in self.mic_streams:
            stream.stop()

        for thread in self.threads:
            thread.join(timeout=1.0)

        print("모든 스트림 정지 완료")

    def display_status(self):
        """현재 VAD 상태 표시"""
        with self.results_lock:
            print("\n" + "=" * 70)
            print(f"{'Mic':<6} {'Speech Prob':<15} {'Speaking':<12} {'Inferences':<12}")
            print("-" * 70)

            for mic_id in range(self.num_mics):
                r = self.results[mic_id]
                prob_bar = "█" * int(r["prob"] * 20) + "░" * (20 - int(r["prob"] * 20))
                status = "🎤 SPEAKING" if r["speaking"] else "   silent"
                print(f"Mic {mic_id:<3} [{prob_bar}] {r['prob']:.3f}  {status:<12} {r['count']:<8}")

            print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Multi-Microphone VAD Triton Client")
    parser.add_argument("--url", type=str, default="localhost:8001",
                        help="Triton 서버 URL (default: localhost:8001)")
    parser.add_argument("--num-mics", type=int, default=10,
                        help="동시 처리할 마이크 수 (default: 10)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="샘플링 레이트 (default: 16000)")
    parser.add_argument("--list-devices", action="store_true",
                        help="사용 가능한 오디오 장치 목록 출력 후 종료")
    args = parser.parse_args()

    if args.list_devices:
        print("사용 가능한 오디오 장치:")
        print(sd.query_devices())
        return

    print("=" * 70)
    print("Silero VAD - Multi-Microphone Triton Client")
    print("=" * 70)

    processor = MultiMicVADProcessor(
        triton_url=args.url,
        num_mics=args.num_mics,
        sample_rate=args.sample_rate
    )

    # Ctrl+C 핸들러
    def signal_handler(sig, frame):
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        processor.setup_microphones()
        processor.start()

        # 상태 표시 루프
        while True:
            time.sleep(0.5)
            processor.display_status()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()

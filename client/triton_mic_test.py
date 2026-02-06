#!/usr/bin/env python3
"""
Triton Server gRPC를 사용하는 실제 마이크 동시 부하 테스트
TCP 서버와 비교하기 위한 클라이언트
"""
import numpy as np
import sounddevice as sd
import tritonclient.grpc as grpcclient
import threading
import queue
import time
import argparse
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque


@dataclass
class ClientStats:
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_prob: float = 0.0
    is_speaking: bool = False


class TritonVADClient:
    """Triton gRPC VAD 클라이언트"""

    CONTEXT_SIZE = 64  # Silero VAD v5: 16kHz에서 context 64 샘플

    def __init__(self, client_id: int, url: str = "localhost:9001"):
        self.client_id = client_id
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=url)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT_SIZE, dtype=np.float32)

    def infer(self, audio: np.ndarray) -> Optional[float]:
        try:
            audio = audio.flatten().astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)

            # Silero VAD v5: context 붙이기 (512 → 576)
            audio_with_context = np.concatenate([self.context, audio])
            self.context = audio[-self.CONTEXT_SIZE:].copy()

            audio_input = audio_with_context.reshape(1, -1)
            sr_input = np.array([16000], dtype=np.int64)

            inputs = [
                grpcclient.InferInput("input", audio_input.shape, "FP32"),
                grpcclient.InferInput("state", self.state.shape, "FP32"),
                grpcclient.InferInput("sr", sr_input.shape, "INT64"),
            ]
            inputs[0].set_data_from_numpy(audio_input)
            inputs[1].set_data_from_numpy(self.state)
            inputs[2].set_data_from_numpy(sr_input)

            outputs = [
                grpcclient.InferRequestedOutput("output"),
                grpcclient.InferRequestedOutput("stateN"),
            ]

            result = self.client.infer(
                model_name="silero_vad",
                inputs=inputs,
                outputs=outputs
            )

            prob = float(result.as_numpy("output").flatten()[0])
            self.state = result.as_numpy("stateN")
            return prob

        except Exception as e:
            return None


class TritonMicTest:
    """Triton 기반 마이크 동시 부하 테스트"""

    def __init__(self, url: str = "localhost:9001", num_clients: int = 30,
                 sample_rate: int = 16000, chunk_size: int = 512,
                 device_index: Optional[int] = None):
        self.url = url
        self.num_clients = num_clients
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index

        self.clients: List[TritonVADClient] = []
        self.stats: List[ClientStats] = []
        self.client_queues: List[queue.Queue] = []

        self.running = False
        self.stream = None
        self.threads: List[threading.Thread] = []
        self.stats_lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
        audio = np.clip(audio, -1.0, 1.0)
        for q in self.client_queues:
            try:
                q.put_nowait(audio)
            except queue.Full:
                pass

    def _client_worker(self, client_id: int):
        client = self.clients[client_id]
        stats = self.stats[client_id]
        my_queue = self.client_queues[client_id]

        while self.running:
            try:
                audio = my_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            start = time.perf_counter()
            prob = client.infer(audio)
            latency = (time.perf_counter() - start) * 1000

            if prob is not None:
                with self.stats_lock:
                    stats.inference_count += 1
                    stats.total_latency_ms += latency
                    stats.last_prob = prob
                    stats.is_speaking = prob > 0.5

    def setup(self):
        print(f"\n{'='*60}")
        print(f"  [Triton] 실제 마이크 → {self.num_clients}개 동시 VAD 요청")
        print(f"{'='*60}")

        print("\n사용 가능한 입력 장치:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                marker = " <-- 기본" if i == sd.default.device[0] else ""
                print(f"  [{i}] {d['name']}{marker}")

        if self.device_index is None:
            self.device_index = sd.default.device[0]

        # Triton 서버 확인
        test_client = grpcclient.InferenceServerClient(url=self.url)
        if not test_client.is_server_live():
            raise RuntimeError(f"Triton 서버 연결 실패: {self.url}")
        if not test_client.is_model_ready("silero_vad"):
            raise RuntimeError("silero_vad 모델이 준비되지 않음")
        print(f"\nTriton 서버 연결: {self.url}")

        print(f"{self.num_clients}개 클라이언트 생성 중...")
        for i in range(self.num_clients):
            self.clients.append(TritonVADClient(i, self.url))
            self.stats.append(ClientStats())
            self.client_queues.append(queue.Queue(maxsize=50))
        print("완료")

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            device=self.device_index, channels=1,
            samplerate=self.sample_rate, blocksize=self.chunk_size,
            dtype=np.float32, callback=self._audio_callback
        )
        self.stream.start()
        for i in range(self.num_clients):
            t = threading.Thread(target=self._client_worker, args=(i,))
            t.daemon = True
            t.start()
            self.threads.append(t)
        print(f"\n{self.num_clients}개 클라이언트 시작\n")

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        for t in self.threads:
            t.join(timeout=1.0)
        print("\n완료")

    def display_status(self):
        print("\033[2J\033[H", end="")
        print("=" * 80)
        print(f"  [Triton Server] 실제 마이크 → {self.num_clients}개 동시 VAD 요청")
        print("=" * 80)
        print(f"  서버: {self.url}  |  마이크: 장치 {self.device_index}")
        print("-" * 80)

        total_inferences = 0
        total_latency = 0.0
        avg_prob = 0.0

        with self.stats_lock:
            for s in self.stats:
                total_inferences += s.inference_count
                total_latency += s.total_latency_ms
                avg_prob += s.last_prob

        avg_prob /= max(self.num_clients, 1)
        avg_latency = total_latency / max(total_inferences, 1)

        bar_len = 40
        filled = int(avg_prob * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"\n  평균 VAD 확률: [{bar}] {avg_prob:.3f}")
        print(f"\n  총 추론: {total_inferences:,}  |  평균 레이턴시: {avg_latency:.2f}ms")

        print("\n" + "-" * 80)
        print(f"  {'ID':<4} {'Prob':^8} {'Latency':^10} {'Infers':^10}")
        print("  " + "-" * 36)
        with self.stats_lock:
            for i, s in enumerate(self.stats[:10]):
                avg_lat = s.total_latency_ms / max(s.inference_count, 1)
                print(f"  {i:<4} {s.last_prob:^8.3f} {avg_lat:^10.2f} {s.inference_count:^10}")

        print("\n" + "=" * 80)

    def run(self, duration: Optional[float] = None):
        start_time = time.time()
        try:
            while self.running:
                time.sleep(0.3)
                self.display_status()
                if duration and (time.time() - start_time) > duration:
                    break
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Triton VAD Mic Test")
    parser.add_argument("--url", default="localhost:9001", help="Triton gRPC URL")
    parser.add_argument("--num-clients", type=int, default=30)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    test = TritonMicTest(url=args.url, num_clients=args.num_clients, device_index=args.device)

    signal.signal(signal.SIGINT, lambda s, f: (test.stop(), sys.exit(0)))

    try:
        test.setup()
        test.start()
        test.run(duration=args.duration)
    except Exception as e:
        print(f"오류: {e}")
    finally:
        test.stop()


if __name__ == "__main__":
    main()

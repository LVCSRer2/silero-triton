#!/usr/bin/env python3
"""
실제 마이크 입력 1개를 받아서 N개의 동시 클라이언트로 VAD 서버에 전송하는 테스트
- 마이크에서 오디오를 캡처
- 동일한 오디오를 모든 클라이언트가 동시에 서버로 전송 (N개 마이크 시뮬레이션)
- 각 클라이언트의 VAD 결과를 실시간으로 표시
"""
import numpy as np
import sounddevice as sd
import socket
import struct
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
    """클라이언트 통계"""
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_prob: float = 0.0
    is_speaking: bool = False
    probs_history: deque = field(default_factory=lambda: deque(maxlen=20))


class TCPVADClient:
    """TCP VAD 클라이언트"""

    def __init__(self, client_id: int, host: str, port: int):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connected = True
            return True
        except Exception as e:
            print(f"[Client {self.client_id}] 연결 실패: {e}")
            return False

    def infer(self, audio: np.ndarray) -> Optional[float]:
        if not self.connected:
            return None
        try:
            audio_bytes = audio.astype(np.float32).tobytes()
            header = struct.pack('!I', len(audio_bytes))
            self.socket.send(header + audio_bytes)
            result = self.socket.recv(4)
            if len(result) < 4:
                return None
            return struct.unpack('!f', result)[0]
        except Exception:
            self.connected = False
            return None

    def close(self):
        if self.socket:
            try:
                self.socket.send(struct.pack('!I', 0))
                self.socket.close()
            except:
                pass
        self.connected = False


class RealMicMultiClient:
    """실제 마이크 입력을 N개 클라이언트 모두에게 복제하여 동시 전송"""

    def __init__(self, host: str = 'localhost', port: int = 8001,
                 num_clients: int = 30, sample_rate: int = 16000,
                 chunk_size: int = 512, device_index: Optional[int] = None):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index

        self.clients: List[TCPVADClient] = []
        self.stats: List[ClientStats] = []
        # 각 클라이언트마다 개별 큐 (오디오 복제를 위해)
        self.client_queues: List[queue.Queue] = []

        self.running = False
        self.stream = None
        self.threads: List[threading.Thread] = []
        self.stats_lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        """마이크 콜백: 동일한 오디오를 모든 클라이언트 큐에 복제"""
        if status:
            pass

        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

        # 모든 클라이언트 큐에 동일한 오디오 넣기
        for q in self.client_queues:
            try:
                q.put_nowait(audio)
            except queue.Full:
                pass  # 느린 클라이언트는 드롭

    def _client_worker(self, client_id: int):
        """클라이언트 워커: 자기 큐에서 오디오를 가져와서 서버로 전송"""
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
                    stats.probs_history.append(prob)

    def setup(self, device_indices: Optional[list] = None):
        print(f"\n{'='*60}")
        print(f"  실제 마이크 입력 → {self.num_clients}개 동시 VAD 요청 테스트")
        print(f"{'='*60}")

        print("\n사용 가능한 입력 장치:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                marker = " <-- 기본" if i == sd.default.device[0] else ""
                print(f"  [{i}] {d['name']}{marker}")

        if self.device_index is None:
            self.device_index = sd.default.device[0]
        print(f"\n선택된 장치: [{self.device_index}]")

        print(f"\n{self.num_clients}개 클라이언트 연결 중...")
        connected = 0
        for i in range(self.num_clients):
            client = TCPVADClient(i, self.host, self.port)
            self.clients.append(client)
            self.stats.append(ClientStats())
            self.client_queues.append(queue.Queue(maxsize=50))
            if client.connect():
                connected += 1

        print(f"연결 완료: {connected}/{self.num_clients}")
        if connected == 0:
            raise RuntimeError("서버에 연결된 클라이언트가 없습니다!")
        return connected

    def start(self):
        self.running = True

        self.stream = sd.InputStream(
            device=self.device_index,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self.stream.start()
        print(f"\n마이크 스트림 시작 (16kHz, {self.chunk_size} samples = 32ms)")

        for i in range(self.num_clients):
            thread = threading.Thread(target=self._client_worker, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        print(f"{self.num_clients}개 클라이언트 워커 시작\n")

    def stop(self):
        print("\n종료 중...")
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        for client in self.clients:
            client.close()
        for thread in self.threads:
            thread.join(timeout=1.0)
        print("완료")

    def display_status(self):
        print("\033[2J\033[H", end="")

        print("=" * 80)
        print(f"  실제 마이크 입력 → {self.num_clients}개 동시 VAD 요청")
        print("=" * 80)
        print(f"  서버: {self.host}:{self.port}  |  마이크: 장치 {self.device_index}")
        print("-" * 80)

        total_inferences = 0
        total_latency = 0.0
        speaking_count = 0
        avg_prob = 0.0

        with self.stats_lock:
            for stats in self.stats:
                total_inferences += stats.inference_count
                total_latency += stats.total_latency_ms
                if stats.is_speaking:
                    speaking_count += 1
                avg_prob += stats.last_prob

        avg_prob /= max(self.num_clients, 1)
        overall_avg_latency = total_latency / max(total_inferences, 1)

        bar_len = 40
        filled = int(avg_prob * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"\n  평균 VAD 확률: [{bar}] {avg_prob:.3f}")
        print(f"  음성 감지: {'SPEAKING' if avg_prob > 0.5 else '   quiet'}")
        print(f"\n  총 추론: {total_inferences:,}  |  평균 레이턴시: {overall_avg_latency:.2f}ms")
        print(f"  Speaking 클라이언트: {speaking_count}/{self.num_clients}")

        print("\n" + "-" * 80)
        print("  클라이언트별 상태 (처음 10개):")
        print(f"  {'ID':<4} {'Prob':^8} {'Latency':^10} {'Infers':^10}")
        print("  " + "-" * 36)

        with self.stats_lock:
            for i, stats in enumerate(self.stats[:10]):
                avg_lat = stats.total_latency_ms / max(stats.inference_count, 1)
                prob_indicator = "SPEAK" if stats.is_speaking else ""
                print(f"  {i:<4} {stats.last_prob:^8.3f} {avg_lat:^10.2f} {stats.inference_count:^10} {prob_indicator}")

        print("\n" + "=" * 80)
        print("  Ctrl+C로 종료")

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
    parser = argparse.ArgumentParser(description="Real Mic + N Concurrent VAD Clients")
    parser.add_argument("--host", default="localhost", help="VAD 서버 호스트")
    parser.add_argument("--port", type=int, default=8001, help="VAD 서버 포트")
    parser.add_argument("--num-clients", type=int, default=30, help="동시 클라이언트 수")
    parser.add_argument("--device", type=int, default=None, help="마이크 장치 인덱스")
    parser.add_argument("--duration", type=float, default=None, help="테스트 시간(초)")
    parser.add_argument("--list-devices", action="store_true", help="오디오 장치 목록")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    client = RealMicMultiClient(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients,
        device_index=args.device
    )

    def signal_handler(sig, frame):
        client.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        client.setup()
        client.start()
        client.run(duration=args.duration)
    except Exception as e:
        print(f"오류: {e}")
    finally:
        client.stop()


if __name__ == "__main__":
    main()

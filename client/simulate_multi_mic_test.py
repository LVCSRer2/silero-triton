#!/usr/bin/env python3
"""
실제 마이크 없이 10개의 동시 연결을 시뮬레이션하는 테스트 클라이언트
랜덤 오디오 또는 실제 오디오 파일을 사용하여 VAD 서버 테스트
"""
import numpy as np
import socket
import struct
import threading
import time
import argparse
import signal
import sys
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ClientStats:
    """클라이언트 통계"""
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_prob: float = 0.0
    errors: int = 0


class TCPVADClient:
    """TCP VAD 서버 클라이언트"""

    def __init__(self, host: str = 'localhost', port: int = 8001):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """서버 연결"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def infer(self, audio: np.ndarray) -> float:
        """VAD 추론"""
        audio_bytes = audio.astype(np.float32).tobytes()
        header = struct.pack('!I', len(audio_bytes))
        self.socket.send(header + audio_bytes)

        result = self.socket.recv(4)
        prob = struct.unpack('!f', result)[0]
        return prob

    def close(self):
        """연결 종료"""
        if self.socket:
            try:
                self.socket.send(struct.pack('!I', 0))
                self.socket.close()
            except:
                pass


class AudioGenerator:
    """테스트용 오디오 생성기"""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.phase = 0

    def generate_silence(self) -> np.ndarray:
        """무음 생성"""
        noise = np.random.randn(self.chunk_size) * 0.001
        return noise.astype(np.float32)

    def generate_speech_like(self) -> np.ndarray:
        """음성 유사 신호 생성 (여러 주파수의 사인파 조합)"""
        t = np.arange(self.chunk_size) / self.sample_rate + self.phase

        # 음성 대역 주파수들 조합 (100-4000Hz)
        signal = (
            0.3 * np.sin(2 * np.pi * 150 * t) +  # 기본 주파수
            0.2 * np.sin(2 * np.pi * 300 * t) +
            0.15 * np.sin(2 * np.pi * 500 * t) +
            0.1 * np.sin(2 * np.pi * 1000 * t) +
            0.05 * np.sin(2 * np.pi * 2000 * t)
        )

        # 약간의 노이즈 추가
        signal += np.random.randn(self.chunk_size) * 0.02

        self.phase += self.chunk_size / self.sample_rate

        return (signal * 0.5).astype(np.float32)

    def generate_random_pattern(self) -> np.ndarray:
        """랜덤하게 음성/무음 패턴 생성"""
        if np.random.random() > 0.7:  # 30% 확률로 음성
            return self.generate_speech_like()
        else:
            return self.generate_silence()


class SimulatedMicClient:
    """시뮬레이션 마이크 클라이언트"""

    def __init__(self, client_id: int, host: str, port: int,
                 sample_rate: int = 16000, chunk_size: int = 512):
        self.client_id = client_id
        self.client = TCPVADClient(host, port)
        self.audio_gen = AudioGenerator(sample_rate, chunk_size)
        self.stats = ClientStats()
        self.running = False

    def run(self, duration: Optional[float] = None, mode: str = 'random'):
        """클라이언트 실행"""
        try:
            self.client.connect()
        except Exception as e:
            print(f"[Client {self.client_id}] 연결 실패: {e}")
            return

        self.running = True
        start_time = time.time()
        interval = 0.032  # 32ms (512 samples at 16kHz)

        while self.running:
            if duration and (time.time() - start_time) > duration:
                break

            # 오디오 생성
            if mode == 'speech':
                audio = self.audio_gen.generate_speech_like()
            elif mode == 'silence':
                audio = self.audio_gen.generate_silence()
            else:  # random
                audio = self.audio_gen.generate_random_pattern()

            try:
                infer_start = time.perf_counter()
                prob = self.client.infer(audio)
                latency = (time.perf_counter() - infer_start) * 1000

                self.stats.inference_count += 1
                self.stats.total_latency_ms += latency
                self.stats.last_prob = prob

            except Exception as e:
                self.stats.errors += 1
                if self.running:
                    break

            # 실시간 시뮬레이션을 위한 대기
            time.sleep(max(0, interval - (time.perf_counter() - infer_start)))

        self.client.close()

    def stop(self):
        """정지"""
        self.running = False


class MultiClientSimulator:
    """다중 클라이언트 시뮬레이터"""

    def __init__(self, host: str, port: int, num_clients: int = 10):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.clients: list[SimulatedMicClient] = []
        self.threads: list[threading.Thread] = []
        self.running = False

    def setup(self):
        """설정"""
        for i in range(self.num_clients):
            client = SimulatedMicClient(i, self.host, self.port)
            self.clients.append(client)

    def start(self, duration: Optional[float] = None, mode: str = 'random'):
        """시작"""
        self.running = True

        print(f"\n{self.num_clients}개 클라이언트 시뮬레이션 시작")
        print(f"서버: {self.host}:{self.port}")
        print(f"모드: {mode}")
        if duration:
            print(f"지속시간: {duration}초")
        print()

        for client in self.clients:
            thread = threading.Thread(
                target=client.run,
                args=(duration, mode)
            )
            thread.start()
            self.threads.append(thread)

    def stop(self):
        """정지"""
        self.running = False
        for client in self.clients:
            client.stop()
        for thread in self.threads:
            thread.join(timeout=2.0)

    def display_status(self):
        """상태 표시"""
        print("\033[2J\033[H", end="")

        print("=" * 80)
        print("  Silero VAD - Simulated Multi-Client Test")
        print("=" * 80)
        print(f"  서버: {self.host}:{self.port}  |  클라이언트: {self.num_clients}개")
        print("-" * 80)
        print(f"{'Client':<8} {'Prob':^8} {'Visual':^25} {'Infers':>10} {'Avg ms':>10} {'Errors':>8}")
        print("-" * 80)

        total_inferences = 0
        total_latency = 0.0
        total_errors = 0

        for client in self.clients:
            stats = client.stats
            prob = stats.last_prob
            count = stats.inference_count
            avg_latency = stats.total_latency_ms / max(count, 1)

            total_inferences += count
            total_latency += stats.total_latency_ms
            total_errors += stats.errors

            # 시각화
            bar_len = 20
            filled = int(prob * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(f"  {client.client_id:<6} {prob:>6.3f}  [{bar}]  {count:>10} {avg_latency:>9.2f} {stats.errors:>8}")

        print("-" * 80)
        overall_avg = total_latency / max(total_inferences, 1)
        throughput = total_inferences / max(time.time() - self.start_time, 0.001)
        print(f"  총 추론: {total_inferences:,}  |  평균 레이턴시: {overall_avg:.2f}ms  |  처리량: {throughput:.1f}/s  |  오류: {total_errors}")
        print("=" * 80)
        print("\n  Ctrl+C로 종료")

    def run_with_display(self, duration: Optional[float] = None, mode: str = 'random'):
        """디스플레이와 함께 실행"""
        self.start_time = time.time()
        self.start(duration, mode)

        try:
            while self.running:
                time.sleep(0.3)
                self.display_status()

                # 지속시간 체크
                if duration and (time.time() - self.start_time) > duration + 1:
                    break

                # 모든 스레드 완료 체크
                if all(not t.is_alive() for t in self.threads):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            self.display_status()
            print("\n테스트 완료!")


def main():
    parser = argparse.ArgumentParser(description="Simulated Multi-Client VAD Test")
    parser.add_argument("--host", default="localhost", help="VAD 서버 호스트")
    parser.add_argument("--port", type=int, default=8001, help="VAD 서버 포트")
    parser.add_argument("--num-clients", type=int, default=10, help="동시 클라이언트 수")
    parser.add_argument("--duration", type=float, default=None, help="테스트 지속시간 (초)")
    parser.add_argument("--mode", choices=['random', 'speech', 'silence'], default='random',
                        help="오디오 생성 모드")
    args = parser.parse_args()

    simulator = MultiClientSimulator(
        host=args.host,
        port=args.port,
        num_clients=args.num_clients
    )

    # 시그널 핸들러
    def signal_handler(sig, frame):
        simulator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    simulator.setup()
    simulator.run_with_display(duration=args.duration, mode=args.mode)


if __name__ == "__main__":
    main()

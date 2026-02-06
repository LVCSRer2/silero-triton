#!/usr/bin/env python3
"""
TCP 기반 VAD 서버에 연결하는 10개 마이크 동시 테스트 클라이언트
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
from dataclasses import dataclass
from typing import Optional


@dataclass
class MicStats:
    """마이크 통계"""
    inference_count: int = 0
    total_latency_ms: float = 0.0
    last_prob: float = 0.0
    is_speaking: bool = False
    speech_segments: int = 0


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
        # 오디오 데이터 전송
        audio_bytes = audio.astype(np.float32).tobytes()
        header = struct.pack('!I', len(audio_bytes))
        self.socket.send(header + audio_bytes)

        # 결과 수신
        result = self.socket.recv(4)
        prob = struct.unpack('!f', result)[0]
        return prob

    def close(self):
        """연결 종료"""
        if self.socket:
            # 종료 신호 (길이 0)
            self.socket.send(struct.pack('!I', 0))
            self.socket.close()


class MicrophoneStream:
    """마이크 스트림"""

    def __init__(self, mic_id: int, device_index: Optional[int] = None,
                 sample_rate: int = 16000, chunk_size: int = 512):
        self.mic_id = mic_id
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=100)
        self.stream = None
        self.running = False

    def audio_callback(self, indata, frames, time_info, status):
        """오디오 콜백"""
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            pass

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

    def stop(self):
        """스트림 정지"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """오디오 가져오기"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class MultiMicProcessor:
    """멀티 마이크 프로세서"""

    def __init__(self, server_host: str = 'localhost', server_port: int = 8001,
                 num_mics: int = 10, sample_rate: int = 16000):
        self.server_host = server_host
        self.server_port = server_port
        self.num_mics = num_mics
        self.sample_rate = sample_rate

        self.mic_streams: list[MicrophoneStream] = []
        self.clients: list[TCPVADClient] = []
        self.stats: list[MicStats] = []
        self.threads: list[threading.Thread] = []
        self.running = False
        self.stats_lock = threading.Lock()

    def setup(self, device_indices: Optional[list[int]] = None):
        """설정"""
        # 사용 가능한 장치 출력
        print("\n사용 가능한 입력 장치:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']}")

        # 기본 장치 사용
        if device_indices is None:
            default_device = sd.default.device[0]
            device_indices = [default_device] * self.num_mics
            print(f"\n기본 장치 {default_device}를 {self.num_mics}개 스트림에 사용")

        # 마이크 스트림 및 클라이언트 초기화
        for i in range(self.num_mics):
            device_idx = device_indices[i % len(device_indices)]

            stream = MicrophoneStream(
                mic_id=i,
                device_index=device_idx,
                sample_rate=self.sample_rate
            )
            self.mic_streams.append(stream)

            client = TCPVADClient(self.server_host, self.server_port)
            self.clients.append(client)

            self.stats.append(MicStats())

    def _process_mic(self, mic_id: int):
        """마이크 처리 스레드"""
        stream = self.mic_streams[mic_id]
        client = self.clients[mic_id]
        was_speaking = False

        while self.running:
            audio = stream.get_audio(timeout=0.1)
            if audio is None:
                continue

            try:
                start = time.perf_counter()
                prob = client.infer(audio)
                latency = (time.perf_counter() - start) * 1000

                is_speaking = prob > 0.5

                with self.stats_lock:
                    stats = self.stats[mic_id]
                    stats.inference_count += 1
                    stats.total_latency_ms += latency
                    stats.last_prob = prob
                    stats.is_speaking = is_speaking

                    if is_speaking and not was_speaking:
                        stats.speech_segments += 1

                was_speaking = is_speaking

            except Exception as e:
                if self.running:
                    print(f"[Mic {mic_id}] 오류: {e}")
                    break

    def start(self):
        """시작"""
        self.running = True

        # 서버 연결
        print(f"\n서버 연결 중: {self.server_host}:{self.server_port}")
        for i, client in enumerate(self.clients):
            try:
                client.connect()
                print(f"  Mic {i}: 연결됨")
            except Exception as e:
                print(f"  Mic {i}: 연결 실패 - {e}")
                return False

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
        return True

    def stop(self):
        """정지"""
        print("\n처리 정지 중...")
        self.running = False

        for stream in self.mic_streams:
            stream.stop()

        for client in self.clients:
            try:
                client.close()
            except:
                pass

        for thread in self.threads:
            thread.join(timeout=1.0)

        print("완료")

    def display_status(self):
        """상태 표시"""
        # 화면 클리어
        print("\033[2J\033[H", end="")

        print("=" * 80)
        print("  Silero VAD - Multi-Microphone Test Client (TCP)")
        print("=" * 80)
        print(f"  서버: {self.server_host}:{self.server_port}")
        print("-" * 80)
        print(f"{'Mic':<5} {'Prob':^8} {'Visual':^25} {'Status':^12} {'Infers':>8} {'Avg ms':>8}")
        print("-" * 80)

        total_inferences = 0
        total_latency = 0.0

        with self.stats_lock:
            for i, stats in enumerate(self.stats):
                prob = stats.last_prob
                count = stats.inference_count
                avg_latency = stats.total_latency_ms / max(count, 1)

                total_inferences += count
                total_latency += stats.total_latency_ms

                # 시각화
                bar_len = 20
                filled = int(prob * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)

                if prob > 0.8:
                    status = "🔊 LOUD"
                elif prob > 0.5:
                    status = "🎤 speaking"
                else:
                    status = "   quiet"

                print(f"  {i:<3} {prob:>6.3f}  [{bar}]  {status:<12} {count:>8} {avg_latency:>7.2f}")

        print("-" * 80)
        overall_avg = total_latency / max(total_inferences, 1)
        print(f"  총 추론: {total_inferences:,}  |  평균 레이턴시: {overall_avg:.2f}ms")
        print("=" * 80)
        print("\n  Ctrl+C로 종료")


def main():
    parser = argparse.ArgumentParser(description="Multi-Mic VAD TCP Client")
    parser.add_argument("--host", default="localhost", help="VAD 서버 호스트")
    parser.add_argument("--port", type=int, default=8001, help="VAD 서버 포트")
    parser.add_argument("--num-mics", type=int, default=10, help="마이크 수")
    parser.add_argument("--sample-rate", type=int, default=16000, help="샘플레이트")
    parser.add_argument("--list-devices", action="store_true", help="장치 목록 출력")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    processor = MultiMicProcessor(
        server_host=args.host,
        server_port=args.port,
        num_mics=args.num_mics,
        sample_rate=args.sample_rate
    )

    # 시그널 핸들러
    def signal_handler(sig, frame):
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        processor.setup()
        if processor.start():
            while True:
                time.sleep(0.3)
                processor.display_status()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"오류: {e}")
    finally:
        processor.stop()


if __name__ == "__main__":
    main()

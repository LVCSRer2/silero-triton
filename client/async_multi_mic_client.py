#!/usr/bin/env python3
"""
비동기 방식으로 10개의 마이크 입력을 동시에 처리하는 Triton VAD 클라이언트
더 높은 성능을 위해 asyncio와 gRPC 비동기 클라이언트 사용
"""
import numpy as np
import tritonclient.grpc.aio as grpcclient
import sounddevice as sd
import asyncio
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import signal


@dataclass
class VADState:
    """VAD 상태"""
    h: np.ndarray = field(default_factory=lambda: np.zeros((2, 1, 64), dtype=np.float32))
    c: np.ndarray = field(default_factory=lambda: np.zeros((2, 1, 64), dtype=np.float32))
    speech_prob: float = 0.0
    is_speaking: bool = False


@dataclass
class MicStats:
    """마이크 통계"""
    inference_count: int = 0
    total_latency_ms: float = 0.0
    speech_segments: int = 0
    last_prob: float = 0.0
    probs_history: deque = field(default_factory=lambda: deque(maxlen=50))


class AsyncTritonVADClient:
    """비동기 Triton VAD 클라이언트"""

    def __init__(self, url: str = "localhost:8001", model_name: str = "silero_vad"):
        self.url = url
        self.model_name = model_name
        self.client = None

    async def connect(self):
        """서버 연결"""
        self.client = grpcclient.InferenceServerClient(url=self.url)

        if not await self.client.is_server_live():
            raise ConnectionError(f"Triton 서버 연결 실패: {self.url}")

        if not await self.client.is_model_ready(self.model_name):
            raise RuntimeError(f"모델 준비 안됨: {self.model_name}")

        print(f"✓ Triton 서버 연결: {self.url}")

    async def infer(self, audio: np.ndarray, state: VADState) -> tuple[float, VADState, float]:
        """
        비동기 VAD 추론

        Returns:
            (speech_prob, new_state, latency_ms)
        """
        start_time = time.perf_counter()

        # 입력 준비
        audio_input = audio.reshape(1, -1).astype(np.float32)
        sr_input = np.array([16000], dtype=np.int64)

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

        outputs = [
            grpcclient.InferRequestedOutput("output"),
            grpcclient.InferRequestedOutput("hn"),
            grpcclient.InferRequestedOutput("cn"),
        ]

        result = await self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        speech_prob = float(result.as_numpy("output")[0, 0])
        new_state = VADState(
            h=result.as_numpy("hn"),
            c=result.as_numpy("cn"),
            speech_prob=speech_prob,
            is_speaking=speech_prob > 0.5
        )

        return speech_prob, new_state, latency_ms

    async def close(self):
        if self.client:
            await self.client.close()


class AsyncMicrophoneProcessor:
    """비동기 마이크 프로세서"""

    def __init__(self, mic_id: int, client: AsyncTritonVADClient,
                 device_index: Optional[int] = None,
                 sample_rate: int = 16000, chunk_size: int = 512):
        self.mic_id = mic_id
        self.client = client
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.state = VADState()
        self.stats = MicStats()
        self.running = False
        self.stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 콜백"""
        if status:
            pass  # 오류 무시
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
        try:
            self.audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            pass  # 큐 가득 차면 드롭

    async def start_stream(self):
        """오디오 스트림 시작"""
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

    async def stop_stream(self):
        """오디오 스트림 정지"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    async def process_loop(self):
        """처리 루프"""
        was_speaking = False

        while self.running:
            try:
                audio = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue

            try:
                prob, new_state, latency = await self.client.infer(audio, self.state)
                self.state = new_state

                # 통계 업데이트
                self.stats.inference_count += 1
                self.stats.total_latency_ms += latency
                self.stats.last_prob = prob
                self.stats.probs_history.append(prob)

                # 음성 세그먼트 카운트
                if prob > 0.5 and not was_speaking:
                    self.stats.speech_segments += 1
                was_speaking = prob > 0.5

            except Exception as e:
                if self.running:  # 정상 종료 중이 아닐 때만 출력
                    print(f"[Mic {self.mic_id}] 오류: {e}")


class AsyncMultiMicVAD:
    """비동기 멀티 마이크 VAD 시스템"""

    def __init__(self, triton_url: str = "localhost:8001",
                 num_mics: int = 10, sample_rate: int = 16000):
        self.triton_url = triton_url
        self.num_mics = num_mics
        self.sample_rate = sample_rate
        self.client = AsyncTritonVADClient(triton_url)
        self.processors: list[AsyncMicrophoneProcessor] = []
        self.running = False

    def _list_devices(self):
        """입력 장치 목록"""
        devices = sd.query_devices()
        print("\n사용 가능한 입력 장치:")
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                print(f"  [{i}] {d['name']}")
        return [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]

    async def setup(self, device_indices: Optional[list[int]] = None):
        """시스템 설정"""
        await self.client.connect()

        available = self._list_devices()
        if not available:
            raise RuntimeError("입력 장치 없음!")

        if device_indices is None:
            default_device = sd.default.device[0]
            device_indices = [default_device] * self.num_mics
            print(f"\n기본 장치 {default_device}를 {self.num_mics}개 스트림에 사용")

        for i in range(self.num_mics):
            device_idx = device_indices[i % len(device_indices)]
            processor = AsyncMicrophoneProcessor(
                mic_id=i,
                client=self.client,
                device_index=device_idx,
                sample_rate=self.sample_rate
            )
            self.processors.append(processor)

    async def start(self):
        """모든 처리 시작"""
        self.running = True

        # 스트림 시작
        for p in self.processors:
            await p.start_stream()

        print(f"\n✓ {self.num_mics}개 마이크 스트림 시작")

        # 처리 태스크 생성
        tasks = [asyncio.create_task(p.process_loop()) for p in self.processors]

        # 디스플레이 태스크
        display_task = asyncio.create_task(self._display_loop())

        try:
            await asyncio.gather(*tasks, display_task)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        """모든 처리 정지"""
        self.running = False
        for p in self.processors:
            p.running = False
            await p.stop_stream()
        await self.client.close()

    async def _display_loop(self):
        """상태 표시 루프"""
        while self.running:
            await asyncio.sleep(0.3)
            self._print_status()

    def _print_status(self):
        """상태 출력"""
        # ANSI 이스케이프로 화면 클리어 및 커서 이동
        print("\033[2J\033[H", end="")

        print("=" * 80)
        print("  Silero VAD - Async Multi-Microphone Triton Client")
        print("=" * 80)
        print(f"{'Mic':<5} {'Prob':^8} {'Visual':^25} {'Status':^12} {'Infers':>8} {'Avg ms':>8}")
        print("-" * 80)

        total_inferences = 0
        total_latency = 0.0

        for p in self.processors:
            prob = p.stats.last_prob
            count = p.stats.inference_count
            avg_latency = p.stats.total_latency_ms / max(count, 1)

            total_inferences += count
            total_latency += p.stats.total_latency_ms

            # 시각화 바
            bar_len = 20
            filled = int(prob * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)

            # 상태
            if prob > 0.8:
                status = "🔊 LOUD"
            elif prob > 0.5:
                status = "🎤 speaking"
            else:
                status = "   quiet"

            print(f"  {p.mic_id:<3} {prob:>6.3f}  [{bar}]  {status:<12} {count:>8} {avg_latency:>7.2f}")

        print("-" * 80)
        overall_avg = total_latency / max(total_inferences, 1)
        print(f"  총 추론: {total_inferences:,}  |  평균 레이턴시: {overall_avg:.2f}ms")
        print("=" * 80)
        print("\n  Ctrl+C로 종료")


async def main():
    parser = argparse.ArgumentParser(description="Async Multi-Mic VAD Client")
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC URL")
    parser.add_argument("--num-mics", type=int, default=10, help="마이크 수")
    parser.add_argument("--sample-rate", type=int, default=16000, help="샘플레이트")
    parser.add_argument("--list-devices", action="store_true", help="장치 목록 출력")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    system = AsyncMultiMicVAD(
        triton_url=args.url,
        num_mics=args.num_mics,
        sample_rate=args.sample_rate
    )

    # 시그널 핸들러
    loop = asyncio.get_event_loop()

    async def shutdown():
        print("\n\n종료 중...")
        await system.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(shutdown())
        )

    try:
        await system.setup()
        await system.start()
    except Exception as e:
        print(f"오류: {e}")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())

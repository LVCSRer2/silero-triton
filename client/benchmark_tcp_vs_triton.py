#!/usr/bin/env python3
"""
TCP vs Triton VAD 서버 벤치마크
동시 N개 클라이언트로 레이턴시/처리량 비교
"""
import numpy as np
import socket
import struct
import threading
import time
import argparse
import sys
import tritonclient.grpc as grpcclient
from dataclasses import dataclass, field
from typing import List, Optional


CONTEXT_SIZE = 64  # Silero VAD v5


@dataclass
class ClientResult:
    client_id: int = 0
    inference_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latencies: list = field(default_factory=list)
    errors: int = 0
    last_prob: float = 0.0


class AudioGenerator:
    """테스트용 오디오 생성기"""
    def __init__(self, chunk_size: int = 512, sample_rate: int = 16000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.phase = 0.0

    def speech_like(self) -> np.ndarray:
        t = np.arange(self.chunk_size) / self.sample_rate + self.phase
        signal = (
            0.3 * np.sin(2 * np.pi * 150 * t) +
            0.2 * np.sin(2 * np.pi * 300 * t) +
            0.15 * np.sin(2 * np.pi * 500 * t) +
            0.1 * np.sin(2 * np.pi * 1000 * t) +
            0.05 * np.sin(2 * np.pi * 2000 * t)
        )
        signal += np.random.randn(self.chunk_size) * 0.02
        self.phase += self.chunk_size / self.sample_rate
        return np.clip(signal * 0.5, -1.0, 1.0).astype(np.float32)

    def silence(self) -> np.ndarray:
        return (np.random.randn(self.chunk_size) * 0.001).astype(np.float32)


def run_tcp_client(client_id: int, host: str, port: int,
                   duration: float, result: ClientResult, barrier: threading.Barrier):
    """TCP 클라이언트 벤치마크 워커"""
    gen = AudioGenerator()
    context = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
    except Exception as e:
        result.errors += 1
        return

    # 모든 클라이언트가 준비될 때까지 대기
    barrier.wait()

    deadline = time.time() + duration
    while time.time() < deadline:
        audio = gen.speech_like() if np.random.random() > 0.5 else gen.silence()

        try:
            audio_bytes = audio.astype(np.float32).tobytes()
            header = struct.pack('!I', len(audio_bytes))

            start = time.perf_counter()
            sock.send(header + audio_bytes)
            resp = sock.recv(4)
            latency = (time.perf_counter() - start) * 1000

            if len(resp) == 4:
                prob = struct.unpack('!f', resp)[0]
                result.inference_count += 1
                result.total_latency_ms += latency
                result.min_latency_ms = min(result.min_latency_ms, latency)
                result.max_latency_ms = max(result.max_latency_ms, latency)
                result.latencies.append(latency)
                result.last_prob = prob
            else:
                result.errors += 1
        except Exception:
            result.errors += 1
            break

    try:
        sock.send(struct.pack('!I', 0))
        sock.close()
    except:
        pass


def run_triton_client(client_id: int, url: str,
                      duration: float, result: ClientResult, barrier: threading.Barrier):
    """Triton gRPC 클라이언트 벤치마크 워커"""
    gen = AudioGenerator()
    context = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    state = np.zeros((2, 1, 128), dtype=np.float32)

    try:
        client = grpcclient.InferenceServerClient(url=url)
    except Exception as e:
        result.errors += 1
        return

    # 모든 클라이언트가 준비될 때까지 대기
    barrier.wait()

    deadline = time.time() + duration
    while time.time() < deadline:
        audio = gen.speech_like() if np.random.random() > 0.5 else gen.silence()
        audio = np.clip(audio, -1.0, 1.0)

        # context prepending
        audio_with_context = np.concatenate([context, audio])
        context = audio[-CONTEXT_SIZE:].copy()

        audio_input = audio_with_context.reshape(1, -1)
        sr_input = np.array([16000], dtype=np.int64)

        inputs = [
            grpcclient.InferInput("input", audio_input.shape, "FP32"),
            grpcclient.InferInput("state", state.shape, "FP32"),
            grpcclient.InferInput("sr", sr_input.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(audio_input)
        inputs[1].set_data_from_numpy(state)
        inputs[2].set_data_from_numpy(sr_input)

        outputs = [
            grpcclient.InferRequestedOutput("output"),
            grpcclient.InferRequestedOutput("stateN"),
        ]

        try:
            start = time.perf_counter()
            res = client.infer(model_name="silero_vad", inputs=inputs, outputs=outputs)
            latency = (time.perf_counter() - start) * 1000

            prob = float(res.as_numpy("output").flatten()[0])
            state = res.as_numpy("stateN")

            result.inference_count += 1
            result.total_latency_ms += latency
            result.min_latency_ms = min(result.min_latency_ms, latency)
            result.max_latency_ms = max(result.max_latency_ms, latency)
            result.latencies.append(latency)
            result.last_prob = prob
        except Exception:
            result.errors += 1


def percentile(latencies: list, p: float) -> float:
    if not latencies:
        return 0.0
    sorted_l = sorted(latencies)
    idx = int(len(sorted_l) * p / 100)
    return sorted_l[min(idx, len(sorted_l) - 1)]


def run_benchmark(name: str, worker_fn, num_clients: int, duration: float, **kwargs) -> List[ClientResult]:
    """벤치마크 실행"""
    print(f"\n{'='*60}")
    print(f"  {name} 벤치마크 ({num_clients} 동시 클라이언트, {duration}초)")
    print(f"{'='*60}")

    results = [ClientResult(client_id=i) for i in range(num_clients)]
    barrier = threading.Barrier(num_clients)
    threads = []

    for i in range(num_clients):
        t = threading.Thread(target=worker_fn, args=(i,), kwargs={
            'duration': duration,
            'result': results[i],
            'barrier': barrier,
            **kwargs,
        })
        threads.append(t)

    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.time() - start_time

    # 결과 집계
    total_infers = sum(r.inference_count for r in results)
    total_errors = sum(r.errors for r in results)
    all_latencies = []
    for r in results:
        all_latencies.extend(r.latencies)

    if not all_latencies:
        print(f"  결과 없음 (errors: {total_errors})")
        return results

    avg_lat = sum(all_latencies) / len(all_latencies)
    p50 = percentile(all_latencies, 50)
    p95 = percentile(all_latencies, 95)
    p99 = percentile(all_latencies, 99)
    min_lat = min(all_latencies)
    max_lat = max(all_latencies)
    throughput = total_infers / wall_time

    print(f"\n  결과:")
    print(f"  ─────────────────────────────────────")
    print(f"  총 추론 횟수     : {total_infers:,}")
    print(f"  처리량 (infer/s) : {throughput:,.1f}")
    print(f"  오류             : {total_errors}")
    print(f"  ─────────────────────────────────────")
    print(f"  레이턴시 (ms):")
    print(f"    평균  : {avg_lat:.2f}")
    print(f"    중위  : {p50:.2f}")
    print(f"    P95   : {p95:.2f}")
    print(f"    P99   : {p99:.2f}")
    print(f"    최소  : {min_lat:.2f}")
    print(f"    최대  : {max_lat:.2f}")
    print(f"  ─────────────────────────────────────")

    # 클라이언트별 요약
    print(f"\n  클라이언트별:")
    print(f"  {'ID':>4}  {'추론':>8}  {'평균ms':>8}  {'P95ms':>8}  {'오류':>6}  {'마지막확률':>10}")
    for r in results:
        avg = r.total_latency_ms / max(r.inference_count, 1)
        p95_c = percentile(r.latencies, 95)
        print(f"  {r.client_id:>4}  {r.inference_count:>8,}  {avg:>8.2f}  {p95_c:>8.2f}  {r.errors:>6}  {r.last_prob:>10.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="TCP vs Triton VAD Benchmark")
    parser.add_argument("--tcp-host", default="localhost")
    parser.add_argument("--tcp-port", type=int, default=8001)
    parser.add_argument("--triton-url", default="localhost:9001")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--duration", type=float, default=10.0, help="테스트 지속시간 (초)")
    parser.add_argument("--only", choices=["tcp", "triton"], default=None, help="하나만 테스트")
    args = parser.parse_args()

    print(f"\n  TCP vs Triton VAD 벤치마크")
    print(f"  동시 클라이언트: {args.num_clients}개  |  지속시간: {args.duration}초\n")

    tcp_results = None
    triton_results = None

    if args.only != "triton":
        tcp_results = run_benchmark(
            "TCP (onnx_vad_server)",
            run_tcp_client,
            args.num_clients,
            args.duration,
            host=args.tcp_host,
            port=args.tcp_port,
        )

    if args.only != "tcp":
        triton_results = run_benchmark(
            "Triton gRPC",
            run_triton_client,
            args.num_clients,
            args.duration,
            url=args.triton_url,
        )

    # 비교 요약
    if tcp_results and triton_results:
        tcp_lats = [l for r in tcp_results for l in r.latencies]
        tri_lats = [l for r in triton_results for l in r.latencies]

        if tcp_lats and tri_lats:
            tcp_total = sum(r.inference_count for r in tcp_results)
            tri_total = sum(r.inference_count for r in triton_results)

            print(f"\n{'='*60}")
            print(f"  비교 요약")
            print(f"{'='*60}")
            print(f"  {'':>20}  {'TCP':>12}  {'Triton':>12}  {'차이':>10}")
            print(f"  {'─'*58}")

            tcp_avg = sum(tcp_lats) / len(tcp_lats)
            tri_avg = sum(tri_lats) / len(tri_lats)
            diff_avg = ((tri_avg - tcp_avg) / tcp_avg) * 100

            tcp_p50 = percentile(tcp_lats, 50)
            tri_p50 = percentile(tri_lats, 50)

            tcp_p95 = percentile(tcp_lats, 95)
            tri_p95 = percentile(tri_lats, 95)

            tcp_tps = tcp_total / args.duration
            tri_tps = tri_total / args.duration
            diff_tps = ((tri_tps - tcp_tps) / tcp_tps) * 100

            print(f"  {'총 추론':>20}  {tcp_total:>12,}  {tri_total:>12,}")
            print(f"  {'처리량 (infer/s)':>20}  {tcp_tps:>12,.1f}  {tri_tps:>12,.1f}  {diff_tps:>+9.1f}%")
            print(f"  {'평균 레이턴시 (ms)':>20}  {tcp_avg:>12.2f}  {tri_avg:>12.2f}  {diff_avg:>+9.1f}%")
            print(f"  {'P50 레이턴시 (ms)':>20}  {tcp_p50:>12.2f}  {tri_p50:>12.2f}")
            print(f"  {'P95 레이턴시 (ms)':>20}  {tcp_p95:>12.2f}  {tri_p95:>12.2f}")
            print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

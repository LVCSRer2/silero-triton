#!/usr/bin/env python3
"""
Docker 없이 실행 가능한 ONNX Runtime 기반 Silero VAD TCP 서버
Silero VAD v5 ONNX 모델 인터페이스에 맞춤
"""
import numpy as np
import onnxruntime as ort
import socket
import struct
import threading
import time
import argparse
import signal
import sys
from typing import Dict, Tuple


class VADSession:
    """개별 클라이언트의 VAD 세션 상태"""
    def __init__(self, state_shape=(2, 1, 128), context_size=64):
        # Silero VAD v5의 state 형태
        self.state = np.zeros(state_shape, dtype=np.float32)
        # Silero VAD v5는 이전 청크의 마지막 context_size 샘플을 현재 청크 앞에 붙여야 함
        self.context = np.zeros(context_size, dtype=np.float32)
        self.context_size = context_size
        self.last_access = time.time()


class SileroVADServer:
    """Silero VAD ONNX 서버"""

    def __init__(self, model_path: str, num_threads: int = 4):
        # ONNX Runtime 세션 옵션
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 모델 로드
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # 입출력 이름 확인
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"모델 로드 완료: {model_path}")
        print(f"입력: {self.input_names}")
        print(f"출력: {self.output_names}")

        # 입력 형태 확인하여 state 크기 결정
        # Silero VAD v5: state shape = [2, batch, 128]
        self.state_shape = (2, 1, 128)  # batch=1 고정

        for inp in self.session.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")

        print(f"State shape: {self.state_shape}")

        # 세션 관리
        self.sessions: Dict[str, VADSession] = {}
        self.sessions_lock = threading.Lock()

        # 세션 정리 스레드
        self.cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
        self.cleanup_thread.start()

    def _cleanup_sessions(self):
        """오래된 세션 정리"""
        while True:
            time.sleep(60)
            current_time = time.time()
            with self.sessions_lock:
                expired = [k for k, v in self.sessions.items()
                          if current_time - v.last_access > 300]
                for k in expired:
                    del self.sessions[k]
                if expired:
                    print(f"만료된 세션 {len(expired)}개 정리됨")

    def get_or_create_session(self, session_id: str) -> VADSession:
        """세션 가져오기 또는 생성"""
        with self.sessions_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = VADSession(self.state_shape)
            session = self.sessions[session_id]
            session.last_access = time.time()
            return session

    def infer(self, audio: np.ndarray, session: 'VADSession') -> Tuple[float, np.ndarray]:
        """VAD 추론 - Silero VAD v5 인터페이스 (context 포함)"""
        audio = audio.flatten().astype(np.float32)

        # 오디오를 [-1, 1] 범위로 클램핑
        audio = np.clip(audio, -1.0, 1.0)

        # Silero VAD v5: 이전 context를 현재 청크 앞에 붙임 (512 → 576)
        audio_with_context = np.concatenate([session.context, audio])

        # 다음 추론을 위해 현재 청크의 마지막 context_size 샘플 저장
        session.context = audio[-session.context_size:].copy()

        # 입력 준비
        audio_input = audio_with_context.reshape(1, -1)
        sr = np.array(16000, dtype=np.int64)

        # 추론
        outputs = self.session.run(
            self.output_names,
            {
                'input': audio_input,
                'state': session.state,
                'sr': sr
            }
        )

        # output, stateN
        prob = float(outputs[0].flatten()[0])
        new_state = outputs[1]

        return prob, new_state

    def infer_with_session(self, session_id: str, audio: np.ndarray) -> float:
        """세션 기반 추론 (상태 자동 관리)"""
        session = self.get_or_create_session(session_id)
        prob, session.state = self.infer(audio, session)
        return prob


class TCPVADServer:
    """TCP 기반 VAD 서버"""

    def __init__(self, vad_server: SileroVADServer, host: str = '0.0.0.0', port: int = 8001):
        self.vad = vad_server
        self.host = host
        self.port = port
        self.running = False
        self.server_socket = None

    def handle_client(self, conn: socket.socket, addr):
        """클라이언트 연결 처리"""
        session_id = f"{addr[0]}:{addr[1]}"
        print(f"클라이언트 연결: {session_id}")

        try:
            while self.running:
                # 헤더 읽기: 4바이트 (오디오 길이)
                header = conn.recv(4)
                if not header or len(header) < 4:
                    break

                audio_len = struct.unpack('!I', header)[0]

                if audio_len == 0:
                    break

                # 오디오 데이터 읽기
                audio_data = b''
                while len(audio_data) < audio_len:
                    chunk = conn.recv(min(4096, audio_len - len(audio_data)))
                    if not chunk:
                        break
                    audio_data += chunk

                if len(audio_data) != audio_len:
                    break

                # float32 배열로 변환
                audio = np.frombuffer(audio_data, dtype=np.float32)

                # VAD 추론
                prob = self.vad.infer_with_session(session_id, audio)

                # 결과 전송: 4바이트 float
                conn.send(struct.pack('!f', prob))

        except Exception as e:
            print(f"클라이언트 오류 ({session_id}): {e}")
        finally:
            conn.close()
            print(f"클라이언트 연결 종료: {session_id}")

    def start(self):
        """서버 시작"""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(100)

        print(f"\n{'='*60}")
        print(f"  VAD 서버 시작: {self.host}:{self.port}")
        print(f"{'='*60}\n")

        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                conn, addr = self.server_socket.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"서버 오류: {e}")

    def stop(self):
        """서버 정지"""
        print("\n서버 종료 중...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()


def main():
    parser = argparse.ArgumentParser(description="Silero VAD ONNX Server")
    parser.add_argument("--model", default="model_repository/silero_vad/1/model.onnx",
                        help="ONNX 모델 경로")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8001, help="서버 포트")
    parser.add_argument("--threads", type=int, default=4, help="ONNX 스레드 수")
    args = parser.parse_args()

    # VAD 서버 초기화
    vad = SileroVADServer(args.model, args.threads)
    server = TCPVADServer(vad, args.host, args.port)

    # 시그널 핸들러
    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 서버 시작
    server.start()


if __name__ == "__main__":
    main()

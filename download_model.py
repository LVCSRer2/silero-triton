#!/usr/bin/env python3
"""
Silero VAD ONNX 모델을 GitHub에서 직접 다운로드하는 스크립트
torch 의존성 없음
"""
import os
import urllib.request

SILERO_VAD_ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

def download_silero_vad():
    model_dir = "model_repository/silero_vad/1"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.onnx")

    if os.path.exists(model_path):
        print(f"모델이 이미 존재합니다: {model_path}")
        return model_path

    print(f"Silero VAD ONNX 모델 다운로드 중...")
    print(f"  URL: {SILERO_VAD_ONNX_URL}")

    urllib.request.urlretrieve(SILERO_VAD_ONNX_URL, model_path)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"다운로드 완료: {model_path} ({size_mb:.1f}MB)")
    return model_path

if __name__ == "__main__":
    download_silero_vad()

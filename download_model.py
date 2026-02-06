#!/usr/bin/env python3
"""
Silero VAD 모델을 다운로드하고 ONNX로 변환하는 스크립트
"""
import torch
import os

def download_and_export_silero_vad():
    model_dir = "model_repository/silero_vad/1"
    os.makedirs(model_dir, exist_ok=True)

    # Silero VAD 모델 로드
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True,
        onnx=True
    )

    # ONNX 모델 파일 위치 확인 및 복사
    import shutil
    cache_dir = torch.hub.get_dir()

    # silero-vad 캐시에서 ONNX 모델 찾기
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file == "silero_vad.onnx":
                src = os.path.join(root, file)
                dst = os.path.join(model_dir, "model.onnx")
                shutil.copy2(src, dst)
                print(f"모델 복사 완료: {dst}")
                return dst

    print("ONNX 모델을 찾을 수 없습니다. 직접 내보내기를 시도합니다.")

    # 직접 ONNX로 내보내기
    model_path = os.path.join(model_dir, "model.onnx")

    # Silero VAD 입력 형식
    # input: [batch, samples] - 16kHz 오디오
    # sr: sample rate
    # h, c: LSTM hidden states

    batch_size = 1
    num_samples = 512  # 32ms at 16kHz (Silero VAD chunk size)

    dummy_input = torch.randn(batch_size, num_samples)
    dummy_sr = torch.tensor([16000])

    # 초기 hidden states
    h = torch.zeros(2, batch_size, 64)
    c = torch.zeros(2, batch_size, 64)

    torch.onnx.export(
        model,
        (dummy_input, dummy_sr, h, c),
        model_path,
        input_names=['input', 'sr', 'h', 'c'],
        output_names=['output', 'hn', 'cn'],
        dynamic_axes={
            'input': {0: 'batch', 1: 'samples'},
            'h': {1: 'batch'},
            'c': {1: 'batch'},
            'output': {0: 'batch'},
            'hn': {1: 'batch'},
            'cn': {1: 'batch'}
        },
        opset_version=16
    )

    print(f"ONNX 모델 내보내기 완료: {model_path}")
    return model_path

if __name__ == "__main__":
    download_and_export_silero_vad()

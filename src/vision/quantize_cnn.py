"""Export Fine-Tuned ResNet18 to ONNX and apply static INT8 quantization (Area D)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import datasets

from src.vision.dataset import get_transforms
from src.vision.models import build_transfer_resnet18


class FoodCalibrationReader(CalibrationDataReader):
    """Provides preprocessed Food-101 calibration samples for ONNX static INT8 quantization."""

    def __init__(self, data_dir: Path, input_name: str, num_samples: int = 120) -> None:
        _, eval_tf = get_transforms(224)
        ds = datasets.ImageFolder(data_dir / "train", transform=eval_tf)
        step = max(1, len(ds) // num_samples)
        indices = list(range(0, len(ds), step))[:num_samples]
        self._iter: Iterator[dict[str, np.ndarray]] = (
            {input_name: ds[i][0].unsqueeze(0).numpy().astype(np.float32)} for i in indices
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)


def export_and_quantize_cnn(
    weights_path: Path = Path("weights/vision/resnet18_finetuned.pt"),
    data_dir: Path = Path("data/food101_10class"),
    out_dir: Path = Path("weights/vision"),
    num_classes: int = 10,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = out_dir / "resnet18_fp32.onnx"
    prep_onnx = out_dir / "resnet18_prep.onnx"
    int8_onnx = out_dir / "resnet18_int8.onnx"

    model = build_transfer_resnet18(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(fp32_onnx),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=18,
    )
    print(f"Exported FP32 ONNX model to {fp32_onnx}")

    quant_pre_process(str(fp32_onnx), str(prep_onnx), skip_symbolic_shape=False)
    reader = FoodCalibrationReader(data_dir=data_dir, input_name="input", num_samples=120)

    quantize_static(
        model_input=str(prep_onnx),
        model_output=str(int8_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    if prep_onnx.exists():
        prep_onnx.unlink()
    print(f"Quantized INT8 ONNX model saved to {int8_onnx}")
    return fp32_onnx, int8_onnx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/vision/resnet18_finetuned.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/food101_10class"))
    parser.add_argument("--out-dir", type=Path, default=Path("weights/vision"))
    args = parser.parse_args()
    export_and_quantize_cnn(args.weights, args.data_dir, args.out_dir)

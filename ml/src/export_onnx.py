"""
Export EfficientNet-B0 Model to ONNX for High-Speed Mobile/Cloud CPU Inference
6-Class Clinical Schema (Eczema, Psoriasis, Tinea, Acne, Healthy, Suspicious Lesion)
"""

import os
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

from model import build_model, SkinDiseaseEfficientNetB0


def export_to_onnx(
    checkpoint_path: str = None,
    output_path: str = "models/model.onnx",
    num_classes: int = 6,
    opset_version: int = 17
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[Export] Instantiating model for {num_classes} classes...")
    model = build_model(
        num_classes=num_classes,
        pretrained=(checkpoint_path is None),
        checkpoint_path=checkpoint_path,
        device="cpu"
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[Export] Exporting to {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    # 1. Verify ONNX model integrity
    print("[Export] Verifying ONNX graph validity...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  --> ONNX model check passed successfully!")

    # 2. Check parity against PyTorch output
    print("[Export] Checking numerical parity against PyTorch...")
    with torch.no_grad():
        torch_logits = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    ort_logits = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_logits - ort_logits))
    print(f"  --> Max difference between PyTorch and ONNX: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  --> Parity check PASSED (< 1e-4 tolerance). Model is ready for ultra-fast CPU inference!")
    else:
        print("  --> Warning: Difference exceeded 1e-4 threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export EfficientNet-B0 to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="models/model.onnx", help="Output path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset
    )

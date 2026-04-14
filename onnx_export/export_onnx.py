import os
import torch
from pathlib import Path
import sys
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import BBoxDetector
from config.config import onnx_export_dir, model_checkpoint_dir

def export_onnx(checkpoint_path, onnx_path):
    device = torch.device("cpu")
    
    # Create dummy input
    torch.manual_seed(0)
    dummy_scene = torch.randn(1, 8192, 3)
    dummy_instance = torch.randn(1, 1024, 3)

    # Load model
    model = BBoxDetector().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}")
    
    model.set_onnx_export(True) # Switch to ONNX-safe uniform sampling")


    # Run inference
    with torch.no_grad():
        pytorch_out = model(dummy_scene, dummy_instance)
    print(f"PyTorch output shape: {pytorch_out.shape}")
    
    torch.onnx.export(
        model,
        (dummy_scene, dummy_instance),
        str(onnx_path),
        export_params=True, # store trained weights in the file
        opset_version=18, # opset 18 is well supported by ORT and TRT
        do_constant_folding=True, # optimize constants at export time
        input_names=["scene_points", "instance_points"], # names for inputs in the graph
        output_names=["pred_corners"], # names for outputs
        dynamic_axes={
            # Allow batch size to vary at runtime
            "scene_points": {0: "batch_size"},
            "instance_points": {0: "batch_size"},
            "pred_corners": {0: "batch_size"},
        },
    )
    print(f"Exported to {onnx_path}")

def verify_export(checkpoint_path, onnx_path):
    device = torch.device("cpu")

    # Load and run with ORT
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()
    print(f"ONNX inputs: {[i.name for i in input_name]}")  # sanity check

    # Create dummy input
    torch.manual_seed(0)
    dummy_scene = torch.randn(1, 8192, 3)
    dummy_instance = torch.randn(1, 1024, 3)

    ort_inputs = {
        input_name[0].name: dummy_scene.numpy(),
        input_name[1].name: dummy_instance.numpy()
    }
    ort_out = sess.run(None, ort_inputs)
    ort_out = torch.from_numpy(ort_out[0])

    # Run same input through PyTorch for comparison
    model = BBoxDetector().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.set_onnx_export(True) # Switch to ONNX-safe uniform sampling")
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}")

    # Run inference
    with torch.no_grad():
        pytorch_out = model(dummy_scene, dummy_instance)

    # Compare outputs
    max_diff = (pytorch_out - ort_out).abs().max().item()
    mean_diff = (pytorch_out - ort_out).abs().mean().item()

    print(f"Comparison")
    print(f"PyTorch output shape: {pytorch_out.shape}")
    print(f"ORT output shape:     {ort_out.shape}")
    print(f"Max absolute diff:    {max_diff:.6f}")
    print(f"Mean absolute diff:   {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("Outputs match; export verified")
    else:
        print("Large diff detected; check set_onnx_export was called")

def main():
    onnx_export_dir.mkdir(exist_ok=True)

    for model_name in os.listdir(model_checkpoint_dir):
        checkpoint_path = model_checkpoint_dir / model_name
        onnx_path = onnx_export_dir / model_name.replace(".pth", ".onnx")
        export_onnx(checkpoint_path, onnx_path)

        verify_export(checkpoint_path, onnx_path)


if __name__ == "__main__":
    main()
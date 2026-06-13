import os
import torch
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[Error] 'ultralytics' is not installed. Please run: pip install ultralytics")
    sys.exit(1)

def main():
    # Define working directory paths
    workspace_dir = Path(__file__).parent.parent
    config_path = workspace_dir / "configs" / "data.yaml"
    
    if not config_path.exists():
        print(f"[Error] Configuration file not found at: {config_path}")
        return
        
    print("[Training] Initializing training script...")
    
    # 1. Select device: use Apple Silicon GPU (MPS) if available, fallback to CPU
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("[Device] Apple Silicon GPU (Metal/MPS) detected! Accelerating training on your MacBook M2.")
    else:
        print("[Device] Fallback to CPU execution.")

    # 2. Load model (using yolov8n.pt as baseline weights)
    print("[Model] Loading YOLOv8 nano pre-trained weights...")
    model = YOLO("yolov8n.pt")

    # 3. Train the model
    # Adjust epochs, batch, and imgsize as needed
    print(f"[Training] Starting training on {config_path}...")
    model.train(
        data=str(config_path),
        epochs=50,
        batch=16,
        imgsz=640,
        device=device,
        project=str(workspace_dir / "runs"),
        name="chili_disease_run"
    )

    print("[Training] Training complete! Exporting model to ONNX...")
    
    # 4. Export model to ONNX format
    onnx_path = model.export(format="onnx")
    print(f"[Training] ✓ Successfully exported ONNX model to: {onnx_path}")
    print("[Training] Copy the exported ONNX model and classes.txt to your 'dist/' folder to use in the GUI.")

if __name__ == "__main__":
    main()

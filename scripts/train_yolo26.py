import os
import sys
import shutil
import random
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("[Error] 'ultralytics' is not installed. Please run: pip install ultralytics")
    sys.exit(1)

def split_classification_dataset(src_dir: Path, target_dir: Path, split_ratio: float = 0.8):
    """Splits an unstructured directory of class folders into train and val splits for YOLO Classification."""
    print(f"[Prepare] Splitting raw images in {src_dir} into train/val sets...")
    
    train_dir = target_dir / "train"
    val_dir = target_dir / "val"
    
    # Clean up previous splits if they exist
    if target_dir.exists():
        shutil.rmtree(target_dir)
        
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through class folders
    for class_folder in src_dir.iterdir():
        if not class_folder.is_dir() or class_folder.name.startswith("."):
            continue
            
        class_name = class_folder.name.replace(" ", "_") # Remove spaces for YOLO compatibility
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Gather all image files
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f" └── Class '{class_name}': {len(train_images)} train, {len(val_images)} val")
        
        # Copy files
        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)
            
    print(f"[Prepare] ✓ Classification dataset split created at: {target_dir}")

def check_is_detection(dataset_dir: Path) -> bool:
    """Checks if the dataset contains bounding box text annotations (Object Detection task)."""
    txt_files = list(dataset_dir.glob("**/*.txt"))
    # Filter out configuration files like data.yaml or requirements.txt
    annotation_files = [f for f in txt_files if f.name not in ["data.yaml", "requirements.txt"]]
    return len(annotation_files) > 0

def main():
    script_dir = Path(__file__).parent
    workspace_dir = script_dir.parent
    dataset_dir = workspace_dir / "dataset"
    
    if not dataset_dir.exists():
        print(f"[Error] Dataset directory '{dataset_dir}' does not exist.")
        print("Please run 'python training/scripts/download_dataset.py' to download the dataset first.")
        sys.exit(1)
        
    print("=================================================================")
    print("      YOLO26 Hardware-Adaptive Model Training Script             ")
    print("=================================================================")

    # 1. Device Selection & Hardware-based Hyperparameters
    device = "cpu"
    
    # Defaults (Safe profile for MacBook M2 / CPUs)
    model_scale = "n"        # 'n' = Nano (lightweight, runs fast anywhere)
    batch_size = 16          # Safe batch size to prevent out-of-memory errors
    num_workers = 4          # Standard multi-threading loader count
    enable_amp = False       # Mixed-precision disabled for MPS stability
    
    if torch.backends.mps.is_available():
        device = "mps"
        print("[Device] Apple Silicon GPU (Metal/MPS) detected! Accelerating training on your MacBook M2.")
        
    elif torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] CUDA GPU detected: {gpu_name}")
        
        # Optimize for powerhouse GPU rigs (RTX 4090 / 3090 / 4080 / i9 processor rigs)
        if any(power_gpu in gpu_name for power_gpu in ["4090", "3090", "4080", "A100", "H100"]):
            print("[Device] RTX 4090 / Powerhouse setup detected!")
            print(" └── Scaling up model architecture to YOLO26 Medium (much higher accuracy)")
            print(" └── Maximizing batch size (batch=64) to fully utilize 24GB VRAM")
            print(" └── Increasing dataloader workers (workers=8) to exploit Intel i9 multi-threading")
            print(" └── Enabling Automatic Mixed Precision (AMP=True) to accelerate Tensor Cores")
            
            model_scale = "m"       # YOLO26 Medium
            batch_size = 64
            num_workers = 8
            enable_amp = True
        else:
            print("[Device] Standard CUDA GPU detected.")
            print(" └── Using YOLO26 Small (s) for balanced speed and accuracy.")
            print(" └── Setting batch=32 and workers=4.")
            
            model_scale = "s"       # YOLO26 Small
            batch_size = 32
            num_workers = 4
            enable_amp = True
    else:
        print("[Device] Running on CPU (no hardware acceleration detected).")
        
    print("=================================================================\n")

    # 2. Dataset Scanning & Workflow Selection
    is_detection = check_is_detection(dataset_dir)
    
    if is_detection:
        print("[Task] Object Detection task detected (bounding box files found).")
        config_path = workspace_dir / "configs" / "data.yaml"
        if not config_path.exists():
            print(f"[Error] Configuration file '{config_path}' not found.")
            sys.exit(1)
            
        model_name = f"yolo26{model_scale}.pt"
        print(f"[Model] Loading YOLO26 Detection weights: {model_name}...")
        model = YOLO(model_name)
        
        train_args = {
            "data": str(config_path),
            "epochs": 50,
            "imgsz": 640,
            "batch": batch_size,
            "workers": num_workers,
            "amp": enable_amp,
            "device": device,
            "project": str(workspace_dir / "runs"),
            "name": f"chili_disease_detection_{model_scale}"
        }
    else:
        print("[Task] Image Classification task detected (no bounding box files found).")
        
        # Locate the subfolders (Chili Leaf Disease / Chili Growth Stage)
        subfolders = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        
        if len(subfolders) == 1:
            src_dataset = subfolders[0]
        elif len(subfolders) > 1:
            print("Multiple dataset directories found:")
            for idx, sf in enumerate(subfolders):
                print(f" [{idx}] {sf.name}")
            # Default to the Leaf Disease dataset if found, otherwise choose the first one
            disease_folder = next((sf for sf in subfolders if "disease" in sf.name.lower() or "leaf" in sf.name.lower()), None)
            src_dataset = disease_folder if disease_folder else subfolders[0]
            print(f"Selected category for training: '{src_dataset.name}'")
        else:
            src_dataset = dataset_dir

        target_split_dir = workspace_dir / "split_dataset"
        split_classification_dataset(src_dataset, target_split_dir)
        
        model_name = f"yolo26{model_scale}-cls.pt"
        print(f"[Model] Loading YOLO26 Classification weights: {model_name}...")
        model = YOLO(model_name)
        
        train_args = {
            "data": str(target_split_dir),
            "epochs": 50,
            "imgsz": 224,
            "batch": batch_size,
            "workers": num_workers,
            "amp": enable_amp,
            "device": device,
            "project": str(workspace_dir / "runs"),
            "name": f"chili_disease_classification_{model_scale}"
        }

    # 3. Train the Model
    print(f"\n[Training] Starting training with YOLO26 {model_scale.upper()} for 50 epochs...")
    model.train(**train_args)
    
    print("\n[Export] Training completed! Exporting model to ONNX...")
    
    # 4. Export the resulting model to ONNX for use in the GUI
    try:
        onnx_path = model.export(format="onnx")
        print(f"[Export] ✓ Successfully exported ONNX model to: {onnx_path}")
        print("\n=================================================================")
        print(" ✓ Training and Export complete!")
        print(" Copy the exported ONNX model file to your 'dist/' directory to use in the GUI.")
        print("=================================================================")
    except Exception as e:
        print(f"[Export] ✗ Failed to export to ONNX: {e}")

if __name__ == "__main__":
    main()

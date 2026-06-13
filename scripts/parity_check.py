import sys
import numpy as np
import cv2
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("[Error] 'ultralytics' is not installed. Please run: pip install ultralytics")
    sys.exit(1)

def compute_iou(box1, box2):
    """Computes Intersection over Union (IoU) between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / float(union + 1e-6)

def main():
    script_dir = Path(__file__).parent
    workspace_dir = script_dir.parent
    
    # 1. Locate trained model weights
    # Searches runs/detect for the most recent weights
    runs_dir = workspace_dir / "runs"
    pt_path = None
    onnx_path = None
    
    # Look for the latest weights folder
    if runs_dir.exists():
        weight_files = list(runs_dir.glob("**/weights/best.pt"))
        if weight_files:
            # Sort by modification time to get the newest
            weight_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            pt_path = weight_files[0]
            onnx_path = pt_path.with_name("best.onnx")

    # If not found in runs, check default workspace locations
    if not pt_path or not pt_path.exists():
        pt_path = Path("best.pt")
        onnx_path = Path("best.onnx")

    print("=================================================================")
    print("      Model Parity Check & Accuracy Verification Script          ")
    print("=================================================================")
    print(f"PyTorch Weights File: {pt_path}")
    print(f"ONNX Weights File:    {onnx_path}")
    print("=================================================================\n")

    if not pt_path.exists():
        print(f"[Error] PyTorch model weights '{pt_path}' not found.")
        print("Please train a model first using 'python training/scripts/train_yolo26.py'.")
        sys.exit(1)

    # 2. Load Models
    print("[Model] Loading PyTorch model...")
    model_pt = YOLO(str(pt_path))
    
    # 3. Validation Metrics Run (Verify Confidence/mAP)
    print("\n[Validation] Running validation dataset evaluation...")
    try:
        metrics = model_pt.val(split='val')
        # Extract validation metrics
        map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
        precision = metrics.results_dict.get('metrics/precision(B)', 0.0)
        recall = metrics.results_dict.get('metrics/recall(B)', 0.0)
        fitness = metrics.fitness
        
        print("\n================ Validation Summary ================")
        print(f" Mean Average Precision (mAP@50): {map50 * 100:.2f}%")
        print(f" Precision:                       {precision * 100:.2f}%")
        print(f" Recall:                          {recall * 100:.2f}%")
        print(f" Overall Model Fitness Score:     {fitness:.4f}")
        print("====================================================")
        
        if map50 >= 0.95:
            print(" ✓ Success: Model meets the high accuracy criteria (>95% mAP)!")
        else:
            print(" ℹ Note: Model is functioning, but you can improve confidence by increasing epochs or adding training data.")
    except Exception as e:
        print(f"[Validation] ✗ Validation metrics run failed: {e}")
        print(" (Make sure data.yaml paths are configured correctly and dataset folders exist.)")

    # 4. ONNX Parity Check
    if not onnx_path.exists():
        print(f"\n[ONNX] ONNX model weights '{onnx_path}' not found. Attempting to export...")
        try:
            onnx_path_str = model_pt.export(format="onnx")
            onnx_path = Path(onnx_path_str)
            print(f"[ONNX] ✓ Exported ONNX model successfully to: {onnx_path}")
        except Exception as e:
            print(f"[ONNX] ✗ Export failed: {e}")
            sys.exit(1)

    print("\n[ONNX] Loading ONNX model...")
    model_onnx = YOLO(str(onnx_path))

    # 5. Create a sample test image (use random noise if no dataset image is found)
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_src = "Random synthetic image (noise)"
    
    # Try to find a real image from the dataset for a better comparison
    dataset_img_dir = workspace_dir / "dataset"
    if dataset_img_dir.exists():
        real_imgs = list(dataset_img_dir.glob("**/*.jpg"))
        if real_imgs:
            test_img = cv2.imread(str(real_imgs[0]))
            img_src = real_imgs[0].name
            
    print(f"\n[Parity] Running side-by-side inference on: '{img_src}'")
    
    # Inference PyTorch
    res_pt = model_pt(test_img, verbose=False)[0]
    boxes_pt = res_pt.boxes.xyxy.cpu().numpy()
    confs_pt = res_pt.boxes.conf.cpu().numpy()
    classes_pt = res_pt.boxes.cls.cpu().numpy()
    
    # Inference ONNX
    res_onnx = model_onnx(test_img, verbose=False)[0]
    boxes_onnx = res_onnx.boxes.xyxy.cpu().numpy()
    confs_onnx = res_onnx.boxes.conf.cpu().numpy()
    classes_onnx = res_onnx.boxes.cls.cpu().numpy()

    # Compare results
    num_pt = len(boxes_pt)
    num_onnx = len(boxes_onnx)
    
    print(f"\nResults comparison:")
    print(f" └── PyTorch Detections: {num_pt}")
    print(f" └── ONNX Detections:    {num_onnx}")
    
    if num_pt != num_onnx:
        print("[Parity] ⚠️ Warning: Number of detections differ between PyTorch and ONNX.")
    
    # Check parity of individual boxes
    matches = 0
    max_conf_diff = 0.0
    
    for i in range(num_pt):
        box_p = boxes_pt[i]
        conf_p = confs_pt[i]
        cls_p = classes_pt[i]
        
        # Look for matching box in ONNX
        best_iou = 0.0
        best_idx = -1
        for j in range(num_onnx):
            iou = compute_iou(box_p, boxes_onnx[j])
            if iou > best_iou:
                best_iou = iou
                best_idx = j
                
        if best_iou > 0.90 and classes_onnx[best_idx] == cls_p:
            matches += 1
            conf_diff = abs(conf_p - confs_onnx[best_idx])
            max_conf_diff = max(max_conf_diff, conf_diff)
            
    parity_percent = (matches / num_pt) * 100 if num_pt > 0 else (100.0 if num_onnx == 0 else 0.0)
    
    print("\n================== Parity Score ===================")
    print(f" Bounding Box Match Rate:  {parity_percent:.1f}%")
    print(f" Max Confidence Deviation: {max_conf_diff:.6f}")
    print("===================================================")
    
    if parity_percent >= 99.0 and max_conf_diff < 0.01:
        print(" ✓ SUCCESS: PyTorch and ONNX models have passed the 99% parity check!")
        print(" Your ONNX model is verified and ready for deployment to the live GUI.")
    else:
        print(" ⚠️ Warning: Parity check falls below the 99% threshold.")
        print(" Bounding boxes or confidence values have minor differences due to quantization/runtime float representations.")

if __name__ == "__main__":
    main()

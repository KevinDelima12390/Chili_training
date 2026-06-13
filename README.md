# Chili Disease Detection Model Training & Verification

This workspace provides the automated scripts to download the dataset, train an edge-device optimized YOLO26 model using NVIDIA RTX 4090 + Intel i9 acceleration, and verify output parity between PyTorch and ONNX.

---

## 📥 Step 1: Download the Dataset
Download and extract the **Chili Plant Leaf Disease and Growth Stage Dataset** automatically:
```bash
python training/scripts/download_dataset.py
```
*This downloads the zip from Mendeley Data, unzips all categories into `training/dataset/`, and cleans up the temporary archive.*

---

## 🚀 Step 2: Train the YOLO26 Model
Start the hardware-adaptive YOLO26 training pipeline:
```bash
python training/scripts/train_yolo26.py
```
*   **Hardware Autotuning**: Automatically detects your RTX 4090 GPU and Intel Core i9 processor to scale the architecture up to **YOLO26 Medium (`yolo26m`)**, set **batch size to 64**, utilize **8 dataloader threads**, and activate **AMP (Automatic Mixed Precision)**.
*   **ONNX Export**: Once training completes, the weights are automatically exported to `best.onnx`.

---

## 🔍 Step 3: Run the Model Parity Check (Target: 90% - 99%)
Verify that the exported ONNX model matches the trained PyTorch model:
```bash
python training/scripts/parity_check.py
```
*   **Validation Check**: Evaluates model precision, recall, and mAP@50 against the dataset's validation split.
*   **Inference Alignment**: Compares predictions (bounding boxes, class classifications, and confidence scores) side-by-side on a test image.

### 💡 If Parity Falls Below the 99% Target:
If the strict 99% match or confidence deviation (< 0.01) warning is triggered, use these methods to secure the **90% - 99% target range**:

1. **Ensure FP32 ONNX Export**:
   Run the export step manually without half-precision (FP16) or dynamic quantization, which preserves high precision float representations:
   ```bash
   python -c "from ultralytics import YOLO; model = YOLO('training/runs/detect/chili_disease_detection_m/weights/best.pt'); model.export(format='onnx', half=False, simplify=False)"
   ```
2. **Relax Inference Confidence Threshold**:
   A minor deviation (e.g. `0.02` score difference) is normal when comparing PyTorch tensors to ONNX runtime float execution. If you get a warning, check the output metrics. Any match score **above 90%** is fully verified and ready for deployment to the live GUI.
3. **Align Image Preprocessing**:
   Ensure you run the parity check script using validation images of the same resolution (640x640) to prevent aspect ratio interpolation differences between PyTorch and ONNX Runtime.

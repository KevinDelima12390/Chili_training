# Chili Disease Detection Model Training Workspace

This workspace is designed to organize and streamline the process of retraining and upgrading the plant leaf disease detection models. You can either train using the cloud-based Roboflow platform or run fine-tuning locally on your MacBook M2.

---

## 📥 Dataset Acquisition (Mendeley Data)

Before training the system, you need to acquire the **Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh**. We have provided a zero-dependency script that programmatically downloads and extracts the entire dataset automatically.

### How to Run:
```bash
# Execute the downloader script
python training/scripts/download_dataset.py
```
This script will:
1. Download the full ZIP file from Mendeley Data (`w9mr3vf56s` version 1).
2. Show a real-time progress bar.
3. Automatically extract all images into `training/dataset/`.
4. Safely delete the temporary ZIP archive to save storage space.
5. Display the extracted directory structure.

---

## Method 1: Cloud Training on Roboflow (Recommended)

Since you are already retraining your model on Roboflow, follow this workflow to integrate new models into this live station:

1. **Upload Datasets**: Upload new leaf images (from dataset snapshots or camera captures) to your Roboflow workspace.
2. **Train Model**: Use Roboflow's serverless train system to train a new version of the **Chili Plant Disease** model.
3. **Download & Cache Weights**:
   Use the helper download script located in `scratch/` to pull the latest ONNX weights and class mappings directly from your Roboflow project:
   ```bash
   # Run the download helper script (it pulls from version 1 by default; update the VERSION variable as needed)
   python scratch/download_chili_retrained.py
   ```
4. **Load in Web GUI**:
   Select the **Chili Disease (Retrained v1)** or **Chili V2 (ba7v6)** preset in the dashboard UI. The system will load the ONNX runtime with hardware accelerated CoreML execution.

---

## Method 2: Local YOLOv8 Fine-Tuning (On-Device)

If you prefer to train models directly on your MacBook M2 using local datasets:

### 1. Install Training Dependencies
Install the `ultralytics` package (which installs PyTorch, torchvision, and training utilities):
```bash
./venv/bin/pip install ultralytics
```

### 2. Prepare Your Dataset
Structure your dataset folder under `training/dataset/` as follows:
```text
dataset/
├── data.yaml          # Class names and folder paths configuration
├── train/
│   ├── images/        # .jpg/.png images
│   └── labels/        # YOLO format text files (.txt)
└── val/
    ├── images/
    └── labels/
```

### 3. Run the Training Script
Run the local training script:
```bash
python training/scripts/train_local.py
```
*Note: PyTorch will automatically use the Apple Silicon GPU via Metal Performance Shaders (MPS) if available, accelerating the training process on your M2 chip.*

### 4. Export the Model to ONNX
After training, convert the resulting `.pt` model file to the ONNX format required by the Web GUI:
```bash
python -c "from ultralytics import YOLO; model = YOLO('training/runs/detect/train/weights/best.pt'); model.export(format='onnx')"
```
Copy the exported `best.onnx` weights into the `dist/` directory as `weights.onnx` to make it selectable in the dashboard.

---

## Method 3: Local YOLO26 Fine-Tuning (Edge-Device Optimized)

For the absolute best performance on resource-constrained edge devices (like your MacBook M2 or mobile architectures), we recommend training with the **YOLO26** model family.

### Why YOLO26?
* **Natively NMS-Free**: YOLO26 eliminates post-processing Non-Maximum Suppression (NMS) calculations from the detection head, significantly reducing CPU latency.
* **M2 CoreML Friendly**: Its lighter architecture compiles beautifully onto Apple Silicon Neural Engine (ANE) partitions without hitting dimension capability ceilings.

### How to Run:
```bash
# Execute the YOLO26 training script
python training/scripts/train_yolo26.py
```
This script will:
1. Automatically scan the Mendeley dataset to check if it's an **Object Detection** or **Image Classification** dataset.
2. If it is a Classification dataset, it splits the folders into `train/val` sets (80/20 ratio) automatically.
3. Automatically load the respective YOLO26 model (`yolo26n.pt` for detection or `yolo26n-cls.pt` for classification).
4. Run the training loop using **MacBook M2 GPU Hardware Acceleration (Metal/MPS)**.
5. Export the trained model directly into **ONNX format** when training completes.

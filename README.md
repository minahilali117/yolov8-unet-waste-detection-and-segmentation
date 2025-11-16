# YOLOv8 + U-Net Waste Detection and Segmentation

## Deep Learning for Perception (CS4045) - Project Part 1

### Group Members
- **Minahil Ali** - 22i-0849
- **Ayaan Khan** - 22i-0832

---

## Project Overview

This project focuses on waste object detection and segmentation using the TACO (Trash Annotations in Context) dataset. We combine:
- **YOLOv8** for object-level detection
- **U-Net** for pixel-level semantic segmentation

The goal is to identify and segment recyclable and non-recyclable materials from real-world waste images, contributing to environmental sustainability applications.

---

## Dataset

**TACO (Trash Annotations in Context)**
- 1,500+ real-world waste images
- 60 labeled classes (plastic, paper, metal, glass, etc.)
- COCO-format bounding boxes and segmentation masks
- Dataset Link: [Kaggle - TACO Trash Dataset](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)

For this project, we use only the **5 most frequent classes (IDs 0-4)** to create a focused subset.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/minahilali117/yolov8-unet-waste-detection-and-segmentation.git
cd yolov8-unet-waste-detection-and-segmentation
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using conda
conda create -n waste_detection python=3.9
conda activate waste_detection

# OR using venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download TACO Dataset

**Option 1: Using Kaggle API (Recommended)**
```bash
# Install Kaggle API
pip install kaggle

# Setup Kaggle credentials (download kaggle.json from your Kaggle account)
# Place kaggle.json in: ~/.kaggle/ (Linux/Mac) or C:\Users\<YourUser>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d kneroma/tacotrashdataset
unzip tacotrashdataset.zip -d data/
```

**Option 2: Manual Download**
1. Visit: https://www.kaggle.com/datasets/kneroma/tacotrashdataset
2. Download the dataset
3. Extract to `data/` folder in project root

### 5. Run the Notebook
```bash
jupyter notebook waste_detection_segmentation.ipynb
```

---

## Project Structure
```
yolov8-unet-waste-detection-and-segmentation/
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── waste_detection_segmentation.ipynb  # Main notebook
├── data/                               # TACO dataset (gitignored)
├── runs/                               # Training outputs (gitignored)
│   ├── detect/                         # YOLOv8 results
│   └── segment/                        # U-Net results
└── weights/                            # Saved model checkpoints (gitignored)
```

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- ~10GB disk space for dataset and models

---

## Project Components

### Milestone 1: Project Setup & Environment ✓
- GitHub repository initialization
- Dependencies configuration
- Development environment setup

### Milestone 2: Exploratory Data Analysis (EDA) ✓
- Dataset statistics and class distribution
- Visualization of images and annotations
- Class imbalance analysis
- Top 5 class selection (IDs: 0, 1, 2, 3, 4)

### Milestone 3: Data Augmentation & Preprocessing ✓
- **Baseline (no_aug)**: No augmentation
- **Moderate (aug_v1)**: Horizontal flip, rotation (±15°), color jitter, Gaussian blur
- **Aggressive (aug_v2)**: Extended rotations (±30°), stronger color jitter, elastic transforms
- Custom PyTorch Dataset classes
- 80/20 train/validation split

### Milestone 4: YOLOv8 Object Detection ✓
- COCO to YOLO format conversion
- YOLOv8n training (50 epochs, seed=42)
- **Baseline**: mAP@50 = 0.45, mAP@50-95 = 0.28
- **With Augmentation**: mAP@50 = 0.52 (+15.6%), mAP@50-95 = 0.33 (+17.9%)
- Confusion matrix, training curves, inference visualizations
- Outputs: `data/yolo_experiments.csv`, `runs/yolo/`

### Milestone 5: U-Net Semantic Segmentation ✓
- Custom U-Net architecture (31M parameters, 256×256 input)
- 4 loss function experiments:
  - **BCE+Dice (best)**: IoU=0.42, Dice=0.59
  - **Dice Only**: IoU=0.40, Dice=0.57
  - **Focal Loss**: IoU=0.38, Dice=0.55
  - **BCE Only**: IoU=0.36, Dice=0.53
- Per-class metrics across all classes
- Segmentation overlays, training curves
- Outputs: `results/unet_metrics.csv`, `results/unet_per_class_metrics.csv`, `runs/unet/`, `models/unet/`

### Milestone 6: Model Comparison & Discussion ✓
- **Quantitative Comparison**:
  - YOLO: Faster (real-time), smaller (3M params), better for counting/localization
  - U-Net: More precise boundaries, better for material analysis, 31M params
- **Failure Case Analysis**: Occlusion, small objects, class confusion
- **Dataset Challenges**: Class imbalance, annotation quality, scale variation
- **Recommendations**: 16 techniques (focal loss, TTA, ensemble methods, multi-scale training)
- **Future Work**: Expand to 60 classes, instance segmentation, edge deployment

---

## Results Summary

| Model | Architecture | Params | Input Size | Best Metric | Training Time |
|-------|-------------|---------|-----------|-------------|---------------|
| YOLOv8n | Detection | 3.0M | 640×640 | mAP@50: 0.52 | ~2 hours |
| U-Net | Segmentation | 31.0M | 256×256 | IoU: 0.42 | ~3 hours |

**Key Findings**:
- Augmentation improves YOLO by 15-18%
- BCE+Dice loss outperforms other U-Net variants
- Both models struggle with minority classes (class imbalance)
- YOLO excels at speed, U-Net at boundary precision
- Complementary strengths suggest ensemble potential

---

## Reproducibility

### Quick Start (One-Command Reproduction)
```bash
# Linux/Mac
chmod +x run.sh
./run.sh

# Windows
run.bat
```

### Manual Reproduction
All experiments use fixed random seeds documented in `seeds.txt`:
- `PYTHONHASHSEED=0` (environment variable)
- `numpy.random.seed(42)`
- `random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`

### Files for Reproducibility
- `seeds.txt`: Complete seed documentation
- `run.sh`: Automated reproduction script (Linux/Mac)
- `run.bat`: Automated reproduction script (Windows)
- `requirements.txt`: Exact dependency versions
- `waste_detection_segmentation.ipynb`: All code cells with detailed comments

---

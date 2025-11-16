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

1. **Exploratory Data Analysis (EDA)**
   - Dataset statistics and class distribution
   - Visualization of images and annotations
   - Class imbalance analysis

2. **Data Preprocessing**
   - Filtering to top 5 classes
   - Data augmentation (flips, rotations, color jitter, mosaic)
   - COCO format validation

3. **YOLOv8 Object Detection**
   - Training on filtered subset
   - Evaluation: Precision, Recall, mAP@50, mAP@50-95
   - Inference visualization

4. **U-Net Semantic Segmentation**
   - Custom U-Net architecture
   - Training on filtered subset
   - Evaluation: IoU, Dice Coefficient
   - Mask overlay visualization

5. **Comparison & Discussion**
   - YOLO vs U-Net analysis
   - Challenges and limitations
   - Sustainability and IoT applications

---

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- `PYTHONHASHSEED=0`
- `numpy.random.seed(42)`
- `random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`

---

## License

This project is for educational purposes as part of CS4045 coursework.

---

## Contact

For questions or issues:
- Minahil Ali: 22i-0849
- Ayaan Khan: 22i-0832

#!/bin/bash
################################################################################
# YOLOv8 + U-Net Waste Detection and Segmentation
# One-Command Reproduction Script (Linux/Mac)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "WASTE DETECTION & SEGMENTATION - AUTOMATED REPRODUCTION"
echo "================================================================================"
echo ""
echo "Project: YOLOv8 + U-Net Waste Detection and Segmentation"
echo "Course: CS4045 Deep Learning for Perception"
echo "Team: Minahil Ali (22i-0849), Ayaan Khan (22i-0832)"
echo ""
echo "================================================================================"
echo "STEP 1: Environment Setup"
echo "================================================================================"

# Set reproducibility seed
export PYTHONHASHSEED=0
echo "✓ Set PYTHONHASHSEED=0 for deterministic hashing"

# Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "================================================================================"
echo "STEP 2: Installing Dependencies"
echo "================================================================================"
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ All dependencies installed"

# Verify installations
echo ""
echo "================================================================================"
echo "STEP 3: Verifying Installations"
echo "================================================================================"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import ultralytics; print(f'✓ Ultralytics {ultralytics.__version__}')"
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'✓ CUDA Available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'✓ CUDA Device: {torch.cuda.get_device_name(0)}')"
else
    echo "⚠ CUDA not available - will use CPU (slower)"
fi

# Create output directories
echo ""
echo "================================================================================"
echo "STEP 4: Creating Output Directories"
echo "================================================================================"
mkdir -p data/taco_subset/{train,val}
mkdir -p data/yolo_dataset/{train,val}
mkdir -p results
mkdir -p runs/yolo
mkdir -p runs/unet
mkdir -p models/unet
echo "✓ All directories created"

# Download TACO dataset (if not present)
echo ""
echo "================================================================================"
echo "STEP 5: Dataset Verification"
echo "================================================================================"
if [ ! -d "data/TACO" ]; then
    echo "⚠ TACO dataset not found in data/TACO/"
    echo "Please download TACO dataset manually:"
    echo "  1. Visit: http://tacodataset.org/"
    echo "  2. Download and extract to data/TACO/"
    echo "  3. Re-run this script"
    exit 1
else
    echo "✓ TACO dataset found"
    num_images=$(find data/TACO -name "*.jpg" | wc -l)
    echo "  - Total images: $num_images"
fi

# Run Jupyter notebook
echo ""
echo "================================================================================"
echo "STEP 6: Running Experiments"
echo "================================================================================"
echo ""
echo "Choose execution method:"
echo "  1. Run in Jupyter Notebook (interactive)"
echo "  2. Execute all cells automatically (non-interactive)"
echo ""
read -p "Enter choice [1/2]: " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Starting Jupyter Notebook..."
    echo "✓ Open waste_detection_segmentation.ipynb and run all cells"
    jupyter notebook waste_detection_segmentation.ipynb
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Executing notebook automatically..."
    jupyter nbconvert --to notebook --execute --inplace waste_detection_segmentation.ipynb
    echo "✓ Notebook executed successfully"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Summary
echo ""
echo "================================================================================"
echo "EXECUTION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - YOLO experiments: data/yolo_experiments.csv"
echo "  - U-Net metrics: results/unet_metrics.csv"
echo "  - Per-class metrics: results/unet_per_class_metrics.csv"
echo "  - YOLO outputs: runs/yolo/"
echo "  - U-Net outputs: runs/unet/"
echo "  - Trained models: models/unet/"
echo ""
echo "To view results:"
echo "  - Open waste_detection_segmentation.ipynb in Jupyter"
echo "  - Check visualization cells"
echo "  - Review metrics tables"
echo ""
echo "================================================================================"
echo "Thank you for using this reproduction script!"
echo "================================================================================"

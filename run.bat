@echo off
REM =============================================================================
REM YOLOv8 + U-Net Waste Detection and Segmentation
REM One-Command Reproduction Script (Windows)
REM =============================================================================

setlocal enabledelayedexpansion

echo ================================================================================
echo WASTE DETECTION ^& SEGMENTATION - AUTOMATED REPRODUCTION
echo ================================================================================
echo.
echo Project: YOLOv8 + U-Net Waste Detection and Segmentation
echo Course: CS4045 Deep Learning for Perception
echo Team: Minahil Ali (22i-0849), Ayaan Khan (22i-0832)
echo.
echo ================================================================================
echo STEP 1: Environment Setup
echo ================================================================================

REM Set reproducibility seed
set PYTHONHASHSEED=0
echo %CHECKMARK% Set PYTHONHASHSEED=0 for deterministic hashing

REM Create virtual environment (optional)
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo %CHECKMARK% Virtual environment created
) else (
    echo %CHECKMARK% Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo %CHECKMARK% Virtual environment activated

REM Install dependencies
echo.
echo ================================================================================
echo STEP 2: Installing Dependencies
echo ================================================================================
python -m pip install --upgrade pip
pip install -r requirements.txt
echo %CHECKMARK% All dependencies installed

REM Verify installations
echo.
echo ================================================================================
echo STEP 3: Verifying Installations
echo ================================================================================
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import ultralytics; print(f'✓ Ultralytics {ultralytics.__version__}')"
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')"

REM Check CUDA availability
python -c "import torch; print(f'✓ CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if %errorlevel% equ 0 (
    python -c "import torch; print(f'✓ CUDA Device: {torch.cuda.get_device_name(0)}')"
) else (
    echo ⚠ CUDA not available - will use CPU (slower)
)

REM Create output directories
echo.
echo ================================================================================
echo STEP 4: Creating Output Directories
echo ================================================================================
if not exist "data\taco_subset\train" mkdir data\taco_subset\train
if not exist "data\taco_subset\val" mkdir data\taco_subset\val
if not exist "data\yolo_dataset\train" mkdir data\yolo_dataset\train
if not exist "data\yolo_dataset\val" mkdir data\yolo_dataset\val
if not exist "results" mkdir results
if not exist "runs\yolo" mkdir runs\yolo
if not exist "runs\unet" mkdir runs\unet
if not exist "models\unet" mkdir models\unet
echo %CHECKMARK% All directories created

REM Download TACO dataset (if not present)
echo.
echo ================================================================================
echo STEP 5: Dataset Verification
echo ================================================================================
if not exist "data\TACO" (
    echo ⚠ TACO dataset not found in data\TACO\
    echo Please download TACO dataset manually:
    echo   1. Visit: http://tacodataset.org/
    echo   2. Download and extract to data\TACO\
    echo   3. Re-run this script
    pause
    exit /b 1
) else (
    echo %CHECKMARK% TACO dataset found
    for /f %%i in ('dir /s /b data\TACO\*.jpg ^| find /c /v ""') do set num_images=%%i
    echo   - Total images: !num_images!
)

REM Run Jupyter notebook
echo.
echo ================================================================================
echo STEP 6: Running Experiments
echo ================================================================================
echo.
echo Choose execution method:
echo   1. Run in Jupyter Notebook (interactive)
echo   2. Execute all cells automatically (non-interactive)
echo.
set /p choice="Enter choice [1/2]: "

if "!choice!"=="1" (
    echo.
    echo Starting Jupyter Notebook...
    echo %CHECKMARK% Open waste_detection_segmentation.ipynb and run all cells
    jupyter notebook waste_detection_segmentation.ipynb
) else if "!choice!"=="2" (
    echo.
    echo Executing notebook automatically...
    jupyter nbconvert --to notebook --execute --inplace waste_detection_segmentation.ipynb
    echo %CHECKMARK% Notebook executed successfully
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

REM Summary
echo.
echo ================================================================================
echo EXECUTION COMPLETE
echo ================================================================================
echo.
echo Results saved to:
echo   - YOLO experiments: data\yolo_experiments.csv
echo   - U-Net metrics: results\unet_metrics.csv
echo   - Per-class metrics: results\unet_per_class_metrics.csv
echo   - YOLO outputs: runs\yolo\
echo   - U-Net outputs: runs\unet\
echo   - Trained models: models\unet\
echo.
echo To view results:
echo   - Open waste_detection_segmentation.ipynb in Jupyter
echo   - Check visualization cells
echo   - Review metrics tables
echo.
echo ================================================================================
echo Thank you for using this reproduction script!
echo ================================================================================

pause

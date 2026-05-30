# Installation

## Prerequisites

- Python 3.10 or later
- Git
- ~2 GB disk space (for dataset and model weights)

## Environment Setup

=== "Windows (Miniconda)"

    ```bash
    conda create -n anomaly_detection python=3.11 -y
    conda activate anomaly_detection

    # PyTorch with CUDA (if NVIDIA GPU)
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

    # PyTorch CPU-only
    conda install pytorch torchvision cpuonly -c pytorch -y
    ```

=== "macOS (venv)"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision
    ```

## Install Project Dependencies

```bash
pip install -r requirements.txt
```

## Download the Dataset

1. Download the **metal_nut.zip** file from [Google Drive](https://drive.google.com/file/d/1BdOOGCY9gmLVa7nlwEcci7COs_htwnXq/view?usp=sharing).
2. Extract the zip and place the `metal_nut` folder inside the `data/` directory so the structure looks like:

```
data/
└── metal_nut/
    ├── train/
    │   └── good/        # ~220 normal images
    ├── test/
    │   ├── good/        # normal test images
    │   ├── bent/         # bent defects
    │   ├── color/        # color defects
    │   ├── flip/         # orientation defects
    │   └── scratch/      # scratch defects
    └── ground_truth/     # pixel-level masks
```

## Train the Model

PatchCore builds a feature memory bank from the "good" images. This is a one-time step:

```bash
python -m anomaly_detection.train
```

This takes approximately 2–5 minutes depending on your hardware. The model checkpoint is saved to `models/patchcore/`.

## Verify Installation

```bash
# Check imports
python -c "from anomaly_detection.inference import DefectDetector; print('OK')"

# Launch the app
streamlit run anomaly_detection/app.py
```

The Streamlit app should open at `http://localhost:8501`.

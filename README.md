# 🔍 Anomaly Detection

Anomaly-based defect detection system for industrial quality control, built with PatchCore and Streamlit.

Part of the **Building AI-Powered Defect Detection Systems for Industrial Quality Control** course.

## Quick Start

```bash
# Install PyTorch first (see docs for CUDA/MPS options)
pip install torch torchvision

# Install project dependencies
pip install -r requirements.txt

# Train PatchCore on MVTec Metal Nut (one-time, ~2 min)
python -m anomaly_detection.train

# Launch the inspection app
streamlit run anomaly_detection/app.py
```

## Project Structure

```
anomaly_detection/
├── anomaly_detection/     # Main Python package
│   ├── app.py             # Streamlit frontend
│   ├── acquisition.py     # Camera simulator
│   ├── inference.py       # PatchCore model inference
│   ├── preprocessing.py   # Image transforms
│   ├── train.py           # Model training script
│   └── utils.py           # Config, logging, paths
├── .streamlit/            # Streamlit theme config
├── data/                  # MVTec dataset (gitignored)
├── models/                # Saved model checkpoints
├── docs/                  # MkDocs documentation source
├── tests/                 # Unit tests
├── .github/workflows/     # CI/CD pipeline
├── Dockerfile             # Container deployment
└── mkdocs.yml             # Docs configuration
```

## Documentation

```bash
mkdocs serve    # Preview at http://localhost:8000
```

## License

This project is for educational purposes as part of the Building AI-Powered Defect Detection Systems for Industrial Quality Control course.

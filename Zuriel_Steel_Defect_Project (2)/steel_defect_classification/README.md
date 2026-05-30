# Steel Defect Classification

A CNN-based steel defect classification system built with PyTorch and Streamlit.

## Setup

```bash
pip install -r requirements.txt
```

Place your dataset in `data/steel_defect/` with one subdirectory per class:

```
data/steel_defect/
├── no_defect/
├── defect_1/
├── defect_2/
├── defect_3/
└── defect_4/
```

## Placeholders

Complete the following placeholders in order:

### Workshop 1 — Data Preparation

| ID | File | Task |
|----|------|------|
| `PREPROCESS-1` | `steel_defect/preprocessing.py` | Build training transform pipeline |
| `PREPROCESS-2` | `steel_defect/preprocessing.py` | Build validation transform pipeline |
| `DATA-1` | `steel_defect/dataset.py` | Scan class directories into (path, label) list |
| `DATA-2` | `steel_defect/dataset.py` | Implement Dataset `__getitem__` |
| `DATA-3` | `steel_defect/dataset.py` | Create train/val/test splits |

### Workshop 2 — Model Training

| ID | File | Task |
|----|------|------|
| `MODEL-1` | `steel_defect/model.py` | Define CNN layers in `__init__` |
| `MODEL-2` | `steel_defect/model.py` | Implement `forward()` |
| `TRAIN-1` | `steel_defect/train.py` | Set up loss function and optimizer |
| `TRAIN-2` | `steel_defect/train.py` | Implement training epoch |
| `TRAIN-3` | `steel_defect/train.py` | Implement validation epoch |
| `TRAIN-4` | `steel_defect/train.py` | Save best model checkpoint |

### Workshop 3 — Inference

| ID | File | Task |
|----|------|------|
| `INFER-1` | `steel_defect/inference.py` | Load model from checkpoint |
| `INFER-2` | `steel_defect/inference.py` | Implement predict() |
| `APP-1` | `steel_defect/app.py` | Wire prediction into Streamlit display |

## Running

```bash
# Train the model
python -m steel_defect.train --epochs 20

# Launch the app
streamlit run steel_defect/app.py

# Run tests
pytest tests/
```

## Solutions

Completed implementations are in the `solutions/` folder.

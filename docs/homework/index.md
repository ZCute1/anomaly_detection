# Steel Defect Classification — Homework

## Overview

Build a complete image classification pipeline from scratch using PyTorch.
You receive a fully scaffolded project — Streamlit frontend, Grad-CAM visualization,
logging, configuration — and fill in **14 numbered placeholders** that implement
the core ML logic: data loading, preprocessing, model definition, training, and inference.

## Project Structure

```
steel_defect_classification/
├── steel_defect/
│   ├── utils.py            ← Constants, paths, device (provided)
│   ├── gradcam.py          ← Grad-CAM visualization (provided)
│   ├── preprocessing.py    ← PREPROCESS-1, PREPROCESS-2
│   ├── dataset.py          ← DATA-1, DATA-2, DATA-3
│   ├── model.py            ← MODEL-1, MODEL-2
│   ├── train.py            ← TRAIN-1, TRAIN-2, TRAIN-3, TRAIN-4
│   ├── inference.py        ← INFER-1, INFER-2
│   └── app.py              ← APP-1
├── solutions/              ← Reference implementations (don't peek early!)
├── tests/                  ← Automated tests for your code
├── data/steel_defect/      ← Dataset (class subdirectories)
├── mvtec_walkthrough.ipynb ← Guided notebook (do this first!)
└── requirements.txt
```

## Dataset

The dataset is organized as one subdirectory per class:

```
data/steel_defect/
├── no_defect/    ← Class 0
├── defect_1/     ← Class 1
├── defect_2/     ← Class 2
├── defect_3/     ← Class 3
└── defect_4/     ← Class 4
```

These are defined in `utils.py` as `CLASS_NAMES` and `NUM_CLASSES`.

## Setup

```bash
cd steel_defect_classification
pip install -r requirements.txt
```

Make sure you have the dataset placed in `data/steel_defect/` before starting.

## Placeholder Format

Every placeholder looks like this:

```python
# ┌──────────────────────────────────────────────┐
# │  PLACEHOLDER-N: Write your code below        │
# └──────────────────────────────────────────────┘
raise NotImplementedError("PLACEHOLDER-N: Description")
```

Delete the `raise` line and replace it with your implementation.
The docstring above each placeholder describes exactly what to write.

## Walkthrough Notebook

Before starting the homework, work through `mvtec_walkthrough.ipynb`.
It covers every concept you need but on a different dataset (MVTec metal nut)
with a different architecture (2-block CNN instead of 3-block).
You cannot copy-paste from the notebook, but the patterns are identical.

## Completion Order

Each step builds on the previous one. Complete them in order:

| Step | File | Placeholders | What You Build |
|------|------|-------------|----------------|
| [Step 1: Preprocessing](step-1-preprocessing.md) | `preprocessing.py` | PREPROCESS-1, PREPROCESS-2 | Image transform pipelines |
| [Step 2: Dataset](step-2-dataset.md) | `dataset.py` | DATA-1, DATA-2, DATA-3 | Data loading and splitting |
| [Step 3: Model](step-3-model.md) | `model.py` | MODEL-1, MODEL-2 | CNN architecture |
| [Step 4: Training](step-4-training.md) | `train.py` | TRAIN-1 .. TRAIN-4 | Training loop |
| [Step 5: Inference](step-5-inference.md) | `inference.py` | INFER-1, INFER-2 | Model loading and prediction |
| [Step 6: App](step-6-app.md) | `app.py` | APP-1 | Streamlit display |

## Verification Checkpoints

Each step has its own test file. Run them as you go:

```bash
pytest tests/test_step1_preprocessing.py -v   # After Step 1
pytest tests/test_step2_dataset.py -v          # After Step 2
pytest tests/test_step3_model.py -v            # After Step 3
pytest tests/test_step4_training.py -v         # After Step 4 (TRAIN-1..3)
python -m steel_defect.train --epochs 2        # After Step 4 (TRAIN-4)
pytest tests/test_step5_inference.py -v        # After Step 5
streamlit run steel_defect/app.py              # After Step 6
```

Or run all tests at once:

```bash
pytest tests/ -v
```

## Key Files You Should Read First

Before writing any code, read these scaffold files to understand the project:

- **`utils.py`** — All constants your code will use: `IMAGE_SIZE`, `CLASS_NAMES`, `NUM_CLASSES`, `DATA_DIR`, `CHECKPOINT_PATH`, `DEVICE`
- **`gradcam.py`** — Provided Grad-CAM implementation (good learning material for PyTorch hooks)
- **`app.py`** — Skim the full Streamlit app to see how your functions get called

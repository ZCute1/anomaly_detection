# Step 1: Preprocessing

**File:** `steel_defect/preprocessing.py`
**Placeholders:** PREPROCESS-1, PREPROCESS-2

## Goal

Build two image transform pipelines using Albumentations — one for training
(with data augmentation) and one for validation/inference (deterministic only).

These pipelines convert raw images from disk into normalized PyTorch tensors
that the model can consume.

## What's Already Provided

The file gives you:

- **`IMAGENET_MEAN` and `IMAGENET_STD`** — Normalization constants. These are the
  channel-wise mean and standard deviation computed across the entire ImageNet dataset.
  Using them keeps pixel values in a well-conditioned range for optimization.
- **`overlay_gradcam()`** — A utility function for Grad-CAM visualization. You don't
  need to modify this.
- **`IMAGE_SIZE`** — Imported from `utils.py`, set to `(256, 256)`.

## PREPROCESS-1: Training Transforms

**Function:** `build_train_transforms()`

Build an `A.Compose` pipeline that applies these transforms in order:

1. **Resize** to `IMAGE_SIZE` (256 x 256)
2. **HorizontalFlip** with probability 0.5
3. **RandomBrightnessContrast** with probability 0.3
4. **Normalize** using `IMAGENET_MEAN` and `IMAGENET_STD`
5. **ToTensorV2** — converts the NumPy HWC uint8 array to a PyTorch CHW float32 tensor

### Why augmentation?

Training transforms include random augmentations (flips, brightness changes) so the
model sees slightly different versions of each image every epoch. This acts as
regularization and improves generalization. The augmentation happens *before*
normalization so it operates on natural pixel values.

### Key concepts

- `A.Compose([...])` chains transforms into a single callable pipeline
- `A.Resize(height=..., width=...)` — Albumentations uses height-first parameter order
- `A.Normalize(mean=..., std=...)` — Scales pixel values from [0, 255] to roughly [-2.5, 2.5]
- `ToTensorV2()` — Must be the last transform. Transposes HWC to CHW and converts dtype

### Common mistakes

- Forgetting `ToTensorV2()` at the end — your Dataset will return NumPy arrays instead of tensors
- Putting `Normalize` before augmentation transforms — augmentation should happen on natural pixel values
- Using `torchvision.transforms` instead of `albumentations` — this project uses Albumentations throughout

## PREPROCESS-2: Validation Transforms

**Function:** `build_val_transforms()`

Same as PREPROCESS-1 but **without any random augmentation**.
Only Resize, Normalize, and ToTensorV2.

### Why no augmentation?

Validation metrics must be deterministic — the same image should produce the same
prediction every time. Random augmentation would make metrics unstable across runs.

## Verification

After completing both placeholders, run:

```bash
pytest tests/test_step1_preprocessing.py -v
```

The tests check:

- Both functions return an `A.Compose` object
- Output tensors have shape `(3, 256, 256)` and dtype `float32`
- Values are in a normalized range (not 0–255)
- Validation transforms produce identical output when called twice on the same image

## Notebook Reference

See **Section 2: Preprocessing with Albumentations** in `mvtec_walkthrough.ipynb`
for a working example of both pipelines. Note the notebook uses 224x224 images —
your homework uses 256x256.

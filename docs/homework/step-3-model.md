# Step 3: Model Architecture

**File:** `steel_defect/model.py`
**Placeholders:** MODEL-1, MODEL-2

## Goal

Define a convolutional neural network (CNN) that takes a 3-channel 256x256 image
and outputs raw logit scores for each of the 5 defect classes.

## What's Already Provided

- **Class skeleton** — `SteelCNN(nn.Module)` with proper imports
- **`num_parameters` property** — Counts trainable parameters
- **`__repr__`** — String representation showing parameter count
- **`NUM_CLASSES`** — Imported from `utils.py` (value: 5)

## Architecture

The CNN has three sections: `features`, `pool`, and `classifier`.

```
Input: (batch, 3, 256, 256)
         │
    ┌────▼────────────────────────────┐
    │  features (nn.Sequential)       │
    │                                 │
    │  Block 1: Conv(3→32) + BN + ReLU + MaxPool  → (batch, 32, 128, 128)
    │  Block 2: Conv(32→64) + BN + ReLU + MaxPool → (batch, 64, 64, 64)
    │  Block 3: Conv(64→128) + BN + ReLU + MaxPool→ (batch, 128, 32, 32)
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │  pool: AdaptiveAvgPool2d(1)     │
    │  (batch, 128, 32, 32) → (batch, 128, 1, 1)
    └────┬────────────────────────────┘
         │
    ┌────▼────────────────────────────┐
    │  classifier (nn.Sequential)     │
    │                                 │
    │  Flatten → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→5)
    │  Output: (batch, 5) raw logits  │
    └─────────────────────────────────┘
```

## MODEL-1: Define Layers

**Method:** `SteelCNN.__init__(self, num_classes)`

### What to do

1. **Call `super().__init__()`** — This is mandatory for all `nn.Module` subclasses.
   Without it, PyTorch cannot track your layers.

2. **Define `self.features`** as an `nn.Sequential` with 3 convolutional blocks.
   Each block follows the pattern:
    - `nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)` — `padding=1` with `kernel_size=3` preserves spatial dimensions
    - `nn.BatchNorm2d(out_channels)` — Normalizes activations, stabilizes training
    - `nn.ReLU()` — Non-linear activation
    - `nn.MaxPool2d(2)` — Halves spatial dimensions (256→128→64→32)

   Channel progression: **3 → 32 → 64 → 128**

3. **Define `self.pool`** as `nn.AdaptiveAvgPool2d(1)` — Collapses any spatial
   size to 1x1 by averaging. This makes the model work with any input resolution.

4. **Define `self.classifier`** as an `nn.Sequential`:
    - `nn.Flatten()` — Reshapes (batch, 128, 1, 1) to (batch, 128)
    - `nn.Linear(128, 64)` — First fully connected layer
    - `nn.ReLU()`
    - `nn.Dropout(0.3)` — Randomly zeros 30% of neurons during training (regularization)
    - `nn.Linear(64, num_classes)` — Output layer, one score per class

### Key concepts

- **`nn.Sequential`** — Stacks layers into a single callable. When you call
  `self.features(x)`, it passes `x` through every layer in order.
- **`nn.Conv2d(in, out, kernel_size=3, padding=1)`** — "Same" convolution:
  output spatial size equals input spatial size. `padding=1` adds a 1-pixel
  border of zeros so the 3x3 kernel doesn't shrink the feature map.
- **`nn.BatchNorm2d`** — Normalizes each channel to zero mean and unit variance
  within a batch. Speeds up convergence and reduces sensitivity to initialization.
- **`nn.MaxPool2d(2)`** — Takes the maximum value in each 2x2 window, halving
  both height and width. This progressively reduces spatial resolution while
  the channel count increases.
- **`nn.AdaptiveAvgPool2d(1)`** — Unlike `MaxPool2d`, this pools to a fixed
  output size (1x1) regardless of input size. The `128` comes from the
  channel count of the last conv block.
- **`nn.Dropout(0.3)`** — Active during `model.train()`, disabled during
  `model.eval()`. Prevents the FC layers from memorizing the training set.

### Common mistakes

- Forgetting `super().__init__()` — PyTorch silently fails to register layers,
  and `model.parameters()` returns nothing
- Wrong `in_channels` on the first Linear layer — must match the last conv block's
  `out_channels` (128), not the spatial size
- Adding softmax to the classifier — `CrossEntropyLoss` applies it internally,
  so double-softmax degrades performance

## MODEL-2: Forward Pass

**Method:** `SteelCNN.forward(self, x)`

### What to do

Pass the input tensor through the three sections in order:

1. `x = self.features(x)` — Convolutional blocks
2. `x = self.pool(x)` — Adaptive average pooling
3. `x = self.classifier(x)` — Flatten + FC layers
4. `return x` — Raw logits (no softmax)

### Why no softmax?

`nn.CrossEntropyLoss` combines `log_softmax` and `NLLLoss` internally
for numerical stability. If you apply softmax in `forward()` and then
use `CrossEntropyLoss`, you effectively apply softmax twice, which
produces worse gradients.

Softmax is applied later, at inference time, when converting logits
to human-readable probabilities.

## Verification

```bash
pytest tests/test_step3_model.py -v
```

The tests check:

- The model has `features`, `pool`, and `classifier` attributes
- A forward pass with shape `(2, 3, 256, 256)` produces output shape `(2, 5)`
- Output dtype is `float32`
- Output is raw logits (not softmax probabilities)
- The model works with a custom `num_classes` value

You can also test manually:

```python
import torch
from steel_defect.model import SteelCNN

model = SteelCNN()
print(model)  # Shows parameter count
x = torch.randn(1, 3, 256, 256)
out = model(x)
print(out.shape)  # Should be (1, 5)
```

## Notebook Reference

See **Section 4: Model Architecture** in `mvtec_walkthrough.ipynb`.
The notebook uses a 2-block CNN (32→64) — your homework requires 3 blocks (32→64→128).
The pattern for each block is identical; you add one more.

# Step 5: Inference

**File:** `steel_defect/inference.py`
**Placeholders:** INFER-1, INFER-2

## Goal

Load a trained model from a checkpoint file and classify single images,
returning a result dictionary with the predicted label, confidence score,
per-class probabilities, and latency.

## What's Already Provided

- **`SteelPredictor.__init__`** — Sets up `checkpoint_path`, `device`,
  validation transforms, and an inference counter
- **`predict_from_file()`** — Convenience method that reads an image from
  disk and calls your `predict()` method
- Logging and timing scaffolding around both placeholders
- Imports: `torch`, `F` (functional), `cv2`, `SteelCNN`, `build_val_transforms`,
  `CLASS_NAMES`, `NUM_CLASSES`, `DEVICE`, `CHECKPOINT_PATH`

## INFER-1: Load Model

**Method:** `SteelPredictor.load_model()`

### What to do

1. **Check the checkpoint exists** — If `self.checkpoint_path` doesn't exist,
   raise `FileNotFoundError` with a message telling the user to train first
2. **Load the checkpoint** — `torch.load(self.checkpoint_path, map_location=self.device)`
3. **Create a model instance** — `SteelCNN(num_classes=...)`.
   Use `checkpoint.get("num_classes", NUM_CLASSES)` to read the class count
   from the checkpoint (falls back to the default if the key is missing)
4. **Load the weights** — `model.load_state_dict(checkpoint["model_state_dict"])`
5. **Move to device** — `model.to(self.device)`
6. **Set eval mode** — `model.eval()`
7. **Store it** — `self.model = model`

### Key concepts

- **`torch.load(path, map_location=device)`** — Deserializes the checkpoint.
  `map_location` remaps tensors to the specified device, so a checkpoint saved
  on GPU can be loaded on CPU (or vice versa).
- **`model.load_state_dict(state_dict)`** — Copies parameter values from the
  saved dictionary into the model's layers. The layer names must match exactly.
- **Symmetry with TRAIN-4** — `load_model` is the inverse of the checkpoint saving
  you wrote in TRAIN-4. The dict keys (`"model_state_dict"`, `"num_classes"`, etc.)
  are the same ones you saved.

### Important note

The scaffold code after the placeholder (`elapsed_ms = ...`, `logger.info(...)`)
will execute after your code. Make sure your code does **not** `return` early — just
let it flow into the scaffold logging.

### Common mistakes

- Forgetting `model.eval()` — Dropout stays active during inference, producing
  inconsistent predictions
- Forgetting `map_location` — If the checkpoint was saved on GPU and you load
  on CPU (or vice versa), you get a device error
- Not assigning to `self.model` — The `predict()` method checks `self.model`
  to decide whether to call `load_model()`

## INFER-2: Predict

**Method:** `SteelPredictor.predict(image)`

### What to do

The scaffold already handles the `self.model is None` check and the timing.
Your code goes between `start = time.perf_counter()` and the
`raise NotImplementedError` line. Replace the raise with:

1. **Apply validation transforms:**

    ```python
    result = self.transform(image=image)
    tensor = result["image"]
    ```

2. **Add batch dimension:**

    ```python
    tensor = tensor.unsqueeze(0)
    ```

    The model expects `(batch, 3, H, W)` but the transform produces `(3, H, W)`.
    `unsqueeze(0)` adds a dimension at position 0.

3. **Move to device:**

    ```python
    tensor = tensor.to(self.device)
    ```

4. **Forward pass with no gradients:**

    ```python
    with torch.no_grad():
        logits = self.model(tensor)
    ```

5. **Convert logits to probabilities:**

    ```python
    probs = F.softmax(logits, dim=1)
    ```

    `dim=1` applies softmax across the class dimension.

6. **Extract prediction:**

    ```python
    confidence, predicted = probs.max(dim=1)
    ```

    `probs.max(dim=1)` returns `(values, indices)` — the highest probability
    and which class it belongs to.

7. **Build the output variables** that the scaffold code after your placeholder expects:

    - `predicted_idx = predicted.item()` — Python `int`
    - `confidence_val = confidence.item()` — Python `float`
    - `label = CLASS_NAMES[predicted_idx]` — Class name string
    - `probs_np = probs.squeeze().cpu().numpy()` — NumPy array of all probabilities
    - `class_scores = {name: float(p) for name, p in zip(CLASS_NAMES, probs_np)}` — Dict

### Important: variable names matter

The scaffold code after your placeholder references `label`, `confidence_val`,
`class_scores`, and `predicted_idx` by name. Your code must define these exact
variable names before the scaffold code runs.

### Key concepts

- **`tensor.unsqueeze(0)`** — Models always work with batches.
  A single image is a batch of size 1.
- **`F.softmax(logits, dim=1)`** — Converts raw scores to probabilities
  that sum to 1.0 across classes. `dim=1` is the class axis.
- **`.item()`** — Extracts a Python scalar from a single-element tensor.
  Necessary because `probs.max()` returns tensors, not plain numbers.
- **`.squeeze().cpu().numpy()`** — Removes the batch dimension, moves
  from GPU to CPU, and converts to NumPy for the class_scores dict.
- **`torch.no_grad()`** — Same rationale as TRAIN-3: no gradients needed
  at inference time, saves memory and compute.

### Common mistakes

- Forgetting `unsqueeze(0)` — The model gets a 3D tensor and crashes
  or produces wrong output
- Using `dim=0` in softmax instead of `dim=1` — Softmax across the
  batch dimension instead of the class dimension
- Forgetting `.to(self.device)` — Device mismatch error

## Verification

After completing both placeholders (requires Steps 1 and 3 to be done):

```bash
pytest tests/test_step5_inference.py -v
```

Then launch the full app:

```bash
streamlit run steel_defect/app.py
```

Or test inference from the command line:

```python
from steel_defect.inference import SteelPredictor
predictor = SteelPredictor()
result = predictor.predict_from_file("data/steel_defect/defect_1/some_image.png")
print(result["label"], result["confidence"])
```

## Notebook Reference

- **Section 6: Evaluation** in `mvtec_walkthrough.ipynb` — Loading checkpoints
  and running predictions on a test set
- **Section 7: Single-Image Inference Pipeline** — The `predict_single()` function
  mirrors exactly what you need to write

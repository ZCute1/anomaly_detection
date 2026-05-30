# Step 4: Training

**File:** `steel_defect/train.py`
**Placeholders:** TRAIN-1, TRAIN-2, TRAIN-3, TRAIN-4

## Goal

Implement the PyTorch training loop: set up loss and optimizer,
write the train and validation epoch functions, and save the best model
checkpoint when validation accuracy improves.

## What's Already Provided

- **`train()`** — The main orchestration function that calls your implementations.
  It handles data loading, model creation, the epoch loop, and logging.
  Read through it to understand how your functions are called.
- **`main()`** — CLI entry point with argparse for epochs, batch size, and learning rate.
- All necessary imports from `torch.nn`, `torch.optim`, and project modules.

## TRAIN-1: Setup Training

**Function:** `setup_training(model, learning_rate)`

### What to do

Create and return two objects:

1. **`criterion`** — `nn.CrossEntropyLoss()`.
   This is the standard loss function for multi-class classification.
   It expects raw logits (not softmax) and integer class labels.

2. **`optimizer`** — `optim.Adam(model.parameters(), lr=learning_rate)`.
   Adam adapts the learning rate per-parameter using running averages
   of gradients and their squares.

Return them as a tuple: `(criterion, optimizer)`.

### Key concepts

- **`nn.CrossEntropyLoss`** — Combines `log_softmax` + `NLLLoss`.
  Input: logits of shape `(batch, num_classes)`.
  Target: integer labels of shape `(batch,)` — not one-hot encoded.
- **`model.parameters()`** — Returns an iterator over all trainable parameters.
  The optimizer needs this to know which tensors to update.
- **Why Adam?** — It's a good default. SGD with momentum can achieve better
  final accuracy with careful tuning, but Adam converges reliably with
  minimal hyperparameter tuning.

## TRAIN-2: Training Epoch

**Function:** `train_one_epoch(model, loader, criterion, optimizer, device)`

This is the core of PyTorch training. For each batch, execute the
five-step pattern: **zero → forward → loss → backward → step**.

### What to do

1. Set `model.train()` — Enables dropout and batch norm training behavior
2. Initialize accumulators: `running_loss = 0.0`, `correct = 0`, `total = 0`
3. Loop over the DataLoader:

    ```
    for images, labels in loader:
    ```

4. For each batch:
    - **Move data to device:** `images = images.to(device)`, `labels = labels.to(device)`
    - **Zero gradients:** `optimizer.zero_grad()` — PyTorch accumulates gradients by default;
      you must clear them before each batch
    - **Forward pass:** `outputs = model(images)` — Produces logits
    - **Compute loss:** `loss = criterion(outputs, labels)` — Scalar loss value
    - **Backward pass:** `loss.backward()` — Computes gradients for every parameter
    - **Update weights:** `optimizer.step()` — Applies the gradient update

5. Track metrics:
    - `running_loss += loss.item()` — `.item()` extracts a Python float from a 0-d tensor
    - `_, predicted = outputs.max(1)` — Index of highest logit per sample
    - `total += labels.size(0)` — Number of samples in this batch
    - `correct += predicted.eq(labels).sum().item()` — Count correct predictions

6. Return `(running_loss / len(loader), correct / total)` — Average loss and accuracy

### The five-step pattern in detail

```
optimizer.zero_grad()           # 1. Reset gradients to zero
outputs = model(images)         # 2. Forward: input → logits
loss = criterion(outputs, labels)  # 3. Compute scalar loss
loss.backward()                 # 4. Backward: compute ∂loss/∂param for every parameter
optimizer.step()                # 5. Update: param -= lr * gradient
```

This is the foundational loop of all PyTorch training. Every training script —
from simple CNNs to large language models — follows this exact pattern.

### Why `model.train()`?

Two layers behave differently in train vs eval mode:

- **Dropout** — Randomly zeros neurons during training; passes everything during eval
- **BatchNorm** — Uses batch statistics during training; uses running averages during eval

Calling `model.train()` ensures both layers are in the correct mode.

### Common mistakes

- Forgetting `optimizer.zero_grad()` — Gradients accumulate across batches,
  causing erratic weight updates
- Forgetting `.to(device)` — If data is on CPU and model is on GPU,
  you get a device mismatch error
- Using `loss` instead of `loss.item()` for accumulation — Keeps the entire
  computation graph in memory, eventually causing OOM

## TRAIN-3: Validation Epoch

**Function:** `validate(model, loader, criterion, device)`

Same structure as TRAIN-2 with three critical differences:

1. **`model.eval()`** instead of `model.train()`
2. **`with torch.no_grad():`** wraps the entire loop — disables gradient computation
3. **No optimizer calls** — no `zero_grad()`, no `backward()`, no `step()`

### What to do

1. Set `model.eval()`
2. Initialize the same accumulators
3. Inside `with torch.no_grad():`, loop over the loader:
    - Move data to device
    - Forward pass: `outputs = model(images)`
    - Compute loss (for monitoring, not for backprop)
    - Track metrics the same way as TRAIN-2
4. Return `(average_loss, accuracy)`

### Why `torch.no_grad()`?

During validation we only observe — we don't learn. `torch.no_grad()`
tells PyTorch not to build the computation graph, which:

- **Saves memory** — No intermediate tensors stored for backpropagation
- **Runs faster** — Less bookkeeping per operation

### Common mistakes

- Forgetting `model.eval()` — Dropout stays active, producing noisy validation metrics
- Calling `loss.backward()` or `optimizer.step()` — This would modify model weights
  during validation, contaminating your evaluation

## TRAIN-4: Save Checkpoint

**Location:** Inside the `train()` function's epoch loop (look for the TRAIN-4 comment block)

### What to do

After each epoch, check if validation accuracy improved:

1. Compare `val_acc > best_val_acc`
2. If improved:
    - Update `best_val_acc = val_acc`
    - Build a checkpoint dictionary containing:
        - `"model_state_dict"`: `model.state_dict()`
        - `"optimizer_state_dict"`: `optimizer.state_dict()`
        - `"epoch"`: current epoch number
        - `"best_val_acc"`: the new best accuracy
        - `"num_classes"`: `NUM_CLASSES`
    - Save with `torch.save(checkpoint, CHECKPOINT_PATH)`
    - Log the event with `logger.info(...)`

### Key concepts

- **`model.state_dict()`** — An `OrderedDict` mapping layer names to parameter tensors.
  This is the standard way to serialize a PyTorch model.
- **`optimizer.state_dict()`** — Saves optimizer state (momentum buffers, learning rate
  schedules). Needed to resume training from a checkpoint.
- **`torch.save(obj, path)`** — Serializes any Python object (dicts, tensors) to disk
  using pickle. The checkpoint dict is the conventional format.
- **Best-model strategy** — Only saving when accuracy improves keeps the best-performing
  weights. Without this, the final checkpoint might be from an overfitting epoch.

### Note on TRAIN-4

Unlike other placeholders, TRAIN-4 uses `pass` instead of `raise NotImplementedError`.
This means training can *run* without TRAIN-4, but it won't save any checkpoint.
Replace the `pass` with your checkpoint logic.

## Verification

After completing TRAIN-1, TRAIN-2, and TRAIN-3:

```bash
pytest tests/test_step4_training.py -v
```

After completing TRAIN-4 as well (plus all previous steps), run actual training:

```bash
python -m steel_defect.train --epochs 2
```

You should see:

- Epoch-by-epoch logging with train/val loss and accuracy
- A "Saved best model" log message when accuracy improves
- A checkpoint file at `models/steel_cnn_best.pt`

For a real training run:

```bash
python -m steel_defect.train --epochs 20
```

## Notebook Reference

- **Section 5: Training** in `mvtec_walkthrough.ipynb` — Shows the complete
  training loop with loss/optimizer setup, `train_one_epoch`, `validate`,
  and checkpoint saving
- Pay attention to the five-step pattern: `zero_grad → forward → loss → backward → step`

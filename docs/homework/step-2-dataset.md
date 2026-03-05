# Step 2: Dataset

**File:** `steel_defect/dataset.py`
**Placeholders:** DATA-1, DATA-2, DATA-3

## Goal

Load images from the directory-per-class dataset structure, implement
the PyTorch `Dataset.__getitem__` protocol, and create stratified train/val/test splits.

## What's Already Provided

- **`IMAGE_EXTENSIONS`** — Set of valid image file suffixes (`.png`, `.jpg`, etc.)
- **`SteelDataset.__init__`** and **`__len__`** — The Dataset class skeleton
- **`create_dataloaders()`** — Helper that wraps your Dataset into DataLoaders with batching
- Imports for `Path`, `cv2`, `CLASS_NAMES`, `DATA_DIR`

## DATA-1: Build File List

**Function:** `build_file_list(data_dir)`

Scan the dataset directory and return a list of `(image_path_str, label_index)` tuples.

### What to do

1. Iterate over subdirectories of `data_dir` using `Path.iterdir()`
2. For each subdirectory, check if its name is in `CLASS_NAMES`
3. If it matches, get the integer label with `CLASS_NAMES.index(folder.name)`
4. Collect all image files (check `IMAGE_EXTENSIONS` for valid suffixes)
5. Append `(str(image_path), label_index)` for each image
6. Return the list (sorted for reproducibility)

### Key concepts

- **`pathlib.Path`** — `.iterdir()` yields child paths, `.is_dir()` checks if directory,
  `.suffix` gives the file extension (with dot), `.name` gives the filename
- **Label mapping** — Folder names map to integers via their position in `CLASS_NAMES`.
  `"no_defect"` is at index 0, `"defect_1"` at index 1, etc.

### Common mistakes

- Forgetting to filter non-directory entries (files in the root)
- Forgetting `.lower()` on the suffix check — some images may be `.PNG` or `.JPG`
- Not converting `Path` to `str` in the tuple — the rest of the pipeline expects strings
- Including folders not in `CLASS_NAMES` (e.g., `.DS_Store` directories)

## DATA-2: Dataset `__getitem__`

**Method:** `SteelDataset.__getitem__(self, idx)`

Load a single image and return a `(tensor, label)` pair.

### What to do

1. Get the `(image_path, label)` tuple from `self.file_list[idx]`
2. Read the image with `cv2.imread(image_path)`
3. Check that the image was loaded (cv2 returns `None` for unreadable files)
4. Convert BGR to RGB with `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`
5. If `self.transform` is not `None`, apply it: `result = self.transform(image=image)` then `image = result["image"]`
6. Return `(image, label)`

### Key concepts

- **OpenCV BGR** — `cv2.imread` loads images in BGR channel order, not RGB.
  Neural networks and visualization tools expect RGB. Always convert.
- **Albumentations interface** — Transforms are called with `transform(image=img)`.
  This returns a dict; the transformed image is in `result["image"]`.
- **PyTorch Dataset protocol** — `__getitem__` receives an integer index and returns
  one sample. The DataLoader calls this method repeatedly to build batches.

### Common mistakes

- Forgetting BGR to RGB conversion — the model trains on wrong colors, reducing accuracy
- Not checking for `None` from `cv2.imread` — corrupted or missing files cause
  cryptic errors later in the DataLoader

## DATA-3: Create Splits

**Function:** `create_splits(file_list, train_ratio, val_ratio, seed)`

Split the file list into train, validation, and test sets using stratified sampling.

### What to do

1. Extract labels: `[label for _, label in file_list]`
2. Compute `test_ratio = 1.0 - train_ratio - val_ratio`
3. **First split** — separate the test set using `train_test_split`:
    - `train_test_split(file_list, test_size=test_ratio, stratify=labels, random_state=seed)`
    - This returns `(train_val_portion, test_list)`
4. **Second split** — split the remaining portion into train and val:
    - Compute `val_relative = val_ratio / (train_ratio + val_ratio)`
    - Extract labels again from the remaining portion
    - `train_test_split(train_val, test_size=val_relative, stratify=..., random_state=seed)`
5. Return `(train_list, val_list, test_list)`

### Why two splits?

`train_test_split` can only produce two groups at a time.
To get three groups (train/val/test), split twice:
first carve off the test set, then split what remains into train and val.

### Why stratification?

Setting `stratify=labels` ensures each split has the same class proportions
as the original dataset. Without it, small classes could end up entirely in
one split by random chance.

### The relative ratio math

After removing the test set, the remaining data represents `train_ratio + val_ratio`
of the total. To get the correct val portion from this subset, use:

```
val_relative = val_ratio / (train_ratio + val_ratio)
```

With defaults (0.70 / 0.15 / 0.15), `val_relative = 0.15 / 0.85 ≈ 0.176`.

### Common mistakes

- Using `val_ratio` directly in the second split instead of the relative ratio — this produces wrong split sizes
- Forgetting `stratify` — leads to unbalanced splits, especially with few samples
- Forgetting `random_state` — splits become non-reproducible

## Verification

After completing all three placeholders, run the dataset tests:

```bash
pytest tests/test_step2_dataset.py -v
```

You can also test data loading manually in a Python shell:

```python
from steel_defect.dataset import build_file_list, create_splits
file_list = build_file_list()
print(f"Total: {len(file_list)}")

train, val, test = create_splits(file_list)
print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
```

## Notebook Reference

- **Section 1** in `mvtec_walkthrough.ipynb` — Directory scanning with `pathlib`
- **Section 3** — `train_test_split` with stratification, `Dataset.__getitem__`

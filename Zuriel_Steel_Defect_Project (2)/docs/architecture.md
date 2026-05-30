# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (app.py)                │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Sidebar  │  │  Image View  │  │  Results + Heatmap     │ │
│  │ Controls  │  │  + Acquire   │  │  + History Dashboard   │ │
│  └──────────┘  └──────┬───────┘  └───────────▲────────────┘ │
└────────────────────────┼─────────────────────┼──────────────┘
                         │                     │
              ┌──────────▼──────────┐  ┌───────┴───────────┐
              │   acquisition.py    │  │   inference.py     │
              │   CameraSimulator   │  │   DefectDetector   │
              │   - acquire_frame() │  │   - predict()      │
              │   - frame buffer    │  │   - load_model()   │
              └──────────┬──────────┘  └───────┬───────────┘
                         │                     │
                    ┌────▼────┐        ┌───────▼───────────┐
                    │  data/  │        │ preprocessing.py   │
                    │  MVTec  │        │ - transforms       │
                    │  images │        │ - heatmap overlay  │
                    └─────────┘        └───────────────────┘
```

## Module Descriptions

### `anomaly_detection/app.py`

The Streamlit entry point. Orchestrates the UI, calls into acquisition and inference modules, and manages session state (inspection history, model caching). Has no business logic of its own.

### `anomaly_detection/acquisition.py`

Simulates industrial camera acquisition. The `CameraSimulator` class loads images from the MVTec dataset directory and serves them through an API that mirrors real industrial camera SDKs:

- `start_acquisition()` / `stop_acquisition()` — session lifecycle
- `acquire_frame()` — software trigger, returns frame + metadata
- Circular buffer (`collections.deque`) — prevents unbounded memory growth
- Frame metadata — tracks IDs, timestamps, source paths

In a real system, this module would be replaced by a Harvesters/GenTL driver or a vendor-specific SDK.

### `anomaly_detection/inference.py`

Wraps the PatchCore model from Anomalib. The `DefectDetector` class provides:

- `load_model()` — loads the trained checkpoint
- `predict(image)` — returns anomaly score, binary label, and heatmap
- Structured logging of every inference (label, score, latency)

### `anomaly_detection/preprocessing.py`

Image transformation pipeline: resizing, ImageNet normalization, tensor conversion. Also provides `overlay_heatmap()` for blending anomaly maps onto original images for visualization.

### `anomaly_detection/train.py`

One-time script that runs PatchCore feature extraction. Builds a coreset memory bank from the "good" training images. No gradient descent — PatchCore uses a frozen pretrained backbone (WideResNet50) and stores extracted feature patches for nearest-neighbor comparison at inference time.

### `anomaly_detection/utils.py`

Shared configuration: project paths, image settings, device selection, and logging setup. All modules import their configuration from here — no hardcoded paths anywhere else.

## Data Flow

1. **Acquisition**: `CameraSimulator.acquire_frame()` loads an image from disk, converts BGR→RGB, stores in buffer
2. **Preprocessing**: `preprocess_frame()` resizes to 256×256, normalizes with ImageNet stats, converts to PyTorch tensor
3. **Inference**: `DefectDetector.predict()` runs the tensor through PatchCore, extracts anomaly score and heatmap
4. **Display**: Streamlit shows the original frame, heatmap overlay, and classification result
5. **Logging**: Every step logs timestamps, parameters, and results to `logs/app.log`

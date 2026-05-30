"""
Defect Detector — Streamlit Application.

Interactive frontend for the MVTec Metal Nut anomaly detection pipeline.
Provides image acquisition simulation, model inference with heatmap
visualization, and inspection history tracking.

Features:
    - Continuous auto-inspection mode (simulates production line)
    - Manual single-shot inspection
    - Adjustable inspection speed, threshold, and heatmap opacity
    - Live inspection statistics

Run with:
    streamlit run anomaly_detection/app.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anomaly_detection.acquisition import CameraSimulator, get_default_simulator
from anomaly_detection.inference import DefectDetector
from anomaly_detection.preprocessing import overlay_heatmap
from anomaly_detection.utils import (
    setup_logging,
    MVTEC_TEST_DIR,
    MVTEC_TRAIN_GOOD,
    IMAGE_SIZE,
)

logger = setup_logging("anomaly_detection.app")

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)

# ── Session State Initialization ──────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False


# ── Helper Functions ──────────────────────────────────────────
@st.cache_resource
def load_detector() -> DefectDetector:
    """Load model once and cache across reruns."""
    logger.info("Loading DefectDetector (cached)")
    detector = DefectDetector()
    detector.load_model()
    return detector


def get_available_categories() -> list[str]:
    """List MVTec test categories that exist on disk."""
    categories = []
    if MVTEC_TRAIN_GOOD.exists():
        categories.append("good")
    if MVTEC_TEST_DIR.exists():
        for d in sorted(MVTEC_TEST_DIR.iterdir()):
            if d.is_dir():
                categories.append(d.name)
    return categories


def run_single_inspection(
    simulator: CameraSimulator,
    detector: DefectDetector,
) -> dict | None:
    """Acquire one frame and run inference. Returns result dict or None."""
    frame, metadata = simulator.acquire_frame()
    if frame is None:
        return None

    result = detector.predict(frame)
    result["frame_id"] = metadata.frame_id
    result["source"] = Path(metadata.source_path).name
    result["frame"] = frame
    return result


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Anomaly Detection")
    st.caption("MVTec Metal Nut — PatchCore Anomaly Detection")
    st.divider()

    # Model controls
    st.subheader("Model Settings")
    threshold = st.slider(
        "Anomaly threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Scores above this threshold are classified as defective.",
    )

    heatmap_alpha = st.slider(
        "Heatmap opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
    )

    st.divider()

    # Data source selection
    st.subheader("Image Source")
    categories = get_available_categories()

    if not categories:
        st.error(
            "⚠️ No MVTec data found.\n\n"
            "Download the Metal Nut dataset and extract to:\n"
            "`data/metal_nut/`"
        )
        st.stop()

    selected_category = st.selectbox(
        "Test category",
        categories,
        help="Select which set of images to inspect.",
        disabled=st.session_state.running,
    )

    st.divider()

    # Inspection speed (for auto mode)
    st.subheader("Auto-Inspect Speed")
    inspect_interval = st.slider(
        "Interval between inspections (sec)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.5,
        help="Lower = faster inspection cycle.",
    )

    st.divider()

    # History
    st.subheader("Inspection History")
    if st.session_state.history:
        good_count = sum(1 for h in st.session_state.history if h["label"] == "Good")
        defect_count = len(st.session_state.history) - good_count
        st.metric("Inspected", len(st.session_state.history))
        col1, col2 = st.columns(2)
        col1.metric("✅ Good", good_count)
        col2.metric("❌ Defective", defect_count)

        if not st.session_state.running:
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
    else:
        st.caption("No inspections yet.")


# ── Main Content ──────────────────────────────────────────────
st.header("Metal Nut Inspection")

# Initialize simulator for selected category
simulator = get_default_simulator(selected_category)
simulator.start_acquisition()

# Load model
try:
    detector = load_detector()
    detector.threshold = threshold
except FileNotFoundError:
    st.error(
        "⚠️ No trained model found.\n\n"
        "Run training first:\n"
        "```\npython -m anomaly_detection.train\n```"
    )
    st.stop()

# ── Control Buttons ───────────────────────────────────────────
col_auto, col_manual, col_spacer = st.columns([1, 1, 2])

with col_auto:
    if st.session_state.running:
        if st.button("⏹  Stop Inspection", type="primary", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    else:
        if st.button("▶  Run Inspection", type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()

with col_manual:
    manual_btn = st.button(
        "🔬 Single Inspect",
        use_container_width=True,
        disabled=st.session_state.running,
    )

# ── Display placeholders (updated in-place during auto mode) ──
st.divider()
status_placeholder = st.empty()
col_img, col_heat, col_stats = st.columns([1, 1, 0.8])

with col_img:
    img_header = st.empty()
    img_display = st.empty()
    img_caption = st.empty()

with col_heat:
    heat_header = st.empty()
    heat_display = st.empty()
    heat_caption = st.empty()

with col_stats:
    stats_header = st.empty()
    stats_label = st.empty()
    stats_score = st.empty()
    stats_latency = st.empty()
    stats_frame = st.empty()
    stats_buffer = st.empty()

history_divider = st.empty()
history_header = st.empty()
history_cols_placeholder = st.empty()


def display_result(result: dict, simulator: CameraSimulator, heatmap_alpha: float):
    """Render a single inspection result into the placeholders."""
    frame = result["frame"]

    img_header.subheader("Acquired Frame")
    img_display.image(frame, use_container_width=True)
    img_caption.caption(f"Source: {result['source']}")

    heat_header.subheader("Anomaly Heatmap")
    frame_resized = cv2.resize(frame, IMAGE_SIZE)
    blended = overlay_heatmap(frame_resized, result["heatmap"], alpha=heatmap_alpha)
    heat_display.image(blended, use_container_width=True)
    heat_caption.caption("Red = high anomaly")

    stats_header.subheader("Inference Result")
    if result["label"] == "Defective":
        stats_label.error(f"❌ **{result['label']}**")
    else:
        stats_label.success(f"✅ **{result['label']}**")

    stats_score.metric("Anomaly Score", f"{result['score']:.4f}")
    stats_latency.metric("Latency", f"{result['latency_ms']:.0f} ms")
    stats_frame.metric("Frame ID", result["frame_id"])

    current, capacity = simulator.get_buffer_level()
    stats_buffer.progress(current / capacity, text=f"Buffer: {current}/{capacity}")


def display_history():
    """Render recent inspection history."""
    if not st.session_state.history:
        return

    history_divider.divider()
    history_header.subheader("Recent Inspections")

    recent = list(reversed(st.session_state.history[-10:]))
    with history_cols_placeholder.container():
        cols = st.columns(min(len(recent), 5))
        for i, item in enumerate(recent[:5]):
            with cols[i]:
                icon = "✅" if item["label"] == "Good" else "❌"
                st.markdown(f"**{icon} {item['label']}**")
                st.caption(f"Score: {item['score']:.3f}")
                st.caption(f"Source: {item.get('source', 'N/A')}")


# ── Manual single-shot mode ──────────────────────────────────
if manual_btn and not st.session_state.running:
    result = run_single_inspection(simulator, detector)
    if result is not None:
        st.session_state.history.append(result)
        display_result(result, simulator, heatmap_alpha)
        display_history()

# ── Auto-inspection loop ─────────────────────────────────────
if st.session_state.running:
    total = len(simulator.image_paths)
    status_placeholder.info(
        f"🔄 **Auto-inspecting** — {selected_category} "
        f"({total} images) — every {inspect_interval:.1f}s  "
        f"*Press Stop to end.*"
    )

    inspected = 0
    while st.session_state.running:
        result = run_single_inspection(simulator, detector)
        if result is None:
            status_placeholder.warning("No more frames available.")
            break

        st.session_state.history.append(result)
        inspected += 1

        display_result(result, simulator, heatmap_alpha)
        display_history()

        # Pause between inspections — this also lets Streamlit
        # detect the Stop button press on the next rerun
        time.sleep(inspect_interval)

        # After cycling through all images, loop restarts from
        # the simulator's circular buffer automatically.
        # Stop after a full pass if desired:
        if inspected >= total:
            # Start over from the beginning
            simulator = get_default_simulator(selected_category)
            simulator.start_acquisition()
            inspected = 0

# Show history if we have any but aren't in a mode
if not st.session_state.running and not manual_btn:
    display_history()

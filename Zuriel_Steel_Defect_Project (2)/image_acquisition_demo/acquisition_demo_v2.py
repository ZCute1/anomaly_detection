"""
Image Acquisition Dynamics Demo
===============================
A standalone Tkinter application demonstrating industrial camera
acquisition concepts using a webcam and YOLOv8 object detection.

Concepts demonstrated:
- Producer-consumer pattern with a shared frame buffer
- Four buffer management policies:
    1. Circular Buffer — drop oldest on overflow, consumer processes FIFO
    2. Time-Based Expiry — consumer skips frames older than max_age_ms
    3. Newest Only — consumer discards backlog, always processes latest frame
    4. Backpressure — producer blocks when buffer is full (throttles camera)
- Frame rate decoupling (capture FPS vs processing FPS)
- Frame drops / skips when consumer can't keep up
- Adjustable processing delay to simulate heavy inference

Requirements:
    pip install opencv-python ultralytics Pillow numpy
    pip install cv2-enumerate-cameras   # optional, for camera discovery

Usage:
    python acquisition_demo.py
"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Optional imports ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Using dummy detector.")
    print("Install with: pip install ultralytics")

try:
    from cv2_enumerate_cameras import enumerate_cameras
    ENUM_AVAILABLE = True
except ImportError:
    ENUM_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

# Buffer policy names (displayed in dropdown)
POLICY_CIRCULAR = "Circular Buffer"
POLICY_TIME_EXPIRY = "Time-Based Expiry"
POLICY_NEWEST_ONLY = "Newest Only"
POLICY_BACKPRESSURE = "Backpressure"

ALL_POLICIES = [POLICY_CIRCULAR, POLICY_TIME_EXPIRY, POLICY_NEWEST_ONLY, POLICY_BACKPRESSURE]

POLICY_DESCRIPTIONS = {
    POLICY_CIRCULAR:     "Producer drops oldest frame when buffer is full. Consumer processes in FIFO order.",
    POLICY_TIME_EXPIRY:  "Consumer skips any frame older than max age. Ensures decisions use fresh data.",
    POLICY_NEWEST_ONLY:  "Consumer discards entire backlog, always processes the most recent frame.",
    POLICY_BACKPRESSURE: "Producer blocks when buffer is full — throttles camera to match consumer speed.",
}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def discover_cameras() -> list[tuple[int, str]]:
    """
    Discover available cameras on the system.
    Returns list of (index, name) tuples.
    """
    if ENUM_AVAILABLE:
        cams = enumerate_cameras()
        seen = {}
        for cam in cams:
            if cam.name not in seen:
                seen[cam.name] = (cam.index, cam.name)
        return list(seen.values())
    else:
        found = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append((i, f"Camera {i}"))
                cap.release()
        return found if found else [(0, "Default Camera")]


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════

@dataclass
class FramePacket:
    """Represents a single acquired frame with metadata."""
    frame_id: int
    image: np.ndarray
    timestamp: float
    acquisition_time_ms: float


@dataclass
class DetectionResult:
    """Result from the object detection model."""
    frame_id: int
    labels: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    boxes: list = field(default_factory=list)
    inference_time_ms: float = 0.0
    annotated_image: Optional[np.ndarray] = None


# ══════════════════════════════════════════════════════════════
# PRODUCER — Camera Acquisition Thread
# ══════════════════════════════════════════════════════════════

class CameraProducer(threading.Thread):
    """
    Captures frames from webcam and pushes to a shared buffer.
    Behavior changes based on the selected buffer policy.
    """

    def __init__(
        self,
        buffer: queue.Queue,
        buffer_size: int = 30,
        camera_index: int = 0,
        policy: str = POLICY_CIRCULAR,
    ):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.camera_index = camera_index
        self.policy = policy

        self.running = False
        self.frame_count = 0
        self.dropped_frames = 0
        self.blocked_time_ms = 0.0      # Total time spent blocked (backpressure)
        self.capture_fps = 0.0
        self._fps_history: deque = deque(maxlen=30)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        print(f"[PRODUCER] Started | camera={self.camera_index} | "
              f"buffer_size={self.buffer_size} | policy={self.policy}")

        while self.running:
            start = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                continue

            acq_time = (time.perf_counter() - start) * 1000
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            packet = FramePacket(
                frame_id=self.frame_count,
                image=frame_rgb,
                timestamp=time.time(),
                acquisition_time_ms=acq_time,
            )

            # ── Policy-specific put behavior ──
            if self.policy == POLICY_BACKPRESSURE:
                # Block until space is available (throttles camera)
                block_start = time.perf_counter()
                while self.running:
                    try:
                        self.buffer.put(packet, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                self.blocked_time_ms += (time.perf_counter() - block_start) * 1000

            else:
                # Circular / Time-Based / Newest-Only:
                # Drop oldest when full, always accept new frame
                if self.buffer.full():
                    try:
                        self.buffer.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                self.buffer.put(packet)

            self.frame_count += 1

            # Track FPS
            elapsed = time.perf_counter() - start
            self._fps_history.append(elapsed)
            if len(self._fps_history) > 1:
                avg = sum(self._fps_history) / len(self._fps_history)
                self.capture_fps = 1.0 / avg if avg > 0 else 0.0

        cap.release()
        print("[PRODUCER] Stopped")

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════
# CONSUMER — Object Detection Thread
# ══════════════════════════════════════════════════════════════

class DetectionConsumer(threading.Thread):
    """
    Pulls frames from buffer and runs object detection.
    Frame selection strategy depends on the buffer policy.
    """

    def __init__(
        self,
        buffer: queue.Queue,
        result_queue: queue.Queue,
        policy: str = POLICY_CIRCULAR,
        max_age_ms: float = 500.0,
    ):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.result_queue = result_queue
        self.policy = policy
        self.max_age_ms = max_age_ms

        self.running = False
        self.extra_delay: float = 0.0
        self.processed_count = 0
        self.skipped_frames = 0          # Frames skipped by policy
        self.processing_fps = 0.0
        self._fps_history: deque = deque(maxlen=30)

        # Load model
        if YOLO_AVAILABLE:
            print("[CONSUMER] Loading YOLOv8n model...")
            self.model = YOLO("yolov8n.pt")
            print("[CONSUMER] Model loaded")
        else:
            self.model = None

    def run(self):
        self.running = True
        print(f"[CONSUMER] Started | policy={self.policy} | max_age_ms={self.max_age_ms}")

        while self.running:
            packet = self._fetch_frame()
            if packet is None:
                continue

            start = time.perf_counter()

            result = self._detect(packet)

            if self.extra_delay > 0:
                time.sleep(self.extra_delay)

            elapsed = time.perf_counter() - start

            self._fps_history.append(elapsed)
            if len(self._fps_history) > 1:
                avg = sum(self._fps_history) / len(self._fps_history)
                self.processing_fps = 1.0 / avg if avg > 0 else 0.0

            self.processed_count += 1

            # Push result to UI (keep only latest)
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(result)

        print("[CONSUMER] Stopped")

    def _fetch_frame(self) -> Optional[FramePacket]:
        """
        Get the next frame to process based on buffer policy.

        - Circular / Backpressure: simple FIFO get
        - Time-Based Expiry: skip frames older than max_age_ms
        - Newest Only: drain buffer, return only the latest frame
        """
        if self.policy == POLICY_NEWEST_ONLY:
            # Drain everything, keep only the last frame
            latest = None
            drained = 0
            while True:
                try:
                    pkt = self.buffer.get_nowait()
                    if latest is not None:
                        drained += 1
                    latest = pkt
                except queue.Empty:
                    break

            self.skipped_frames += drained

            if latest is None:
                # Buffer was empty, wait briefly
                try:
                    return self.buffer.get(timeout=0.1)
                except queue.Empty:
                    return None
            return latest

        elif self.policy == POLICY_TIME_EXPIRY:
            # Keep pulling until we find a fresh frame or buffer is empty
            while True:
                try:
                    pkt = self.buffer.get(timeout=0.1)
                except queue.Empty:
                    return None

                age_ms = (time.time() - pkt.timestamp) * 1000
                if age_ms <= self.max_age_ms:
                    return pkt
                else:
                    self.skipped_frames += 1
                    # Keep pulling — there might be a fresh one behind it

        else:
            # Circular / Backpressure: standard FIFO
            try:
                return self.buffer.get(timeout=0.1)
            except queue.Empty:
                return None

    def _detect(self, packet: FramePacket) -> DetectionResult:
        """Run object detection on a frame."""
        start = time.perf_counter()

        if self.model is not None:
            results = self.model(packet.image, verbose=False)
            r = results[0]

            labels = []
            confidences = []
            boxes = []

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]
                labels.append(label)
                confidences.append(conf)
                boxes.append(box.xyxy[0].cpu().numpy())

            annotated = r.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        else:
            time.sleep(0.05)
            labels = ["(no model)"]
            confidences = [0.0]
            boxes = []
            annotated = packet.image.copy()

        inference_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            frame_id=packet.frame_id,
            labels=labels,
            confidences=confidences,
            boxes=boxes,
            inference_time_ms=inference_ms,
            annotated_image=annotated,
        )

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════
# TKINTER UI
# ══════════════════════════════════════════════════════════════

class AcquisitionDemoApp:
    """
    Main application window showing acquisition dynamics.

    Layout:
    ┌─────────────────────────┬────────────────────┐
    │                         │  Pipeline Status    │
    │    Live Video Feed      │  Buffer Gauge       │
    │    (with detections)    │  FPS Counters       │
    │                         │  Detection Results  │
    ├─────────────────────────┴────────────────────┤
    │  Controls: Policy, Camera, Start/Stop, Delay  │
    └───────────────────────────────────────────────┘
    """

    # Colors
    BG_DARK = "#0F1B2D"
    BG_PANEL = "#162640"
    BG_CARD = "#1E3454"
    ACCENT = "#E8792B"
    TEXT_LIGHT = "#E8ECF1"
    TEXT_MUTED = "#6B7B8D"
    GREEN = "#2D9C5A"
    RED = "#DC3545"
    BLUE = "#2E5090"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Acquisition Dynamics Demo")
        self.root.configure(bg=self.BG_DARK)
        self.root.geometry("1100x780")
        self.root.minsize(900, 650)

        # Shared state
        self.buffer_size = 30
        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
        self.result_queue = queue.Queue(maxsize=2)

        self.producer: Optional[CameraProducer] = None
        self.consumer: Optional[DetectionConsumer] = None
        self.is_running = False

        self._build_ui()
        self._update_ui()

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=self.ACCENT, height=4)
        header.pack(fill="x")

        title_frame = tk.Frame(self.root, bg=self.BG_DARK, pady=10)
        title_frame.pack(fill="x")
        tk.Label(
            title_frame,
            text="⚙  Image Acquisition Dynamics Demo",
            font=("Trebuchet MS", 18, "bold"),
            fg=self.TEXT_LIGHT, bg=self.BG_DARK,
        ).pack(side="left", padx=15)

        # ── Main content area ──
        main = tk.Frame(self.root, bg=self.BG_DARK)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # Left: video feed
        self.video_frame = tk.Frame(main, bg=self.BG_PANEL, relief="flat", bd=0)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.video_label = tk.Label(
            self.video_frame, bg=self.BG_PANEL,
            text="Camera feed will appear here\nClick 'Start' to begin",
            font=("Calibri", 14), fg=self.TEXT_MUTED,
        )
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Right: stats panel
        self.stats_frame = tk.Frame(main, bg=self.BG_DARK, width=320)
        self.stats_frame.pack(side="right", fill="y", padx=(5, 0))
        self.stats_frame.pack_propagate(False)

        self._build_stats_panel()

        # ── Controls ──
        self._build_controls()

    def _build_stats_panel(self):
        sf = self.stats_frame

        # ── Pipeline Stage Indicators ──
        self._make_section_label(sf, "PIPELINE STATUS")

        self.pipeline_frame = tk.Frame(sf, bg=self.BG_PANEL, pady=8, padx=8)
        self.pipeline_frame.pack(fill="x", pady=(0, 8))

        stages = ["Camera", "→", "Buffer", "→", "Model", "→", "Output"]
        self.pipeline_labels = []
        pf = tk.Frame(self.pipeline_frame, bg=self.BG_PANEL)
        pf.pack()
        for s in stages:
            if s == "→":
                lbl = tk.Label(pf, text="→", font=("Calibri", 12), fg=self.TEXT_MUTED, bg=self.BG_PANEL)
            else:
                lbl = tk.Label(
                    pf, text=f" {s} ", font=("Calibri", 10, "bold"),
                    fg=self.TEXT_LIGHT, bg=self.BG_CARD, padx=6, pady=2,
                )
                self.pipeline_labels.append(lbl)
            lbl.pack(side="left", padx=2)

        # ── Buffer Gauge ──
        self._make_section_label(sf, "FRAME BUFFER")

        buf_card = tk.Frame(sf, bg=self.BG_PANEL, padx=10, pady=10)
        buf_card.pack(fill="x", pady=(0, 8))

        self.buffer_text = tk.Label(
            buf_card, text="0 / 30",
            font=("Consolas", 16, "bold"), fg=self.ACCENT, bg=self.BG_PANEL,
        )
        self.buffer_text.pack()

        self.buffer_bar = ttk.Progressbar(
            buf_card, length=280, mode="determinate", maximum=self.buffer_size,
        )
        self.buffer_bar.pack(pady=(5, 5))

        # Frame accounting row
        accounting = tk.Frame(buf_card, bg=self.BG_PANEL)
        accounting.pack(fill="x")

        self.dropped_text = tk.Label(
            accounting, text="Dropped: 0",
            font=("Calibri", 11), fg=self.RED, bg=self.BG_PANEL,
        )
        self.dropped_text.pack(side="left")

        self.skipped_text = tk.Label(
            accounting, text="Skipped: 0",
            font=("Calibri", 11), fg=self.ACCENT, bg=self.BG_PANEL,
        )
        self.skipped_text.pack(side="right")

        # Backpressure blocked time
        self.blocked_text = tk.Label(
            buf_card, text="",
            font=("Calibri", 10), fg=self.TEXT_MUTED, bg=self.BG_PANEL,
        )
        self.blocked_text.pack()

        # ── FPS Counters ──
        self._make_section_label(sf, "THROUGHPUT")

        fps_card = tk.Frame(sf, bg=self.BG_PANEL, padx=10, pady=10)
        fps_card.pack(fill="x", pady=(0, 8))

        fps_grid = tk.Frame(fps_card, bg=self.BG_PANEL)
        fps_grid.pack()

        tk.Label(fps_grid, text="Capture FPS", font=("Calibri", 10), fg=self.TEXT_MUTED, bg=self.BG_PANEL).grid(row=0, column=0, padx=(0, 20))
        tk.Label(fps_grid, text="Process FPS", font=("Calibri", 10), fg=self.TEXT_MUTED, bg=self.BG_PANEL).grid(row=0, column=1)

        self.capture_fps_label = tk.Label(
            fps_grid, text="0.0", font=("Consolas", 20, "bold"),
            fg=self.GREEN, bg=self.BG_PANEL,
        )
        self.capture_fps_label.grid(row=1, column=0, padx=(0, 20))

        self.process_fps_label = tk.Label(
            fps_grid, text="0.0", font=("Consolas", 20, "bold"),
            fg=self.BLUE, bg=self.BG_PANEL,
        )
        self.process_fps_label.grid(row=1, column=1)

        # ── Detection Results ──
        self._make_section_label(sf, "DETECTIONS")

        det_card = tk.Frame(sf, bg=self.BG_PANEL, padx=10, pady=10)
        det_card.pack(fill="x", pady=(0, 8))

        self.det_text = tk.Text(
            det_card, height=6, width=35,
            font=("Consolas", 10), fg=self.TEXT_LIGHT, bg=self.BG_CARD,
            relief="flat", state="disabled", wrap="word",
        )
        self.det_text.pack()

        # ── Frame counter ──
        self.frame_counter_label = tk.Label(
            sf, text="Frames: captured=0  processed=0",
            font=("Calibri", 10), fg=self.TEXT_MUTED, bg=self.BG_DARK,
        )
        self.frame_counter_label.pack(pady=(5, 0))

    def _build_controls(self):
        # ── Top row: Policy + description ──
        policy_row = tk.Frame(self.root, bg=self.BG_DARK, padx=15)
        policy_row.pack(fill="x", padx=10, pady=(5, 0))

        tk.Label(
            policy_row, text="Buffer Policy:",
            font=("Trebuchet MS", 11, "bold"), fg=self.ACCENT, bg=self.BG_DARK,
        ).pack(side="left", padx=(0, 8))

        self.policy_var = tk.StringVar(value=POLICY_CIRCULAR)
        self.policy_dropdown = ttk.Combobox(
            policy_row, textvariable=self.policy_var,
            values=ALL_POLICIES, width=22, state="readonly",
            font=("Calibri", 10),
        )
        self.policy_dropdown.pack(side="left", padx=(0, 15))
        self.policy_dropdown.bind("<<ComboboxSelected>>", self._on_policy_change)

        # Max age control (only visible for Time-Based Expiry)
        self.maxage_frame = tk.Frame(policy_row, bg=self.BG_DARK)
        self.maxage_frame.pack(side="left", padx=(0, 15))

        tk.Label(
            self.maxage_frame, text="Max Age (ms):",
            font=("Calibri", 11), fg=self.TEXT_LIGHT, bg=self.BG_DARK,
        ).pack(side="left", padx=(0, 5))

        self.maxage_var = tk.IntVar(value=500)
        self.maxage_spin = tk.Spinbox(
            self.maxage_frame, from_=50, to=5000, increment=50,
            textvariable=self.maxage_var, width=6,
            font=("Consolas", 11), state="readonly",
        )
        self.maxage_spin.pack(side="left")
        self.maxage_frame.pack_forget()  # Hidden by default

        # Policy description
        self.policy_desc_label = tk.Label(
            policy_row, text=POLICY_DESCRIPTIONS[POLICY_CIRCULAR],
            font=("Calibri", 10, "italic"), fg=self.TEXT_MUTED, bg=self.BG_DARK,
            anchor="w",
        )
        self.policy_desc_label.pack(side="left", padx=(5, 0))

        # ── Bottom row: Camera, Start, Delay, Buffer Size ──
        ctrl = tk.Frame(self.root, bg=self.BG_PANEL, pady=10, padx=15)
        ctrl.pack(fill="x", padx=10, pady=(5, 10))

        # Camera selector
        tk.Label(
            ctrl, text="Camera:",
            font=("Calibri", 11), fg=self.TEXT_LIGHT, bg=self.BG_PANEL,
        ).pack(side="left", padx=(0, 5))

        self.cameras = discover_cameras()
        self.camera_names = [f"{name} (idx {idx})" for idx, name in self.cameras]
        self.camera_var = tk.StringVar(value=self.camera_names[0] if self.camera_names else "No camera")
        self.camera_dropdown = ttk.Combobox(
            ctrl, textvariable=self.camera_var,
            values=self.camera_names, width=25, state="readonly",
            font=("Calibri", 10),
        )
        self.camera_dropdown.pack(side="left", padx=(0, 15))

        # Start/Stop button
        self.toggle_btn = tk.Button(
            ctrl, text="▶  Start", font=("Trebuchet MS", 12, "bold"),
            fg="white", bg=self.GREEN, activebackground="#248C4E",
            relief="flat", padx=20, pady=5,
            command=self._toggle_acquisition,
        )
        self.toggle_btn.pack(side="left")

        # Processing delay slider
        tk.Label(
            ctrl, text="Processing Delay (ms):",
            font=("Calibri", 11), fg=self.TEXT_LIGHT, bg=self.BG_PANEL,
        ).pack(side="left", padx=(30, 5))

        self.delay_var = tk.IntVar(value=0)
        self.delay_slider = tk.Scale(
            ctrl, from_=0, to=500, orient="horizontal",
            variable=self.delay_var, length=200,
            font=("Calibri", 9), fg=self.TEXT_LIGHT, bg=self.BG_PANEL,
            troughcolor=self.BG_CARD, highlightthickness=0,
            command=self._on_delay_change,
        )
        self.delay_slider.pack(side="left", padx=5)

        # Buffer size control
        tk.Label(
            ctrl, text="Buffer Size:",
            font=("Calibri", 11), fg=self.TEXT_LIGHT, bg=self.BG_PANEL,
        ).pack(side="left", padx=(30, 5))

        self.bufsize_var = tk.IntVar(value=30)
        self.bufsize_spin = tk.Spinbox(
            ctrl, from_=5, to=100, increment=5,
            textvariable=self.bufsize_var, width=5,
            font=("Consolas", 11), state="readonly",
            command=self._on_bufsize_change,
        )
        self.bufsize_spin.pack(side="left")

    def _make_section_label(self, parent, text):
        tk.Label(
            parent, text=text, font=("Trebuchet MS", 9, "bold"),
            fg=self.ACCENT, bg=self.BG_DARK, anchor="w",
        ).pack(fill="x", pady=(8, 2))

    # ── Control Handlers ──

    def _on_policy_change(self, _=None):
        """Show/hide max age control based on selected policy."""
        policy = self.policy_var.get()
        self.policy_desc_label.configure(text=POLICY_DESCRIPTIONS.get(policy, ""))

        if policy == POLICY_TIME_EXPIRY:
            self.maxage_frame.pack(side="left", padx=(0, 15))
        else:
            self.maxage_frame.pack_forget()

    def _toggle_acquisition(self):
        if self.is_running:
            self._stop()
        else:
            self._start()

    def _start(self):
        # Get selected camera
        sel_idx = self.camera_dropdown.current()
        if sel_idx < 0 or sel_idx >= len(self.cameras):
            sel_idx = 0
        camera_index = self.cameras[sel_idx][0]

        # Get selected policy
        policy = self.policy_var.get()

        # Reset buffer
        self.buffer_size = self.bufsize_var.get()
        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
        self.buffer_bar.configure(maximum=self.buffer_size)

        # Start threads with policy
        self.producer = CameraProducer(
            self.frame_buffer,
            buffer_size=self.buffer_size,
            camera_index=camera_index,
            policy=policy,
        )
        self.consumer = DetectionConsumer(
            self.frame_buffer,
            self.result_queue,
            policy=policy,
            max_age_ms=float(self.maxage_var.get()),
        )
        self.consumer.extra_delay = self.delay_var.get() / 1000.0

        self.producer.start()
        self.consumer.start()
        self.is_running = True

        # Lock config controls
        self.toggle_btn.configure(text="■  Stop", bg=self.RED, activebackground="#C82333")
        self.bufsize_spin.configure(state="disabled")
        self.camera_dropdown.configure(state="disabled")
        self.policy_dropdown.configure(state="disabled")
        self.maxage_spin.configure(state="disabled")

    def _stop(self):
        if self.producer:
            self.producer.stop()
        if self.consumer:
            self.consumer.stop()
        self.is_running = False

        # Unlock config controls
        self.toggle_btn.configure(text="▶  Start", bg=self.GREEN, activebackground="#248C4E")
        self.bufsize_spin.configure(state="readonly")
        self.camera_dropdown.configure(state="readonly")
        self.policy_dropdown.configure(state="readonly")
        if self.policy_var.get() == POLICY_TIME_EXPIRY:
            self.maxage_spin.configure(state="readonly")

        self.video_label.configure(
            image="", text="Camera stopped\nClick 'Start' to resume",
        )

    def _on_delay_change(self, _=None):
        if self.consumer:
            self.consumer.extra_delay = self.delay_var.get() / 1000.0

    def _on_bufsize_change(self):
        pass  # Takes effect on next start

    # ── UI Update Loop ──

    def _update_ui(self):
        """Called every 33ms (~30Hz) to refresh UI from thread data."""
        if self.is_running:
            self._update_stats()
            self._update_video()
            self._update_pipeline_indicators()

        self.root.after(33, self._update_ui)

    def _update_stats(self):
        if not self.producer or not self.consumer:
            return

        policy = self.producer.policy

        # Buffer level
        buf_level = self.frame_buffer.qsize()
        self.buffer_text.configure(text=f"{buf_level} / {self.buffer_size}")
        self.buffer_bar["value"] = buf_level

        ratio = buf_level / max(self.buffer_size, 1)
        if ratio > 0.8:
            self.buffer_text.configure(fg=self.RED)
        elif ratio > 0.5:
            self.buffer_text.configure(fg=self.ACCENT)
        else:
            self.buffer_text.configure(fg=self.GREEN)

        # Dropped frames (producer side)
        self.dropped_text.configure(text=f"Dropped: {self.producer.dropped_frames}")

        # Skipped frames (consumer side — time-based and newest-only)
        self.skipped_text.configure(text=f"Skipped: {self.consumer.skipped_frames}")

        # Backpressure blocked time
        if policy == POLICY_BACKPRESSURE:
            self.blocked_text.configure(
                text=f"Producer blocked: {self.producer.blocked_time_ms:.0f}ms total"
            )
        else:
            self.blocked_text.configure(text="")

        # FPS
        self.capture_fps_label.configure(text=f"{self.producer.capture_fps:.1f}")
        self.process_fps_label.configure(text=f"{self.consumer.processing_fps:.1f}")

        if self.consumer.processing_fps > 0 and self.producer.capture_fps > 0:
            if self.consumer.processing_fps < self.producer.capture_fps * 0.5:
                self.process_fps_label.configure(fg=self.RED)
            else:
                self.process_fps_label.configure(fg=self.BLUE)

        # Frame counters
        self.frame_counter_label.configure(
            text=f"Frames: captured={self.producer.frame_count}  "
                 f"processed={self.consumer.processed_count}"
        )

    def _update_video(self):
        """Pull latest detection result and display."""
        try:
            result: DetectionResult = self.result_queue.get_nowait()
        except queue.Empty:
            return

        if result.annotated_image is not None:
            img = result.annotated_image
            h, w = img.shape[:2]
            max_w, max_h = 640, 480
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            photo = ImageTk.PhotoImage(Image.fromarray(img_resized))
            self.video_label.configure(image=photo, text="")
            self.video_label._photo = photo

        # Update detection text
        self.det_text.configure(state="normal")
        self.det_text.delete("1.0", "end")

        if result.labels:
            lines = []
            for label, conf in zip(result.labels, result.confidences):
                lines.append(f"  {label}: {conf:.1%}")
            text = f"Frame #{result.frame_id}\n"
            text += f"Inference: {result.inference_time_ms:.0f}ms\n"
            text += f"Objects ({len(result.labels)}):\n"
            text += "\n".join(lines[:8])
            if len(result.labels) > 8:
                text += f"\n  ...+{len(result.labels) - 8} more"
        else:
            text = f"Frame #{result.frame_id}\nNo objects detected"

        self.det_text.insert("1.0", text)
        self.det_text.configure(state="disabled")

    def _update_pipeline_indicators(self):
        """Color pipeline stages based on actual system state."""
        if not self.is_running or not self.producer or not self.consumer:
            for lbl in self.pipeline_labels:
                lbl.configure(bg=self.BG_CARD)
            return

        buf_ratio = self.frame_buffer.qsize() / max(self.buffer_size, 1)
        cap_fps = self.producer.capture_fps
        proc_fps = self.consumer.processing_fps
        policy = self.producer.policy

        # Camera: green if capturing
        # For backpressure, turns orange/red if producer is being throttled
        if policy == POLICY_BACKPRESSURE and cap_fps > 0:
            if proc_fps > 0 and cap_fps < proc_fps * 1.5:
                # Camera throttled down close to consumer speed
                self.pipeline_labels[0].configure(bg=self.ACCENT)
            else:
                self.pipeline_labels[0].configure(bg=self.GREEN)
        elif cap_fps > 0:
            self.pipeline_labels[0].configure(bg=self.GREEN)
        else:
            self.pipeline_labels[0].configure(bg=self.RED)

        # Buffer: based on fill level
        if buf_ratio > 0.8:
            self.pipeline_labels[1].configure(bg=self.RED)
        elif buf_ratio > 0.5:
            self.pipeline_labels[1].configure(bg=self.ACCENT)
        else:
            self.pipeline_labels[1].configure(bg=self.GREEN)

        # Model: based on whether artificial delay is choking it
        if proc_fps > 0:
            if self.consumer.extra_delay > 0.1:
                self.pipeline_labels[2].configure(bg=self.RED)
            elif self.consumer.extra_delay > 0.03:
                self.pipeline_labels[2].configure(bg=self.ACCENT)
            else:
                self.pipeline_labels[2].configure(bg=self.GREEN)
        else:
            self.pipeline_labels[2].configure(bg=self.BG_CARD)

        # Output: green if results are flowing
        if self.consumer.processed_count > 0:
            self.pipeline_labels[3].configure(bg=self.GREEN)
        else:
            self.pipeline_labels[3].configure(bg=self.BG_CARD)

    def destroy(self):
        self._stop()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = AcquisitionDemoApp(root)

    def on_close():
        app.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.destroy()


if __name__ == "__main__":
    main()
"""
Image Data Explorer
-------------------
A tkinter tool for visually exploring image datasets using embeddings.

Dependencies:
    pip install torch torchvision numpy scikit-learn umap-learn matplotlib Pillow scipy

Usage:
    python image_explorer.py
"""
from __future__ import annotations
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Embedding backbone – ResNet‑18 (pretrained) with the classifier head lopped
# off so we get a 512‑d feature vector per image.
# ---------------------------------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif",
}


def _build_feature_extractor():
    """Return a frozen ResNet‑18 that outputs 512‑d embeddings."""
    weights = models.ResNet18_Weights.DEFAULT
    base = models.resnet18(weights=weights)
    # Remove the final FC layer → output of avgpool is (B, 512, 1, 1)
    extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
    extractor.eval()
    extractor.to(_DEVICE)
    for p in extractor.parameters():
        p.requires_grad = False
    return extractor, weights.transforms()


def collect_image_paths(root_dir: str) -> list[str]:
    """Recursively find all image files under *root_dir*."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, fn))
    return paths


def extract_embeddings(
    paths: list[str],
    extractor: nn.Module,
    preprocess,
    batch_size: int = 32,
    progress_cb=None,
):
    """Return an (N, 512) numpy array of embeddings for the given image paths."""
    all_feats = []
    total = len(paths)
    for i in range(0, total, batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception:
                # Fallback: black image so indices stay aligned
                imgs.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(imgs).to(_DEVICE)
        with torch.no_grad():
            feats = extractor(batch_tensor).cpu().numpy()
        all_feats.append(feats)
        if progress_cb:
            progress_cb(min(i + batch_size, total), total)
    return np.concatenate(all_feats, axis=0)


# ---------------------------------------------------------------------------
# Dimensionality‑reduction helpers
# ---------------------------------------------------------------------------

def reduce_pca(feats: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(feats)


def reduce_tsne(feats: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE
    perp = min(30, max(5, len(feats) - 1))
    return TSNE(
        n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto",
    ).fit_transform(feats)


def reduce_umap(feats: np.ndarray) -> np.ndarray:
    import umap
    n_neighbors = min(15, max(2, len(feats) - 1))
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(feats)


REDUCERS = {
    "PCA": reduce_pca,
    "t-SNE": reduce_tsne,
    "UMAP": reduce_umap,
}


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class ImageExplorerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Data Explorer")
        self.root.geometry("1280x780")
        self.root.minsize(960, 600)

        # State
        self.image_paths: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.coords_2d: np.ndarray | None = None
        self.kdtree: cKDTree | None = None
        self.scatter = None
        self._last_highlight_idx: int | None = None
        self._highlight_dot = None
        self._processing = False

        # Lazy‑load heavy model only once
        self._extractor = None
        self._preprocess = None

        self._build_ui()

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self):
        # -- Top control bar --
        ctrl = ttk.Frame(self.root, padding=6)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Root directory:").pack(side=tk.LEFT)
        self.dir_var = tk.StringVar()
        self.dir_entry = ttk.Entry(ctrl, textvariable=self.dir_var, width=50)
        self.dir_entry.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(ctrl, text="Browse …", command=self._browse).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(ctrl, text="Projection:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="PCA")
        method_combo = ttk.Combobox(
            ctrl, textvariable=self.method_var, values=list(REDUCERS.keys()),
            state="readonly", width=8,
        )
        method_combo.pack(side=tk.LEFT, padx=4)

        self.process_btn = ttk.Button(ctrl, text="Process", command=self._on_process)
        self.process_btn.pack(side=tk.LEFT, padx=4)

        self.reproject_btn = ttk.Button(ctrl, text="Re‑project", command=self._on_reproject, state=tk.DISABLED)
        self.reproject_btn.pack(side=tk.LEFT, padx=4)

        # -- Status bar --
        status_frame = ttk.Frame(self.root, padding=(6, 2))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Select a root directory and click Process.")
        ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.progress = ttk.Progressbar(status_frame, length=220, mode="determinate")
        self.progress.pack(side=tk.RIGHT)

        # -- Main paned area --
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: scatter plot
        left = ttk.Frame(pane)
        pane.add(left, weight=3)

        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor="#f0f0f0")
        self.ax_scatter = self.fig.add_subplot(111)
        self.ax_scatter.set_title("Embedding space", fontsize=11)
        self.ax_scatter.set_xticks([])
        self.ax_scatter.set_yticks([])
        self.canvas_scatter = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas_scatter.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_scatter.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas_scatter.mpl_connect("button_press_event", self._on_click)

        # Right: image preview
        right = ttk.Frame(pane)
        pane.add(right, weight=2)

        self.fig_preview = Figure(figsize=(4, 4), dpi=100, facecolor="#f0f0f0")
        self.ax_preview = self.fig_preview.add_subplot(111)
        self.ax_preview.set_xticks([])
        self.ax_preview.set_yticks([])
        self.ax_preview.set_title("Hover / click a point", fontsize=10, color="gray")
        self.canvas_preview = FigureCanvasTkAgg(self.fig_preview, master=right)
        self.canvas_preview.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Info label below preview
        self.info_var = tk.StringVar()
        info_lbl = ttk.Label(right, textvariable=self.info_var, wraplength=380,
                             anchor=tk.W, justify=tk.LEFT, padding=4)
        info_lbl.pack(fill=tk.X)

    # ---- Callbacks ---------------------------------------------------------

    def _browse(self):
        d = filedialog.askdirectory(title="Select image dataset root")
        if d:
            self.dir_var.set(d)

    def _on_process(self):
        root_dir = self.dir_var.get().strip()
        if not root_dir or not os.path.isdir(root_dir):
            messagebox.showwarning("Invalid path", "Please select a valid directory.")
            return
        if self._processing:
            return
        self._processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.reproject_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._process_thread, args=(root_dir,), daemon=True).start()

    def _process_thread(self, root_dir: str):
        try:
            self._set_status("Scanning for images …")
            paths = collect_image_paths(root_dir)
            if not paths:
                self._set_status("No images found.")
                return
            self.image_paths = paths
            self._set_status(f"Found {len(paths)} images. Loading model …")

            # Lazy‑init backbone
            if self._extractor is None:
                self._extractor, self._preprocess = _build_feature_extractor()

            self._set_status("Extracting embeddings …")
            self.embeddings = extract_embeddings(
                paths, self._extractor, self._preprocess,
                progress_cb=self._embedding_progress,
            )

            method = self.method_var.get()
            self._set_status(f"Running {method} …")
            self.coords_2d = REDUCERS[method](self.embeddings)
            self.kdtree = cKDTree(self.coords_2d)

            self.root.after(0, self._draw_scatter)
            self._set_status(f"Done – {len(paths)} images projected with {method}.")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self._processing = False
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.reproject_btn.config(state=tk.NORMAL))

    def _on_reproject(self):
        """Re‑run only the projection (skip embedding extraction)."""
        if self.embeddings is None:
            messagebox.showinfo("Nothing to project", "Process a directory first.")
            return
        if self._processing:
            return
        self._processing = True
        self.reproject_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._reproject_thread, daemon=True).start()

    def _reproject_thread(self):
        try:
            method = self.method_var.get()
            self._set_status(f"Running {method} …")
            self.coords_2d = REDUCERS[method](self.embeddings)
            self.kdtree = cKDTree(self.coords_2d)
            self.root.after(0, self._draw_scatter)
            self._set_status(f"Re‑projected {len(self.image_paths)} images with {method}.")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self._processing = False
            self.root.after(0, lambda: self.reproject_btn.config(state=tk.NORMAL))

    # ---- Drawing -----------------------------------------------------------

    def _draw_scatter(self):
        self.ax_scatter.clear()
        if self.coords_2d is None:
            return

        xs, ys = self.coords_2d[:, 0], self.coords_2d[:, 1]

        # Colour by subdirectory (gives a sense of class / folder structure)
        root = self.dir_var.get().strip()
        labels = []
        for p in self.image_paths:
            rel = os.path.relpath(os.path.dirname(p), root)
            labels.append(rel if rel != "." else "(root)")
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        colours = [cmap(label_to_idx[l]) for l in labels]

        self.scatter = self.ax_scatter.scatter(
            xs, ys, c=colours, s=18, alpha=0.75, edgecolors="white", linewidths=0.3,
        )
        self._highlight_dot = self.ax_scatter.scatter([], [], s=120, facecolors="none",
                                                       edgecolors="red", linewidths=2, zorder=5)
        self.ax_scatter.set_title(f"{self.method_var.get()} projection  ({len(self.image_paths)} images)", fontsize=11)
        self.ax_scatter.set_xticks([])
        self.ax_scatter.set_yticks([])

        # Simple legend (show at most 15 entries)
        if 1 < len(unique_labels) <= 15:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=cmap(label_to_idx[l]), markersize=7, label=l)
                       for l in unique_labels]
            self.ax_scatter.legend(handles=handles, fontsize=7, loc="best",
                                  framealpha=0.7, ncol=1 + len(unique_labels) // 10)

        self.fig.tight_layout()
        self.canvas_scatter.draw_idle()
        self._last_highlight_idx = None

    def _show_image(self, idx: int):
        """Display the image at *idx* on the right preview axes."""
        if idx == self._last_highlight_idx:
            return
        self._last_highlight_idx = idx
        path = self.image_paths[idx]

        # Update highlight ring
        pt = self.coords_2d[idx]
        self._highlight_dot.set_offsets([pt])
        self.canvas_scatter.draw_idle()

        # Show image
        try:
            img = Image.open(path).convert("RGB")
            # Resize for fast display while keeping aspect
            img.thumbnail((512, 512), Image.LANCZOS)
            arr = np.asarray(img)
        except Exception:
            arr = np.zeros((64, 64, 3), dtype=np.uint8)

        self.ax_preview.clear()
        self.ax_preview.imshow(arr)
        self.ax_preview.set_xticks([])
        self.ax_preview.set_yticks([])

        root = self.dir_var.get().strip()
        rel = os.path.relpath(path, root)
        self.ax_preview.set_title(os.path.basename(path), fontsize=9)
        self.info_var.set(rel)
        self.fig_preview.tight_layout()
        self.canvas_preview.draw_idle()

    # ---- Mouse interaction -------------------------------------------------

    def _nearest_idx(self, event) -> int | None:
        """Return index of the nearest point to the mouse, or None."""
        if self.kdtree is None or event.xdata is None or event.ydata is None:
            return None
        dist, idx = self.kdtree.query([event.xdata, event.ydata])
        # Threshold: ignore if too far (adapt to axis range)
        xr = self.ax_scatter.get_xlim()
        yr = self.ax_scatter.get_ylim()
        diag = np.hypot(xr[1] - xr[0], yr[1] - yr[0])
        if dist > diag * 0.03:
            return None
        return int(idx)

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax_scatter:
            return
        idx = self._nearest_idx(event)
        if idx is not None:
            self._show_image(idx)

    def _on_click(self, event):
        if event.inaxes != self.ax_scatter:
            return
        idx = self._nearest_idx(event)
        if idx is not None:
            self._show_image(idx)

    # ---- Helpers -----------------------------------------------------------

    def _embedding_progress(self, done: int, total: int):
        pct = done / total * 100
        self.root.after(0, lambda: self.progress.config(value=pct))
        self.root.after(0, lambda: self.status_var.set(
            f"Extracting embeddings … {done}/{total}"))

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageExplorerApp(root)
    root.mainloop()

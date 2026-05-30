# Step 6: Streamlit App

**File:** `steel_defect/app.py`
**Placeholders:** APP-1

## Goal

Wire the prediction results into the Streamlit UI by filling the stats column
placeholders. The image display and Grad-CAM overlay are already handled
by the scaffold.

## What's Already Provided

The app is almost entirely scaffold code. It handles:

- **Page config** — Title, icon, wide layout
- **Session state** — History tracking and auto-inspect toggle
- **Sidebar** — Grad-CAM opacity slider, category selector, auto-inspect speed, history summary
- **Model loading** — Cached with `@st.cache_resource`
- **Image display** — Reads and shows the input image
- **Grad-CAM overlay** — Generates and displays the heatmap
- **Control buttons** — Auto inspect, single inspect, stop
- **History display** — Shows recent predictions with icons
- **Auto-inspect loop** — Cycles through images in a category

Your job is to fill the **stats column** — the right-hand panel
that shows the prediction label, confidence, latency, and per-class scores.

## APP-1: Display Prediction Stats

**Location:** Inside `display_result(result, image, source)`, after the Grad-CAM section

### What to do

The function receives a `result` dict from `predictor.predict()`. Extract the
fields and fill the pre-defined placeholder widgets:

1. **Extract values from the result dict:**

    ```python
    label = result["label"]
    confidence = result["confidence"]
    latency_ms = result["latency_ms"]
    ```

2. **Section header:**

    ```python
    stats_header.subheader("Prediction")
    ```

3. **Predicted label with conditional color:**
    - If `label == "no_defect"`: use `stats_label.success(f"✅ **{label}**")`
    - Otherwise: use `stats_label.error(f"🔴 **{label}**")`

    This gives a green banner for normal samples and a red one for defects.

4. **Confidence metric:**

    ```python
    stats_confidence.metric("Confidence", f"{confidence:.1%}")
    ```

    The `:.1%` format turns `0.934` into `"93.4%"`.

5. **Latency metric:**

    ```python
    stats_latency.metric("Latency", f"{latency_ms:.0f} ms")
    ```

6. **Per-class score bars:**

    ```python
    with stats_scores.container():
        for class_name, score in result["class_scores"].items():
            st.progress(score, text=f"{class_name}: {score:.1%}")
    ```

    This creates a progress bar for each class showing its probability.

### Key concepts

- **`st.empty()` placeholders** — The widgets `stats_header`, `stats_label`, etc.
  are `st.empty()` instances created earlier in the file. You fill them by calling
  display methods on them (`.subheader()`, `.metric()`, `.success()`, `.error()`).
  This pattern allows updating content in place without appending new elements.
- **`placeholder.container()`** — Groups multiple widgets into a single placeholder
  slot. Without it, calling `st.progress` inside `stats_scores` multiple times
  would only show the last one.
- **`st.metric(label, value)`** — Displays a large number with a label.
  Commonly used for KPIs and measurements.
- **`st.progress(value, text=...)`** — Draws a horizontal bar from 0.0 to 1.0.
  The `text` parameter displays an overlay label.
- **`st.success()` / `st.error()`** — Colored alert banners. Success is green,
  error is red. Using bold markdown (`**text**`) makes the label stand out.
- **f-string formatting** — `:.1%` formats as percentage with one decimal.
  `:.0f` formats as integer (no decimals).

### Common mistakes

- Forgetting `with stats_scores.container():` — Only the last progress bar appears
- Using `st.write()` instead of the placeholder methods — Creates new elements
  at the bottom of the page instead of filling the stats column
- Referencing `result["class_scores"]` keys in wrong order — The dict preserves
  insertion order from `CLASS_NAMES`, so it displays correctly by default

## Verification

```bash
streamlit run steel_defect/app.py
```

You should see:

- Three-column layout: input image | Grad-CAM | prediction stats
- Green "no_defect" or red defect label with the class name
- Confidence and latency metrics
- Five progress bars showing per-class probabilities
- Inspection history at the bottom

Try both **Single Inspect** and **Auto Inspect** modes.

## Notebook Reference

This step doesn't have a direct notebook equivalent since the notebook uses
`matplotlib` for visualization, not Streamlit. Refer to the Streamlit tutorials
for `st.metric`, `st.progress`, `st.success`, and `st.empty` placeholder patterns.

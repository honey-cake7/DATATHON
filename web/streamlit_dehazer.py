#!/usr/bin/env python
"""
DehazeFormer Dashboard - Dehazing, Hazing & Dataset Evaluation
"""

import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import io
import datetime
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.architectures import DehazeFormer
from src.haze_generation.generator import HazeGenerator
from skimage.metrics import structural_similarity as ssim_skimage

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="DehazeFormer Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(model_path: str, device: str):
    """Load DehazeFormer model with weights."""
    model = DehazeFormer(
        in_channels=3, num_blocks=12, embed_dim=64,
        num_heads=8, window_size=8, mlp_ratio=4.0
    )
    if model_path and Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            st.warning(f"Could not load weights: {e}")
    model.to(device)
    model.eval()
    return model


def discover_models():
    """Discover all pretrained models in models/ folder."""
    models_dir = project_root / "models"
    models = {}
    if models_dir.exists():
        for folder in sorted(models_dir.iterdir()):
            if folder.is_dir():
                pth_files = sorted(list(folder.glob("*.pth")) + list(folder.glob("*.pt")))
                if pth_files:
                    models[folder.name] = {
                        'path': folder,
                        'files': [f.name for f in pth_files]
                    }
    return models


def read_image_rgb(path) -> np.ndarray:
    """Read image from path and return as RGB uint8 numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB uint8."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def dehaze_image(image_rgb: np.ndarray, model, device: str) -> np.ndarray:
    """
    Dehaze a single RGB image. Returns RGB uint8 result.
    Color-preserving pipeline: RGB in -> RGB out (no BGR conversion anywhere).
    """
    image_rgb = ensure_rgb_uint8(image_rgb)
    h, w = image_rgb.shape[:2]

    # Resize to model input size
    resized = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0,1] float32, HWC -> CHW
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    # CHW -> HWC, clip, to uint8
    result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # Resize back to original dimensions
    if h != 256 or w != 256:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    return result


def calc_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """Calculate SSIM between two RGB images (resized to match if needed)."""
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    gt_f = gt.astype(np.float32)
    pred_f = pred.astype(np.float32)
    ssim_vals = []
    for c in range(3):
        s = ssim_skimage(gt_f[:, :, c], pred_f[:, :, c], data_range=255.0)
        ssim_vals.append(s)
    return float(np.mean(ssim_vals))


def calc_psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """Calculate PSNR between two RGB images (resized to match if needed)."""
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
    gt_f = gt.astype(np.float64)
    pred_f = pred.astype(np.float64)
    mse = np.mean((gt_f - pred_f) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def get_dataset_pairs():
    """
    Return dict of dataset_name -> list of (hazy_path, gt_path) tuples.
    Handles the three known folder structures:
      Dense_Haze:  GT/01_GT.png  <->  hazy/01_hazy.png
      I-HAZE:      {split}/clear/ih1.png  <->  {split}/hazy/ih1_hazy.png
      NH-HAZE:     NH-GT/01_GT.png  <->  NH-hazy/01_hazy.png
    """
    data_dir = project_root / "data"
    datasets = {}

    # --- Dense_Haze ---
    dh_gt = data_dir / "Dense_Haze" / "GT"
    dh_hazy = data_dir / "Dense_Haze" / "hazy"
    if dh_gt.exists() and dh_hazy.exists():
        pairs = []
        for gt_file in sorted(dh_gt.glob("*_GT.png")):
            num = gt_file.stem.replace("_GT", "")
            hazy_file = dh_hazy / f"{num}_hazy.png"
            if hazy_file.exists():
                pairs.append((hazy_file, gt_file))
        if pairs:
            datasets["DENSEHAZE"] = pairs

    # --- I-HAZE (combine train + val + test splits) ---
    ihaze_base = data_dir / "I-HAZE" / "I-HAZE"
    if ihaze_base.exists():
        pairs = []
        for split in ["train", "val", "test"]:
            clear_dir = ihaze_base / split / "clear"
            hazy_dir = ihaze_base / split / "hazy"
            if clear_dir.exists() and hazy_dir.exists():
                for gt_file in sorted(clear_dir.glob("*.png")):
                    hazy_file = hazy_dir / f"{gt_file.stem}_hazy.png"
                    if hazy_file.exists():
                        pairs.append((hazy_file, gt_file))
        if pairs:
            datasets["IHAZE"] = pairs

    # --- NH-HAZE ---
    nh_gt = data_dir / "NH-HAZE" / "NH-HAZE" / "NH-GT"
    nh_hazy = data_dir / "NH-HAZE" / "NH-HAZE" / "NH-hazy"
    if nh_gt.exists() and nh_hazy.exists():
        pairs = []
        for gt_file in sorted(nh_gt.glob("*_GT.png")):
            num = gt_file.stem.replace("_GT", "")
            hazy_file = nh_hazy / f"{num}_hazy.png"
            if hazy_file.exists():
                pairs.append((hazy_file, gt_file))
        if pairs:
            datasets["NHHAZE"] = pairs

    return datasets


def evaluate_dataset(dataset_name, pairs, model, device, progress_callback=None):
    """Evaluate model on a dataset. Returns list of per-image dicts."""
    results = []
    for idx, (hazy_path, gt_path) in enumerate(pairs):
        try:
            hazy_img = read_image_rgb(hazy_path)
            gt_img = read_image_rgb(gt_path)
            dehazed = dehaze_image(hazy_img, model, device)

            # Resize dehazed to match GT dimensions
            if gt_img.shape != dehazed.shape:
                dehazed = cv2.resize(dehazed, (gt_img.shape[1], gt_img.shape[0]))

            ssim_val = calc_ssim(gt_img, dehazed)
            psnr_val = calc_psnr(gt_img, dehazed)

            results.append({
                "image": hazy_path.name,
                "SSIM": round(ssim_val, 6),
                "PSNR_dB": round(psnr_val, 4),
            })
        except Exception as e:
            results.append({
                "image": hazy_path.name,
                "SSIM": None,
                "PSNR_dB": None,
                "error": str(e),
            })

        if progress_callback:
            progress_callback(idx + 1, len(pairs))

    return results


def write_log_file(dataset_name, results, model_name, log_dir):
    """Write a dedicated evaluation log file for one dataset."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset_name}_evaluation.log"

    valid = [r for r in results if r["SSIM"] is not None]
    avg_ssim = np.mean([r["SSIM"] for r in valid]) if valid else 0
    avg_psnr = np.mean([r["PSNR_dB"] for r in valid]) if valid else 0

    lines = []
    lines.append("=" * 65)
    lines.append(f"  EVALUATION LOG - {dataset_name}")
    lines.append("=" * 65)
    lines.append(f"  Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Model      : {model_name}")
    lines.append(f"  Dataset    : {dataset_name}")
    lines.append(f"  Images     : {len(results)} total, {len(valid)} successful")
    lines.append(f"  Avg SSIM   : {avg_ssim:.6f}")
    lines.append(f"  Avg PSNR   : {avg_psnr:.4f} dB")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"{'Image':<30}  {'SSIM':>10}  {'PSNR (dB)':>10}")
    lines.append("-" * 55)

    for r in results:
        name = r["image"]
        if r["SSIM"] is not None:
            lines.append(f"{name:<30}  {r['SSIM']:>10.6f}  {r['PSNR_dB']:>10.4f}")
        else:
            lines.append(f"{name:<30}  {'ERROR':>10}  {r.get('error', '?')}")

    lines.append("-" * 55)
    lines.append(f"{'AVERAGE':<30}  {avg_ssim:>10.6f}  {avg_psnr:>10.4f}")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"**Device:** {device.upper()}")

    st.divider()
    st.subheader("Model Selection")
    available_models = discover_models()

    model_path = None
    model_label = "none"

    if available_models:
        total = sum(len(v['files']) for v in available_models.values())
        st.success(f"Found {total} pretrained models")

        selected_folder = st.selectbox("Dataset:", list(available_models.keys()))
        selected_model = st.selectbox("Model:", available_models[selected_folder]['files'])

        model_path = str(available_models[selected_folder]['path'] / selected_model)
        model_label = f"{selected_folder}/{selected_model}"
        st.info(f"**{model_label}**")
    else:
        st.warning("No models found in models/ folder")

    st.divider()
    st.subheader("About")
    st.write("""
    **DehazeFormer** — Transformer-based dehazing  
    - 787,907 parameters  
    - Window-based attention  
    - CPU & GPU support
    """)

# ============================================================================
# LOAD MODEL
# ============================================================================

st.title("🌫️ DehazeFormer Dashboard")
st.markdown("*Dehaze images, generate synthetic haze, and evaluate on benchmark datasets*")
st.divider()

if model_path:
    model = load_model(model_path, device)
else:
    st.error("No model selected. Pick one from the sidebar.")
    model = None

if not model:
    st.stop()

# ============================================================================
# TABS
# ============================================================================

tab_dehaze, tab_haze, tab_eval = st.tabs([
    "🌞 Dehaze Image",
    "🌫️ Generate Haze",
    "📊 Dataset Evaluation (SSIM / PSNR)"
])

# ============================================================================
# TAB 1 - DEHAZE
# ============================================================================

with tab_dehaze:
    col_up, col_res = st.columns(2)

    with col_up:
        st.subheader("Upload Hazy Image")
        uploaded = st.file_uploader(
            "Choose image", type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="dehaze_upload"
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            img_np = np.array(pil_img)
            st.image(img_np, caption="Original (hazy)", use_container_width=True)

            if st.button("🚀 Dehaze", use_container_width=True):
                with st.spinner("Dehazing..."):
                    t0 = time.time()
                    dehazed = dehaze_image(img_np, model, device)
                    elapsed = time.time() - t0

                st.session_state["dehaze_original"] = img_np
                st.session_state["dehaze_result"] = dehazed
                st.session_state["dehaze_time"] = elapsed

    with col_res:
        st.subheader("Dehazed Result")
        if "dehaze_result" in st.session_state:
            st.image(
                st.session_state["dehaze_result"],
                caption="Dehazed", use_column_width=True
            )
            st.success(f"Done in {st.session_state['dehaze_time']:.2f}s")

            orig = st.session_state["dehaze_original"]
            dehz = st.session_state["dehaze_result"]
            ssim_v = calc_ssim(orig, dehz)
            psnr_v = calc_psnr(orig, dehz)
            c1, c2 = st.columns(2)
            c1.metric("SSIM (hazy vs dehazed)", f"{ssim_v:.4f}")
            c2.metric("PSNR (hazy vs dehazed)", f"{psnr_v:.2f} dB")

            # Download
            buf = io.BytesIO()
            Image.fromarray(dehz).save(buf, format="PNG")
            st.download_button(
                "⬇️ Download PNG", buf.getvalue(),
                "dehazed.png", "image/png", use_container_width=True
            )
        else:
            st.info("Upload and dehaze an image to see results")

# ============================================================================
# TAB 2 - HAZE GENERATION  (ported from app.py)
# ============================================================================

with tab_haze:
    st.subheader("Add Synthetic Haze to a Clear Image")

    col_cfg, col_preview = st.columns([1, 2])

    with col_cfg:
        haze_upload = st.file_uploader(
            "Upload clear image", type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="haze_upload"
        )
        haze_method = st.selectbox("Haze Method", [
            "atmospheric_scattering",
            "gaussian_blur",
            "depth_progressive"
        ])
        haze_intensity = st.slider("Haze Intensity", 0.1, 1.0, 0.5, 0.05)
        run_dehaze_after = st.checkbox("Also dehaze the hazy result", value=True)
        generate_btn = st.button("⚙️ Generate Haze", use_container_width=True)

    with col_preview:
        if haze_upload and generate_btn:
            clear_pil = Image.open(haze_upload).convert("RGB")
            clear_np = np.array(clear_pil)

            haze_gen = HazeGenerator()
            if haze_method == "atmospheric_scattering":
                hazy_np = haze_gen.atmospheric_scattering(clear_np, beta=haze_intensity)
            elif haze_method == "gaussian_blur":
                hazy_np = haze_gen.gaussian_blur_color_shift(clear_np, intensity=haze_intensity)
            else:
                hazy_np = haze_gen.depth_based_progressive(clear_np, intensity=haze_intensity)

            hazy_np = ensure_rgb_uint8(hazy_np)

            if run_dehaze_after:
                with st.spinner("Dehazing the hazy image..."):
                    dehazed_np = dehaze_image(hazy_np, model, device)

                c1, c2, c3 = st.columns(3)
                c1.image(clear_np, caption="Original Clear", use_container_width=True)
                c2.image(hazy_np, caption="With Haze", use_container_width=True)
                c3.image(dehazed_np, caption="Dehazed", use_container_width=True)

                # Metrics: dehazed vs original clear (ground truth)
                ssim_v = calc_ssim(clear_np, dehazed_np)
                psnr_v = calc_psnr(clear_np, dehazed_np)
                m1, m2 = st.columns(2)
                m1.metric("SSIM (clear vs dehazed)", f"{ssim_v:.4f}")
                m2.metric("PSNR (clear vs dehazed)", f"{psnr_v:.2f} dB")
            else:
                c1, c2 = st.columns(2)
                c1.image(clear_np, caption="Original Clear", use_container_width=True)
                c2.image(hazy_np, caption="With Haze", use_container_width=True)

            # Download hazy
            buf = io.BytesIO()
            Image.fromarray(hazy_np).save(buf, format="PNG")
            st.download_button(
                "⬇️ Download Hazy Image", buf.getvalue(),
                "hazy_output.png", "image/png", use_container_width=True
            )
        elif not haze_upload:
            st.info("Upload a clear image and click Generate Haze")

# ============================================================================
# TAB 3 - DATASET EVALUATION  (SSIM & PSNR with separate log files)
# ============================================================================

with tab_eval:
    st.subheader("Evaluate on Benchmark Datasets")
    st.write(
        "Runs the model on **I-HAZE**, **NH-HAZE**, and **Dense-Haze** datasets, "
        "computes **SSIM** and **PSNR** (dehazed vs ground truth), "
        "and saves a **separate log file** per dataset in `results/logs/`."
    )

    dataset_pairs = get_dataset_pairs()

    if not dataset_pairs:
        st.warning("No datasets found in data/ folder.")
        st.stop()

    # Show available datasets
    for name, pairs in dataset_pairs.items():
        st.write(f"- **{name}**: {len(pairs)} image pairs")

    st.divider()

    datasets_to_run = st.multiselect(
        "Select datasets to evaluate:",
        list(dataset_pairs.keys()),
        default=list(dataset_pairs.keys())
    )

    max_per_dataset = st.number_input(
        "Max images per dataset (0 = all):",
        min_value=0, max_value=500, value=0
    )

    if st.button("🚀 Run Evaluation", use_container_width=True, type="primary"):
        all_summaries = {}
        log_dir = project_root / "results" / "logs"

        for ds_name in datasets_to_run:
            pairs = dataset_pairs[ds_name]
            if max_per_dataset > 0:
                pairs = pairs[:max_per_dataset]

            st.write(f"### Evaluating **{ds_name}** ({len(pairs)} images)...")

            progress = st.progress(0, text=f"{ds_name}: 0/{len(pairs)}")

            def update_progress(done, total, _name=ds_name, _bar=progress):
                _bar.progress(done / total, text=f"{_name}: {done}/{total}")

            t0 = time.time()
            results = evaluate_dataset(ds_name, pairs, model, device, update_progress)
            elapsed = time.time() - t0

            # Write log file
            log_path = write_log_file(ds_name, results, model_label, log_dir)

            valid = [r for r in results if r["SSIM"] is not None]
            avg_ssim = np.mean([r["SSIM"] for r in valid]) if valid else 0
            avg_psnr = np.mean([r["PSNR_dB"] for r in valid]) if valid else 0

            all_summaries[ds_name] = {
                "images": len(results),
                "successful": len(valid),
                "avg_ssim": avg_ssim,
                "avg_psnr": avg_psnr,
                "time_s": elapsed,
                "log_file": str(log_path),
            }

            # Show per-image results
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{ds_name} Avg SSIM", f"{avg_ssim:.4f}")
            c2.metric(f"{ds_name} Avg PSNR", f"{avg_psnr:.2f} dB")
            c3.metric(f"{ds_name} Time", f"{elapsed:.1f}s")

            st.success(f"Log saved: `{log_path.relative_to(project_root)}`")
            st.divider()

        # Summary table across all datasets
        if all_summaries:
            st.subheader("Overall Summary")
            summary_rows = []
            for ds, info in all_summaries.items():
                summary_rows.append({
                    "Dataset": ds,
                    "Images": info["images"],
                    "Avg SSIM": f"{info['avg_ssim']:.4f}",
                    "Avg PSNR (dB)": f"{info['avg_psnr']:.2f}",
                    "Time (s)": f"{info['time_s']:.1f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # Download log buttons
            st.write("**Log files saved in:** `results/logs/`")
            for ds_name in datasets_to_run:
                lp = log_dir / f"{ds_name}_evaluation.log"
                if lp.exists():
                    st.download_button(
                        f"⬇️ {ds_name}_evaluation.log",
                        lp.read_text(encoding="utf-8"),
                        f"{ds_name}_evaluation.log",
                        "text/plain",
                        key=f"dl_{ds_name}"
                    )

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("DehazeFormer Dashboard • PyTorch & Streamlit")

# Image Dehazing

**Objective**: Build a complete image dehazing pipeline to improve visibility in hazy/smoggy images, with applications in road safety, traffic surveillance, and smart-city infrastructure.

## Project Overview

This project uses **DehazeFormer**, a transformer-based architecture, to remove haze from images. It includes 19 pretrained model checkpoints (across 4 dataset categories), an interactive **Streamlit web dashboard** for dehazing, synthetic haze generation, and automated **SSIM / PSNR evaluation** on three benchmark datasets — I-HAZE, NH-HAZE, and Dense-Haze — with per-dataset log files.

### Key Features

- **DehazeFormer Architecture** — Transformer with window-based multi-head self-attention (787,907 parameters)
- **19 Pretrained Models** — Weights for indoor, outdoor, RESIDE-6K, and RS-Haze datasets (sizes: tiny → large)
- **Web Dashboard** — Streamlit app with three tabs: Dehaze, Haze Generation, Dataset Evaluation
- **Haze Generation** — Atmospheric scattering model, Gaussian blur + color shift, depth-based progressive haze
- **Evaluation Metrics** — SSIM and PSNR computed per-image against ground truth, with separate log files per dataset
- **Additional Architectures** — DehazeNet, AODNet, PFFNet, Dark Channel Prior also implemented in `src/models/`

## Directory Structure

```
.
├── config.yaml                         # Model & pipeline configuration
├── requirements.txt                    # Python dependencies
├── README.md
│
├── web/
│   └── streamlit_dehazer.py            # Main dashboard (dehaze + haze gen + evaluation)
│
├── src/
│   ├── models/
│   │   └── architectures.py            # DehazeFormer, DehazeNet, AODNet, PFFNet, DCP
│   ├── inference/
│   │   └── dehaze.py                   # Inference pipeline
│   ├── metrics/
│   │   └── evaluation.py               # SSIM, PSNR, MSE, MAE calculation
│   └── haze_generation/
│       └── generator.py                # Synthetic haze generation (3 methods)
│
├── models/                             # Pretrained DehazeFormer weights (.pth)
│   ├── indoor/     (7 models: t, s, m, b, d, w, l)
│   ├── outdoor/    (4 models: t, s, m, b)
│   ├── reside6k/   (4 models: t, s, m, b)
│   └── rshaze/     (4 models: t, s, m, b)
│
├── data/                               # Benchmark datasets
│   ├── Dense_Haze/  (GT/ + hazy/)      — 55 image pairs
│   ├── I-HAZE/      (train/val/test)   — 35 image pairs
│   └── NH-HAZE/     (NH-GT/ + NH-hazy/)— 55 image pairs
│
├── results/
│   └── logs/                           # Evaluation log files (generated at runtime)
│       ├── DENSEHAZE_evaluation.log
│       ├── IHAZE_evaluation.log
│       └── NHHAZE_evaluation.log
│
├── notebooks/
│   └── 01_image_dehazing_pipeline.ipynb
│
└── scripts/
    └── utilities.py
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA (optional — works fully on CPU)

### Steps

1. **Clone the repository**:

```bash
git clone https://github.com/honey-cake7/DATATHON.git
cd DATATHON
```

2. **Create virtual environment** (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Download pretrained weights** (if not included):

   Place `.pth` files in `models/<dataset>/` folders. Expected structure:

   ```
   models/
   ├── indoor/dehazeformer-t.pth  ... dehazeformer-l.pth
   ├── outdoor/dehazeformer-t.pth ... dehazeformer-b.pth
   ├── reside6k/dehazeformer-t.pth ... dehazeformer-b.pth
   └── rshaze/dehazeformer-t.pth  ... dehazeformer-b.pth
   ```

5. **Download datasets** (if not included):
   - [I-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/)
   - [NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)
   - [Dense-Haze](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/)

   Place in `data/` following the directory structure above.

## Usage

### Launch the Web Dashboard

```bash
streamlit run web/streamlit_dehazer.py
```

Opens at **http://localhost:8501** with three tabs:

| Tab                    | Description                                                                       |
| ---------------------- | --------------------------------------------------------------------------------- |
| **Dehaze Image**       | Upload a hazy image → get dehazed output with SSIM/PSNR + PNG download            |
| **Generate Haze**      | Upload a clear image → add synthetic haze (3 methods) → optionally dehaze it back |
| **Dataset Evaluation** | Run SSIM & PSNR on I-HAZE, NH-HAZE, Dense-Haze → saves separate `.log` files      |

### Web Dashboard Screenshots

**Dehaze tab**: Upload → select model from sidebar → click Dehaze → view side-by-side with metrics.

**Haze Generation tab**: Choose method (atmospheric scattering / gaussian blur / depth progressive) and intensity slider.

**Evaluation tab**: Select datasets → click Run Evaluation → per-image SSIM/PSNR table + downloadable log files.

## Pretrained Models

19 DehazeFormer checkpoints organized by training dataset:

| Folder      | Models | Sizes                                              |
| ----------- | ------ | -------------------------------------------------- |
| `indoor/`   | 7      | dehazeformer-t (2.9 MB) → dehazeformer-l (98.2 MB) |
| `outdoor/`  | 4      | dehazeformer-t → dehazeformer-b                    |
| `reside6k/` | 4      | dehazeformer-t → dehazeformer-b                    |
| `rshaze/`   | 4      | dehazeformer-t → dehazeformer-b                    |

Model variants: **t** (tiny), **s** (small), **m** (medium), **b** (base), **d**, **w** (wide), **l** (large).

Select any model from the sidebar dropdown in the dashboard.

## Evaluation Metrics

| Metric   | Description                                                  | Range                                   |
| -------- | ------------------------------------------------------------ | --------------------------------------- |
| **SSIM** | Structural Similarity Index — measures perceptual similarity | 0 to 1 (higher = better)                |
| **PSNR** | Peak Signal-to-Noise Ratio — measures pixel-level fidelity   | dB (higher = better, typical: 15–35 dB) |

Evaluation computes **dehazed vs ground truth** for each image pair. Results are saved as separate log files:

```
results/logs/
├── DENSEHAZE_evaluation.log    # 55 images
├── IHAZE_evaluation.log        # 35 images
└── NHHAZE_evaluation.log       # 55 images
```

Each log contains per-image SSIM/PSNR values and dataset averages.

## Datasets

| Dataset        | Image Pairs | Structure                                   | Description               |
| -------------- | ----------- | ------------------------------------------- | ------------------------- |
| **I-HAZE**     | 35          | train/val/test splits with clear/ and hazy/ | Indoor synthetic haze     |
| **NH-HAZE**    | 55          | NH-GT/ and NH-hazy/                         | Non-homogeneous real haze |
| **Dense-Haze** | 55          | GT/ and hazy/                               | Dense real-world haze     |

## Model Architecture

**DehazeFormer** — Transformer-based image restoration model:

- 12 transformer blocks with window-based attention
- 8 attention heads, 64-dim embeddings
- Multi-scale feature extraction + skip connections
- Input/Output: 3×256×256 RGB
- Parameters: 787,907

Additional architectures available in `src/models/architectures.py`:

- **DehazeNet** — Lightweight CNN (residual blocks)
- **AODNet** — All-in-One encoder-decoder with skip connections
- **PFFNet** — Progressive fusion framework
- **Dark Channel Prior** — Classical non-neural baseline

## Haze Generation

Three synthetic haze methods in `src/haze_generation/generator.py`:

| Method                          | Description                                                           |
| ------------------------------- | --------------------------------------------------------------------- |
| **Atmospheric Scattering**      | Physics-based: $I(x) = J(x) \cdot e^{-\beta d} + A(1 - e^{-\beta d})$ |
| **Gaussian Blur + Color Shift** | Blur + white haze color overlay                                       |
| **Depth-Based Progressive**     | More haze at bottom of image (simulating depth)                       |

Adjustable intensity (0.1–1.0) via the dashboard slider.


## Team

- Aniket Patil
- Sujith Pedapati
- Lakshya Patidar
- Vishwajeet Singh Bhati


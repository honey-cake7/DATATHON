# Image Dehazing - NTIRE Challenge Datathon

**Objective**: Build a complete Image Dehazing pipeline to improve visibility in hazy/smoggy images, with applications in road safety, traffic surveillance, and smart-city infrastructure.

## Project Overview

This project implements multiple deep learning models for image dehazing, benchmarks them against standard datasets (I-Haze, N-Haze, Dense-Haze), and deploys the best model as an interactive web dashboard.

### Key Features

- **Multiple Model Architectures**: Implementation of state-of-the-art dehazing models (DehazeNet, AODNet, PFFNet, or similar)
- **Inference Pipeline**: Clean, modular inference scripts for batch processing
- **Web Dashboard**: Interactive Gradio/Flask app for real-time dehazing
- **Haze Generation**: Pipeline to artificially add haze using image processing techniques
- **Performance Metrics**: PSNR and SSIM evaluation on standard datasets
- **Hallucination Check**: Structural fidelity validation to prevent artificial detail generation

## Directory Structure

```
.
├── data/
│   ├── raw/                    # Original datasets (I-Haze, N-Haze, Dense-Haze)
│   └── processed/              # Preprocessed images
├── models/                     # Pre-trained model weights
├── notebooks/                  # Exploratory notebooks and analysis
├── src/
│   ├── models/                 # Model architecture definitions
│   ├── inference/              # Inference pipelines
│   ├── metrics/                # Evaluation metrics (PSNR, SSIM)
│   └── haze_generation/        # Haze synthesis pipeline
├── web/                        # Web dashboard (Gradio/Flask)
├── scripts/                    # Utility scripts
├── results/                    # Output images and metrics
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration file
└── README.md
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA 11.x (for GPU acceleration, optional)

### Steps

1. **Clone/Navigate to project**:

```bash
cd c:\Coding\DATATHON
```

2. **Create virtual environment** (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Download datasets**:
   - I-Haze: [NTIRE Challenge](https://github.com/cszn/NTIRE2018)
   - N-Haze: [Dataset Link](https://nhaze.github.io/)
   - Dense-Haze: [Dataset Link](https://www.dropbox.com/sh/86b86d3n1g6b9lx/AABQrNqVPrB3xQU4Fs7fFI3Ea)

   Place in `data/raw/`

## Usage

### 1. Model Inference

```bash
python src/inference/dehaze.py --input <hazy_image> --output <output_path> --model <model_name>
```

### 2. Batch Processing

```bash
python scripts/process_dataset.py --input_dir data/raw --output_dir data/processed
```

### 3. Evaluate on Dataset

```bash
python scripts/evaluate_metrics.py --dataset_dir data/processed --output results/metrics.csv
```

### 4. Launch Web Dashboard

```bash
python web/app.py
```

Then navigate to `http://localhost:7860` (Gradio) or `http://localhost:5000` (Flask)

### 5. Generate Haze & Dehaze

```bash
python scripts/haze_generation_demo.py --image <clear_image> --intensity 0.5
```

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level accuracy
- **SSIM** (Structural Similarity Index): Measures perceptual similarity
- **FID** (Fréchet Inception Distance): Optional - measures distribution similarity
- **Hallucination Score**: Manual inspection for artificial detail generation

## Datasets

| Dataset        | Hazy Images | Scene Type            | Link                                                  |
| -------------- | ----------- | --------------------- | ----------------------------------------------------- |
| **I-Haze**     | 25          | Synthetic indoor      | [NTIRE2018](https://github.com/cszn/NTIRE2018)        |
| **N-Haze**     | 45          | Natural outdoor       | [Dataset](https://nhaze.github.io/)                   |
| **Dense-Haze** | 50          | Dense haze conditions | [Dropbox](https://www.dropbox.com/sh/86b86d3n1g6b9lx) |

## Model Selection

You can test multiple models:

1. **DehazeNet** - Lightweight CNN-based approach
2. **AODNet** (All-in-One Dehazing Network) - Multi-scale features
3. **PFFNet** - Progressive fusion framework
4. **Pretrained Models**: YOLOv5, BRISQUE for auxiliary tasks

Modify `config.yaml` to select which models to train/evaluate.

## Results

After running evaluations, results are saved in `results/` directory:

```
results/
├── metrics.csv          # PSNR, SSIM scores per image
├── visualizations/      # Side-by-side comparisons
└── logs/               # Training/inference logs
```

## Bonus Features

### Haze Generation

Generate synthetic haze using:

- Atmospheric scattering model
- Gaussian blur + color shift
- Depth-based progressive haze

### Qualitative Analysis

- Visual comparisons on test images
- Hallucination detection via manual inspection
- Failure case analysis

## Points Distribution

| Category                          | Points  |
| --------------------------------- | ------- |
| Model Selection & Pipeline        | 30      |
| Hallucination Check & Performance | 35      |
| Viva                              | 10      |
| Innovation                        | 10      |
| Bonus/Optional Task               | 10      |
| Web Dashboard                     | 5       |
| **Total**                         | **100** |

## Team

Aniket Patil
Sujith Pedapati
Lakshya Patidar
Vishwajeet Singh Bhati

## References

1. He, K., Sun, J., & Tang, X. (2009). Single image haze removal using dark channel prior.
2. Li, B., Peng, X., Wang, Z., et al. (2019). AODNet: All-in-One Dehazing Network.
3. Dong, H., Pan, Y., Zhang, L., et al. (2020). Multi-scale Boosted Dehazing Network.
4. Choi, L. K., You, J., & Bovik, A. C. (2015). Referenceless Prediction of Perceptual Fog Density.



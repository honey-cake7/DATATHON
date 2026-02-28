# Image Dehazing Datathon - Setup Instructions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets

Download the three benchmark datasets:

- **I-Haze**: Synthetic indoor haze images (25 images)
  - [NTIRE 2018 Challenge](https://github.com/cszn/NTIRE2018)
- **N-Haze**: Natural outdoor haze images (45 images)
  - [Dataset Link](https://nhaze.github.io/)
- **Dense-Haze**: Dense haze conditions (50 images)
  - [Dropbox Link](https://www.dropbox.com/sh/86b86d3n1g6b9lx)

Extract to:

```
data/raw/I-Haze/
data/raw/N-Haze/
data/raw/Dense-Haze/
```

### 3. Run Jupyter Notebook

```bash
jupyter notebook notebooks/01_image_dehazing_pipeline.ipynb
```

This notebook covers:

- Environment setup
- Dataset exploration
- Model selection and implementation
- Inference pipeline
- Metrics evaluation (PSNR, SSIM)
- Haze generation methods
- Dashboard integration

### 4. Launch Web Dashboard

```bash
python web/app.py
```

Navigate to `http://localhost:7860` to access the interactive interface.

### 5. Run Utility Scripts

**Process dataset:**

```bash
python scripts/utilities.py process --input data/raw/I-Haze --output results/dehazed_images
```

**Evaluate metrics:**

```bash
python scripts/utilities.py evaluate --dehazed results/dehazed_images --ground-truth data/ground_truth --output results/metrics.csv
```

**Generate synthetic haze:**

```bash
python scripts/utilities.py generate-haze --input data/raw/clear_images --output data/synthetic_haze --intensity 0.5
```

**Create comparisons:**

```bash
python scripts/utilities.py compare --original data/raw --hazy data/synthetic_haze --dehazed results/dehazed_images --output results/visual_comparisons
```

## Project Structure

```
c:\Coding\DATATHON/
├── data/
│   ├── raw/              # Download datasets here
│   │   ├── I-Haze/
│   │   ├── N-Haze/
│   │   └── Dense-Haze/
│   └── processed/        # Processed images
├── models/               # Pre-trained weights
├── notebooks/            # Jupyter notebooks
│   └── 01_image_dehazing_pipeline.ipynb
├── src/
│   ├── models/          # Model architectures
│   ├── inference/       # Inference pipeline
│   ├── metrics/         # Evaluation metrics
│   └── haze_generation/ # Haze synthesis
├── web/                 # Gradio web dashboard
│   └── app.py
├── scripts/
│   └── utilities.py     # Batch processing tools
├── results/             # Output images & metrics
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── README.md            # Full documentation
```

## Models Available

1. **AODNet** (All-in-One Dehazing Network)
   - Multi-scale feature extraction
   - Skip connections
   - Recommended for balance of speed and quality
2. **DehazeNet**
   - Lightweight residual network
   - Fast inference
   - Good for resource-constrained devices
3. **PFFNet** (Progressive Fusion Framework)
   - Progressive refinement strategy
   - Multi-output supervision
   - Best for high-quality results

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level accuracy (dB)
  - Target: > 25 dB (good), > 35 dB (excellent)
- **SSIM** (Structural Similarity Index): Perceptual quality
  - Target: > 0.85
  - Range: -1 to 1 (1 = identical)

## Deliverables Checklist

- [ ] Model Selection & Pipeline (30 pts)
- [ ] Hallucination Check & Performance (35 pts)
- [ ] Viva Preparation (10 pts)
- [ ] Innovation (10 pts)
- [ ] Bonus Features (10 pts)
- [ ] Web Dashboard (5 pts)

## Troubleshooting

### GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
python scripts/utilities.py process --input data/raw/I-Haze --output results/ --device cpu
```

### Memory Issues

Reduce batch size in `config.yaml`:

```yaml
inference:
  batch_size: 1 # Reduce from default
```

### Image Load Errors

Ensure image paths are correct and formats are supported (JPG, PNG, BMP, TIFF)

## References

1. He et al. (2010) - Single Image Haze Removal Using Dark Channel Prior
2. Li et al. (2017) - AODNet: All-in-One Dehazing Network
3. Dong et al. (2020) - Multi-scale Boosted Dehazing Network
4. NTIRE Challenge: https://www.ntire.org/

## Support

For issues or questions, refer to:

- `README.md` - Full project documentation
- `notebooks/01_image_dehazing_pipeline.ipynb` - Detailed implementation guide
- Configuration files for customization

---

Last Updated: February 2026

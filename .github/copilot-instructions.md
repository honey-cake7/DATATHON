# GitHub Copilot Custom Instructions

This workspace is configured for the Image Dehazing Datathon project.

## Project Context

**Project**: Image Dehazing Pipeline with Deep Learning

- **Goal**: Remove fog/haze from images in North Indian winter conditions
- **Datasets**: I-Haze, N-Haze, Dense-Haze (NTIRE Challenge)
- **Metrics**: PSNR, SSIM evaluation
- **Deliverables**: Models, Web Dashboard, Haze Generation, Performance Analysis

## Repository Structure

```
.
├── src/                          # Source code
│   ├── models/architectures.py   # Model definitions (AODNet, DehazeNet, PFFNet)
│   ├── inference/dehaze.py       # Inference pipeline
│   ├── metrics/evaluation.py     # PSNR/SSIM calculation
│   ├── haze_generation/generator.py  # Synthetic haze synthesis
├── web/app.py                    # Gradio web dashboard
├── notebooks/                    # Jupyter notebooks
├── scripts/utilities.py          # Batch processing utilities
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                      # Full documentation
```

## Key Technologies

- **Deep Learning**: PyTorch, torchvision
- **Image Processing**: OpenCV, scikit-image
- **Web Interface**: Gradio
- **Metrics**: PSNR, SSIM (scikit-image)
- **Data**: NumPy, Pandas

## Important Guidelines

1. **Hallucination Prevention**
   - Models must preserve structural fidelity
   - No artificial detail generation
   - Validate with qualitative assessment

2. **Code Quality**
   - All modules are well-documented
   - Type hints for clarity
   - Exception handling for robustness

3. **File Naming Convention**
   - `dehaze_<imagename>.jpg` for dehazed outputs
   - `haze_<intensity>.jpg` for synthetic haze
   - `comparison_<name>.jpg` for visualizations

4. **Configuration**
   - Modify `config.yaml` for settings (models, paths, parameters)
   - Use `src.inference.dehaze.create_inference_pipeline()` to load from config

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook notebooks/01_image_dehazing_pipeline.ipynb

# Launch web dashboard
python web/app.py

# Quick inference
python quickstart.py infer --input image.jpg --output output.jpg

# Process dataset
python scripts/utilities.py process --input data/raw/I-Haze --output results/

# Evaluate on benchmark
python scripts/utilities.py evaluate --dehazed results/dehazed_images --ground-truth data/ground_truth --output results/metrics.csv
```

## Model Information

| Model     | Parameters | Speed     | Quality   | Use Case               |
| --------- | ---------- | --------- | --------- | ---------------------- |
| AODNet    | ~120K      | Fast      | Good      | Balanced (recommended) |
| DehazeNet | ~65K       | Very Fast | Fair      | Lightweight            |
| PFFNet    | ~85K       | Medium    | Excellent | High quality           |

## Performance Targets

- **PSNR Target**: ≥ 25 dB (good), ≥ 35 dB (excellent)
- **SSIM Target**: ≥ 0.85
- **Inference Time**: < 100ms per image (GPU)
- **Model Size**: < 5MB preferred

## Development Workflow

1. Explore in Jupyter notebook (`notebooks/01_image_dehazing_pipeline.ipynb`)
2. Implement features in `src/` modules
3. Test with batch scripts (`scripts/utilities.py`)
4. Deploy through web dashboard (`web/app.py`)
5. Evaluate metrics on benchmark datasets

## Testing Checklist

- [ ] All models load successfully
- [ ] Inference works (CPU and GPU)
- [ ] Metrics calculate correctly
- [ ] Haze generation produces realistic results
- [ ] Web dashboard launches without errors
- [ ] Batch processing completes successfully

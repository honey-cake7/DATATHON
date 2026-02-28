"""
Example notebook demonstrating the full dehazing workflow.
This shows how to use all components of the pipeline.
"""

# Run the following in Python/Jupyter to test the pipeline:

from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path('.').resolve()))

# ==============================================================================
# 1. IMPORT ALL MODULES
# ==============================================================================

from src.models.architectures import create_model, AODNet, DehazeNet
from src.inference.dehaze import DehazeInference
from src.metrics.evaluation import DehazingMetrics
from src.haze_generation.generator import HazeGenerator
import cv2
import numpy as np

# ==============================================================================
# 2. INITIALIZE COMPONENTS
# ==============================================================================

# Initialize inference pipeline
dehaze_pipeline = DehazeInference(
    model_name="AODNet",
    device="cuda"  # or "cpu"
)

# Initialize metrics calculator
metrics = DehazingMetrics()

# Initialize haze generator
haze_gen = HazeGenerator()

# ==============================================================================
# 3. EXAMPLE 1: SIMPLE DEHAZING
# ==============================================================================

# Load a hazy image
hazy_image_path = "path/to/hazy/image.jpg"
dehazed_image = dehaze_pipeline.infer(hazy_image_path)

# Save result
output_path = "results/dehazed_image.jpg"
cv2.imwrite(output_path, cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR))

print(f"✓ Dehazed image saved: {output_path}")

# ==============================================================================
# 4. EXAMPLE 2: GENERATE AND DEHAZE
# ==============================================================================

# Start with a clear image
clear_image_path = "path/to/clear/image.jpg"
clear_image = cv2.imread(clear_image_path)
clear_image = cv2.cvtColor(clear_image, cv2.COLOR_BGR2RGB)

# Add synthetic haze
hazy_synthetic = haze_gen.atmospheric_scattering(clear_image, beta=0.6)

# Dehaze it
dehazed = dehaze_pipeline.infer(hazy_synthetic)

print("✓ Haze generation and dehazing complete")

# ==============================================================================
# 5. EXAMPLE 3: BATCH PROCESSING
# ==============================================================================

# Process entire directory
results = dehaze_pipeline.batch_infer(
    image_dir="data/raw/I-Haze",
    output_dir="results/dehazed_images",
    verbose=True
)

print(f"✓ Processed {len(results)} images")

# ==============================================================================
# 6. EXAMPLE 4: EVALUATE METRICS
# ==============================================================================

# Evaluate against ground truth
evaluation_results = metrics.evaluate_batch(
    gt_dir="data/ground_truth",
    pred_dir="results/dehazed_images"
)

# Print summary
metrics.print_summary(evaluation_results)

# ==============================================================================
# 7. EXAMPLE 5: COMPARE MODELS
# ==============================================================================

# Test multiple models on same image
test_image = "path/to/test/image.jpg"

for model_name in ["AODNet", "DehazeNet", "PFFNet"]:
    model = DehazeInference(model_name=model_name, device="cuda")
    result = model.infer(test_image)
    cv2.imwrite(f"results/{model_name}_result.jpg", 
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"✓ {model_name} inference complete")

# ==============================================================================
# 8. LAUNCH WEB DASHBOARD
# ==============================================================================

# From command line:
# python web/app.py
# Then visit http://localhost:7860

print("\nTo launch the web dashboard, run:")
print("  python web/app.py")
print("Then navigate to http://localhost:7860")

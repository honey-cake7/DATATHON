"""
Utility scripts for batch processing and evaluation.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List
import csv
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.dehaze import DehazeInference
from src.metrics.evaluation import DehazingMetrics
from src.haze_generation.generator import HazeGenerator


def process_dataset(input_dir: Union[str, Path],
                   output_dir: Union[str, Path],
                   model_name: str = "AODNet",
                   device: str = "cuda") -> None:
    """Process entire dataset with dehazing model."""
    
    print(f"Processing dataset from {input_dir}")
    
    # Initialize inference
    dehaze = DehazeInference(model_name=model_name, device=device)
    dehaze.print_model_info()
    
    # Run batch inference
    results = dehaze.batch_infer(input_dir, output_dir, verbose=True)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} images")
    if results:
        avg_time = np.mean([t for _, t in results])
        print(f"Average inference time: {avg_time:.1f}ms per image")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


def evaluate_metrics(dehazed_dir: Union[str, Path],
                    ground_truth_dir: Union[str, Path],
                    output_csv: Union[str, Path] = "metrics.csv") -> None:
    """Evaluate dehazed images against ground truth."""
    
    print(f"Evaluating dehazed images")
    print(f"  Ground truth: {ground_truth_dir}")
    print(f"  Dehazed: {dehazed_dir}")
    
    metrics = DehazingMetrics()
    
    # Evaluate batch
    results = metrics.evaluate_batch(ground_truth_dir, dehazed_dir)
    
    # Print summary
    metrics.print_summary(results)
    
    # Save to CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'PSNR (dB)', 'SSIM', 'MSE', 'MAE'])
        
        for i, img_name in enumerate(results['images']):
            writer.writerow([
                img_name,
                f"{results['psnr'][i]:.4f}",
                f"{results['ssim'][i]:.4f}",
                f"{results['mse'][i]:.4f}",
                f"{results['mae'][i]:.4f}"
            ])
        
        # Add averages
        if 'avg_psnr' in results:
            writer.writerow([])
            writer.writerow(['AVERAGE', 
                           f"{results['avg_psnr']:.4f}",
                           f"{results['avg_ssim']:.4f}",
                           f"{results['avg_mse']:.4f}",
                           f"{results['avg_mae']:.4f}"])
    
    print(f"\nMetrics saved to: {output_csv}")


def generate_synthetic_haze(input_dir: Union[str, Path],
                           output_dir: Union[str, Path],
                           intensity: float = 0.5,
                           method: str = "atmospheric_scattering") -> None:
    """Generate synthetic haze on dataset."""
    
    print(f"Generating synthetic haze")
    print(f"  Method: {method}")
    print(f"  Intensity: {intensity}")
    
    generator = HazeGenerator()
    generated = generator.batch_generate(
        input_dir, 
        output_dir, 
        beta_values=[intensity],
        method=method
    )
    
    print(f"Generated {len(generated)} hazy images in {output_dir}")


def create_comparison_viz(original_dir: Union[str, Path],
                         hazy_dir: Union[str, Path],
                         dehazed_dir: Union[str, Path],
                         output_dir: Union[str, Path]) -> None:
    """Create side-by-side comparison visualizations."""
    
    original_dir = Path(original_dir)
    hazy_dir = Path(hazy_dir)
    dehazed_dir = Path(dehazed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    original_files = [f for f in original_dir.iterdir() 
                     if f.suffix.lower() in image_extensions]
    
    print(f"Creating comparison visualizations...")
    
    for i, orig_path in enumerate(sorted(original_files)):
        hazy_path = hazy_dir / orig_path.name
        dehazed_path = dehazed_dir / f"{orig_path.stem}_dehazed.png"
        
        if not hazy_path.exists() or not dehazed_path.exists():
            continue
        
        # Load images
        orig = cv2.imread(str(orig_path))
        hazy = cv2.imread(str(hazy_path))
        dehazed = cv2.imread(str(dehazed_path))
        
        # Resize to same height
        h = min(orig.shape[0], hazy.shape[0], dehazed.shape[0])
        w_ratio = h / orig.shape[0]
        
        orig = cv2.resize(orig, (int(orig.shape[1] * w_ratio), h))
        hazy = cv2.resize(hazy, (int(hazy.shape[1] * w_ratio), h))
        dehazed = cv2.resize(dehazed, (int(dehazed.shape[1] * w_ratio), h))
        
        # Create header
        header_height = 50
        header = np.zeros((header_height, orig.shape[1] * 3, 3), dtype=np.uint8)
        
        headers = ['Original', 'Hazy', 'Dehazed']
        for j, text in enumerate(headers):
            cv2.putText(header, text,
                       (orig.shape[1] * j + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Concatenate
        comparison = np.hstack([orig, hazy, dehazed])
        comparison = np.vstack([header, comparison])
        
        # Save
        output_path = output_dir / f"{orig_path.stem}_comparison.jpg"
        cv2.imwrite(str(output_path), comparison)
        
        print(f"[{i+1}] {orig_path.name}")
    
    print(f"Comparisons saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Image Dehazing Utility Scripts"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process dataset
    process_parser = subparsers.add_parser('process', help='Process dataset')
    process_parser.add_argument('--input', required=True, help='Input directory')
    process_parser.add_argument('--output', required=True, help='Output directory')
    process_parser.add_argument('--model', default='AODNet', help='Model name')
    process_parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    # Evaluate metrics
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate metrics')
    eval_parser.add_argument('--dehazed', required=True, help='Dehazed images directory')
    eval_parser.add_argument('--ground-truth', required=True, help='Ground truth directory')
    eval_parser.add_argument('--output', default='metrics.csv', help='Output CSV file')
    
    # Generate haze
    haze_parser = subparsers.add_parser('generate-haze', help='Generate synthetic haze')
    haze_parser.add_argument('--input', required=True, help='Input directory')
    haze_parser.add_argument('--output', required=True, help='Output directory')
    haze_parser.add_argument('--intensity', type=float, default=0.5, help='Haze intensity')
    haze_parser.add_argument('--method', default='atmospheric_scattering', help='Haze method')
    
    # Create comparison
    compare_parser = subparsers.add_parser('compare', help='Create comparisons')
    compare_parser.add_argument('--original', required=True, help='Original images')
    compare_parser.add_argument('--hazy', required=True, help='Hazy images')
    compare_parser.add_argument('--dehazed', required=True, help='Dehazed images')
    compare_parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        process_dataset(args.input, args.output, args.model, args.device)
    elif args.command == 'evaluate':
        evaluate_metrics(args.dehazed, args.ground_truth, args.output)
    elif args.command == 'generate-haze':
        generate_synthetic_haze(args.input, args.output, args.intensity, args.method)
    elif args.command == 'compare':
        create_comparison_viz(args.original, args.hazy, args.dehazed, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

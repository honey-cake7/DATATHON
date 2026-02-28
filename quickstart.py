"""
Quick start scripts for running the dehazing pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_inference(input_path, output_path, model="AODNet", device="cuda"):
    """Run single image inference."""
    print(f"Running inference on {input_path}")
    print(f"Model: {model}, Device: {device}")
    
    from src.inference.dehaze import DehazeInference
    import cv2
    
    dehaze = DehazeInference(model, device=device)
    result = dehaze.infer(input_path)
    
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved to {output_path}")

def run_dashboard():
    """Launch web dashboard."""
    print("Launching Gradio dashboard...")
    print("Navigate to http://localhost:7860")
    
    subprocess.run([sys.executable, "web/app.py"], cwd=".")

def run_evaluation(dehazed_dir, gt_dir, output_csv):
    """Evaluate metrics."""
    print(f"Evaluating dehazed images")
    print(f"Ground truth: {gt_dir}")
    print(f"Dehazed: {dehazed_dir}")
    
    from src.metrics.evaluation import DehazingMetrics
    
    metrics = DehazingMetrics()
    results = metrics.evaluate_batch(gt_dir, dehazed_dir)
    metrics.print_summary(results)

def main():
    parser = argparse.ArgumentParser(
        description="Image Dehazing Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference
  python quickstart.py infer --input hazy.jpg --output dehazed.jpg
  
  # Launch dashboard
  python quickstart.py dashboard
  
  # Evaluate metrics
  python quickstart.py evaluate --dehazed results/ --gt data/ground_truth/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--input', required=True, help='Input hazy image')
    infer_parser.add_argument('--output', required=True, help='Output dehazed image')
    infer_parser.add_argument('--model', default='AODNet', help='Model name')
    infer_parser.add_argument('--device', default='cuda', help='Device')
    
    # Dashboard command
    subparsers.add_parser('dashboard', help='Launch web dashboard')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate metrics')
    eval_parser.add_argument('--dehazed', required=True, help='Dehazed images dir')
    eval_parser.add_argument('--gt', required=True, help='Ground truth images dir')
    eval_parser.add_argument('--output', default='metrics.csv', help='Output CSV')
    
    args = parser.parse_args()
    
    if args.command == 'infer':
        run_inference(args.input, args.output, args.model, args.device)
    elif args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'evaluate':
        run_evaluation(args.dehazed, args.gt, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

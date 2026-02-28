"""
Inference pipeline for image dehazing.
Handles loading models, processing images, and batch inference.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
import yaml
import time

from ..models.architectures import create_model


class DehazeInference:
    """Inference pipeline for dehazing models."""
    
    def __init__(self, 
                 model_name: str = "AODNet",
                 checkpoint_path: Optional[Union[str, Path]] = None,
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (256, 256)):
        """
        Initialize inference pipeline.
        
        Args:
            model_name: Name of model architecture
            checkpoint_path: Path to pretrained weights
            device: Device to run inference on ("cuda" or "cpu")
            input_size: Target input size (H, W)
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.input_size = input_size
        
        # Create model
        self.model = create_model(model_name)
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Initialized {model_name} on {self.device}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load pretrained weights."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def preprocess(self, image: Union[np.ndarray, str]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (numpy array) or path to image file
        
        Returns:
            Preprocessed tensor and original shape
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Cannot load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original shape
        orig_shape = image.shape[:2]
        
        # Resize to target size
        if image.shape[:2] != self.input_size:
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor, orig_shape
    
    def postprocess(self, output: torch.Tensor, 
                   target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Postprocess model output.
        
        Args:
            output: Model output tensor
            target_shape: Target shape to resize to
        
        Returns:
            Numpy array in range [0, 255]
        """
        # Remove batch dimension and convert to numpy
        output = output.squeeze(0).cpu().detach()
        output = output.permute(1, 2, 0).numpy()
        
        # Clip to valid range
        output = np.clip(output, 0, 1)
        
        # Convert to uint8
        output = (output * 255).astype(np.uint8)
        
        # Resize if needed
        if target_shape is not None:
            output = cv2.resize(output, (target_shape[1], target_shape[0]))
        
        return output
    
    @torch.no_grad()
    def infer(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        Run inference on a single image.
        
        Args:
            image: Input image or path
        
        Returns:
            Dehazed image as numpy array (H, W, 3) in range [0, 255]
        """
        # Preprocess
        tensor, orig_shape = self.preprocess(image)
        
        # Inference
        with torch.cuda.amp.autocast() if str(self.device) == 'cuda' else torch.no_grad():
            output = self.model(tensor)
        
        # Postprocess
        result = self.postprocess(output, orig_shape)
        
        return result
    
    def batch_infer(self, 
                   image_dir: Union[str, Path],
                   output_dir: Union[str, Path],
                   save_format: str = "png",
                   verbose: bool = True) -> List[Tuple[str, float]]:
        """
        Run inference on batch of images.
        
        Args:
            image_dir: Directory with input images
            output_dir: Directory to save results
            save_format: Output image format (jpg, png, etc.)
            verbose: Print progress
        
        Returns:
            List of (filename, inference_time_ms) tuples
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        results = []
        
        for i, img_path in enumerate(sorted(image_files)):
            try:
                # Inference
                start_time = time.time()
                dehazed = self.infer(img_path)
                elapsed = (time.time() - start_time) * 1000  # Convert to ms
                
                # Save result
                output_path = output_dir / f"{img_path.stem}_dehazed.{save_format}"
                dehazed_bgr = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), dehazed_bgr)
                
                results.append((img_path.name, elapsed))
                
                if verbose:
                    print(f"[{i+1}/{len(image_files)}] {img_path.name} ({elapsed:.1f}ms)")
            
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
        
        return results
    
    def print_model_info(self) -> None:
        """Print model information."""
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Input Size: {self.input_size}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                              if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"{'='*60}\n")


def create_inference_pipeline(config_path: Union[str, Path] = "config.yaml") -> DehazeInference:
    """Create inference pipeline from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('models', {})
    primary_model = model_config.get('primary', 'AODNet')
    
    inference = DehazeInference(
        model_name=primary_model,
        device=config.get('inference', {}).get('device', 'cuda')
    )
    
    return inference


if __name__ == "__main__":
    # Example usage
    inference = DehazeInference(model_name="AODNet", device="cpu")
    inference.print_model_info()
    
    # Test inference (need sample image)
    # result = inference.infer("sample_hazy.jpg")
    # cv2.imwrite("result_dehazed.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

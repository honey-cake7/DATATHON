"""
Haze generation pipeline for synthetic haze synthesis.
"""

import numpy as np
import cv2
from typing import Tuple, Union
from pathlib import Path


class HazeGenerator:
    """Generate synthetic haze on clear images."""
    
    @staticmethod
    def atmospheric_scattering(image: np.ndarray, 
                              beta: float = 0.5,
                              airlight: float = 1.0) -> np.ndarray:
        """
        Generate haze using atmospheric scattering model.
        
        Formula: I(x) = J(x)*exp(-beta*d) + A*(1-exp(-beta*d))
        where:
            I(x) = observed hazy image
            J(x) = haze-free image
            A = airlight (typically [1, 1, 1] for white haze)
            beta = scattering coefficient
            d = depth (estimated from dark channel)
        
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            beta: Scattering coefficient (0.1-1.0), higher = more haze
            airlight: Airlight intensity (0.5-1.0)
        
        Returns:
            Hazy image with same shape as input
        """
        h, w, c = image.shape
        
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Estimate depth map from image (use dark channel prior concept)
        # Simplified: use 1 - normalized image as depth
        max_vals = np.maximum(np.max(image, axis=2), 1e-8)
        min_vals = np.min(image, axis=2)
        depth = 1.0 - (min_vals / max_vals)
        
        # Ensure depth is 2D (h, w)
        depth = depth.reshape(h, w)
        
        # Normalize depth to scale the effect
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Apply atmospheric scattering - expand depth to 3 channels
        transmission = np.exp(-beta * depth)
        transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
        
        # Generate airlight (white with some color variation)
        A = np.array([airlight, airlight, airlight], dtype=np.float32)
        
        # Haze formula
        hazy = image * transmission + A * (1 - transmission)
        
        # Clip and convert back to uint8
        hazy = np.clip(hazy * 255, 0, 255).astype(np.uint8)
        
        return hazy
    
    @staticmethod
    def gaussian_blur_color_shift(image: np.ndarray,
                                 intensity: float = 0.5) -> np.ndarray:
        """
        Generate haze using Gaussian blur and color shift.
        Simplified approach: blur + color overlay.
        
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            intensity: Haze intensity (0.1-1.0)
        
        Returns:
            Hazy image
        """
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Apply Gaussian blur
        kernel_size = int(10 + intensity * 40)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Color shift (whitish haze)
        haze_color = np.array([200, 190, 180], dtype=np.float32)
        image_float = image.astype(np.float32)
        blurred_float = blurred.astype(np.float32)
        
        # Blend: more intensity = more haze overlay
        hazy = image_float * (1 - intensity * 0.7) + blurred_float * intensity * 0.7
        hazy = hazy * (1 - intensity * 0.3) + haze_color * intensity * 0.3
        
        hazy = np.clip(hazy, 0, 255).astype(np.uint8)
        
        return hazy
    
    @staticmethod
    def depth_based_progressive(image: np.ndarray,
                               intensity: float = 0.5) -> np.ndarray:
        """
        Generate depth-based progressive haze.
        Applies more haze to bottom of image (simulating depth).
        
        Args:
            image: Input image (H, W, 3) in range [0, 255]
            intensity: Haze intensity (0.1-1.0)
        
        Returns:
            Hazy image with progressive haze
        """
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        h, w = image.shape[:2]
        
        # Create depth map (more haze at bottom)
        depth_map = np.linspace(0, intensity, h)[:, np.newaxis]
        depth_map = np.tile(depth_map, (1, w))
        
        # Apply haze progressively
        image_float = image.astype(np.float32)
        hazy = image_float.copy()
        
        for i in range(h):
            haze_strength = depth_map[i, 0]
            hazy[i] = image_float[i] * (1 - haze_strength) + 200 * haze_strength
        
        hazy = np.clip(hazy, 0, 255).astype(np.uint8)
        
        return hazy
    
    @staticmethod
    def batch_generate(input_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      beta_values: list = None,
                      method: str = "atmospheric_scattering") -> list:
        """
        Generate hazy images from a directory.
        
        Args:
            input_dir: Directory with clear images
            output_dir: Directory to save hazy images
            beta_values: List of beta values to try (for atmospheric method)
            method: Haze generation method
        
        Returns:
            List of paths to generated hazy images
        """
        if beta_values is None:
            beta_values = [0.3, 0.5, 0.8]
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        generated = []
        
        for img_path in input_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for beta in beta_values:
                if method == "atmospheric_scattering":
                    hazy = HazeGenerator.atmospheric_scattering(image, beta=beta)
                elif method == "gaussian_blur":
                    hazy = HazeGenerator.gaussian_blur_color_shift(image, intensity=beta)
                elif method == "depth_progressive":
                    hazy = HazeGenerator.depth_based_progressive(image, intensity=beta)
                else:
                    continue
                
                # Save with suffix indicating haze level
                stem = img_path.stem
                hazy_path = output_dir / f"{stem}_haze_{beta:.1f}.png"
                cv2.imwrite(str(hazy_path), cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))
                generated.append(hazy_path)
        
        return generated


if __name__ == "__main__":
    # Example usage
    generator = HazeGenerator()
    
    # Generate haze on a single image
    # img = cv2.imread("sample.jpg")
    # hazy = generator.atmospheric_scattering(img, beta=0.5)
    # cv2.imwrite("hazy_sample.jpg", hazy)

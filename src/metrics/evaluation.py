"""
Evaluation metrics for image dehazing assessment.
Calculates PSNR, SSIM, and other quality metrics.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from typing import Union, Tuple
from pathlib import Path


class DehazingMetrics:
    """Calculate standard metrics for dehazing evaluation."""
    
    def __init__(self, data_range: int = 255):
        """
        Initialize metrics calculator.
        
        Args:
            data_range: Maximum pixel value (typically 255 for uint8)
        """
        self.data_range = data_range
    
    @staticmethod
    def load_image(path: Union[str, Path], as_float: bool = False) -> np.ndarray:
        """
        Load image from disk.
        
        Args:
            path: Path to image file
            as_float: If True, return normalized [0, 1], else [0, 255]
        
        Returns:
            Image as numpy array (H, W, C)
        """
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if as_float:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        
        return img
    
    def calculate_psnr(self, 
                      reference: np.ndarray, 
                      distorted: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        PSNR = 20 * log10(MAX_VALUE / sqrt(MSE))
        
        Args:
            reference: Reference (ground truth) image
            distorted: Distorted (dehazed) image
        
        Returns:
            PSNR value in dB (higher is better, typically 20-40 dB)
        """
        if reference.shape != distorted.shape:
            raise ValueError(f"Image shapes mismatch: {reference.shape} vs {distorted.shape}")
        
        mse = np.mean((reference - distorted) ** 2)
        
        if mse < 1e-10:  # Identical images
            return 100.0
        
        psnr = 20 * np.log10(self.data_range / np.sqrt(mse))
        return float(psnr)
    
    def calculate_ssim(self, 
                      reference: np.ndarray, 
                      distorted: np.ndarray,
                      multichannel: bool = True) -> float:
        """
        Calculate Structural Similarity Index Measure.
        
        SSIM measures perceived quality preservation.
        
        Args:
            reference: Reference (ground truth) image
            distorted: Distorted (dehazed) image
            multichannel: If True, calculate SSIM for each channel separately
        
        Returns:
            SSIM value in range [-1, 1] (1 = identical, higher is better)
        """
        if reference.shape != distorted.shape:
            raise ValueError(f"Image shapes mismatch: {reference.shape} vs {distorted.shape}")
        
        # Handle both grayscale and color images
        if len(reference.shape) == 3:
            channel_ssims = []
            for c in range(reference.shape[2]):
                ch_ssim = ssim_skimage(
                    reference[:, :, c],
                    distorted[:, :, c],
                    data_range=self.data_range,
                    win_size=11  # Standard window size
                )
                channel_ssims.append(ch_ssim)
            ssim_val = np.mean(channel_ssims)
        else:
            ssim_val = ssim_skimage(
                reference,
                distorted,
                data_range=self.data_range,
                win_size=11
            )
        
        return float(ssim_val)
    
    def calculate_mse(self, 
                     reference: np.ndarray, 
                     distorted: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return float(np.mean((reference - distorted) ** 2))
    
    def calculate_mae(self, 
                     reference: np.ndarray, 
                     distorted: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(reference - distorted)))
    
    def evaluate_batch(self, 
                      gt_dir: Union[str, Path], 
                      pred_dir: Union[str, Path]) -> dict:
        """
        Evaluate all images in directories.
        
        Args:
            gt_dir: Directory with ground truth images
            pred_dir: Directory with predicted/dehazed images
        
        Returns:
            Dictionary with per-image and average metrics
        """
        gt_dir = Path(gt_dir)
        pred_dir = Path(pred_dir)
        
        results = {
            'psnr': [],
            'ssim': [],
            'mse': [],
            'mae': [],
            'images': []
        }
        
        # Find matching image pairs
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        gt_images = sorted([f for f in gt_dir.iterdir() 
                           if f.suffix.lower() in image_extensions])
        
        for gt_path in gt_images:
            pred_path = pred_dir / gt_path.name
            if not pred_path.exists():
                print(f"Warning: No prediction for {gt_path.name}")
                continue
            
            try:
                gt_img = self.load_image(gt_path, as_float=False)
                pred_img = self.load_image(pred_path, as_float=False)
                
                psnr = self.calculate_psnr(gt_img, pred_img)
                ssim = self.calculate_ssim(gt_img, pred_img)
                mse = self.calculate_mse(gt_img, pred_img)
                mae = self.calculate_mae(gt_img, pred_img)
                
                results['psnr'].append(psnr)
                results['ssim'].append(ssim)
                results['mse'].append(mse)
                results['mae'].append(mae)
                results['images'].append(gt_path.name)
                
                print(f"{gt_path.name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
            
            except Exception as e:
                print(f"Error processing {gt_path.name}: {e}")
        
        # Calculate averages
        if results['psnr']:
            results['avg_psnr'] = np.mean(results['psnr'])
            results['avg_ssim'] = np.mean(results['ssim'])
            results['avg_mse'] = np.mean(results['mse'])
            results['avg_mae'] = np.mean(results['mae'])
            results['num_images'] = len(results['psnr'])
        
        return results
    
    def print_summary(self, results: dict) -> None:
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("DEHAZING EVALUATION SUMMARY")
        print("="*60)
        
        if 'avg_psnr' in results:
            print(f"Images Evaluated: {results['num_images']}")
            print(f"Average PSNR: {results['avg_psnr']:.2f} dB")
            print(f"Average SSIM: {results['avg_ssim']:.4f}")
            print(f"Average MSE: {results['avg_mse']:.4f}")
            print(f"Average MAE: {results['avg_mae']:.4f}")
            print(f"\nPSNR Range: {min(results['psnr']):.2f} - {max(results['psnr']):.2f} dB")
            print(f"SSIM Range: {min(results['ssim']):.4f} - {max(results['ssim']):.4f}")
        else:
            print("No valid results to display")
        
        print("="*60)


if __name__ == "__main__":
    # Example usage
    metrics = DehazingMetrics()
    
    # Evaluate a batch
    # results = metrics.evaluate_batch("data/ground_truth", "results/dehazed_images")
    # metrics.print_summary(results)

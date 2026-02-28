"""
Web Dashboard for Image Dehazing using Gradio.
Provides interactive interface for real-time dehazing.
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
import sys
import torch
from typing import Tuple, Optional

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.dehaze import DehazeInference
from src.haze_generation.generator import HazeGenerator
from src.metrics.evaluation import DehazingMetrics


class DehazeDashboard:
    """Interactive dehazing dashboard."""
    
    def __init__(self, 
                 model_name: str = "AODNet",
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda"):
        """Initialize dashboard with model."""
        self.model_name = model_name
        self.device = device
        
        try:
            self.dehaze = DehazeInference(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                device=device
            )
            print(f"✓ Loaded {model_name} model")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.dehaze = None
        
        self.haze_gen = HazeGenerator()
        self.metrics = DehazingMetrics()
    
    def dehaze_image(self, image: np.ndarray) -> np.ndarray:
        """Dehaze a single image."""
        if self.dehaze is None:
            raise ValueError("Model not loaded")
        
        if image is None:
            raise ValueError("No image provided")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Run inference
        result = self.dehaze.infer(image)
        
        return result
    
    def add_haze(self, 
                image: np.ndarray,
                haze_intensity: float = 0.5,
                method: str = "atmospheric_scattering") -> Tuple[np.ndarray, np.ndarray]:
        """Add synthetic haze to image."""
        if image is None:
            raise ValueError("No image provided")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Add haze
        if method == "atmospheric_scattering":
            hazy = self.haze_gen.atmospheric_scattering(image, beta=haze_intensity)
        elif method == "gaussian_blur":
            hazy = self.haze_gen.gaussian_blur_color_shift(image, intensity=haze_intensity)
        else:
            hazy = self.haze_gen.depth_based_progressive(image, intensity=haze_intensity)
        
        return image, hazy
    
    def dehaze_hazy(self,
                   image: np.ndarray,
                   haze_intensity: float = 0.5,
                   haze_method: str = "atmospheric_scattering") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Add haze and then dehaze an image."""
        # Add haze
        clear_img, hazy_img = self.add_haze(image, haze_intensity, haze_method)
        
        # Dehaze
        dehazed_img = self.dehaze_image(hazy_img)
        
        return clear_img, hazy_img, dehazed_img
    
    def compare_dehazed(self,
                       original: np.ndarray,
                       hazy: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison."""
        if original is None or hazy is None:
            raise ValueError("Both images required")
        
        # Dehaze the hazy image
        dehazed = self.dehaze_image(hazy)
        
        # Resize to same dimensions
        h = min(original.shape[0], hazy.shape[0], dehazed.shape[0])
        w = min(original.shape[1], hazy.shape[1], dehazed.shape[1])
        
        original = cv2.resize(original, (w, h))
        hazy = cv2.resize(hazy, (w, h))
        dehazed = cv2.resize(dehazed, (w, h))
        
        # Create comparison image
        comparison = np.hstack([original, hazy, dehazed])
        
        return original, hazy, dehazed, comparison


def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dashboard = DehazeDashboard(
        model_name="AODNet",
        device=device
    )
    
    css = """
    .header {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .comparison {
        display: flex;
        gap: 10px;
    }
    """
    
    with gr.Blocks(title="Image Dehazing Dashboard") as demo:
        gr.Markdown(
            "# 🌫️ Image Dehazing Dashboard\n"
            "Remove haze and fog from images using deep learning"
        )
        
        with gr.Tabs():
            # Tab 1: Simple Dehazing
            with gr.TabItem("Dehaze Image"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input Hazy Image")
                        input_image = gr.Image(label="Upload hazy image", type="numpy")
                    
                    with gr.Column():
                        gr.Markdown("### Dehazed Result")
                        output_image = gr.Image(label="Dehazed image")
                        dehaze_btn = gr.Button("🎯 Dehaze", variant="primary", scale=1)
                
                dehaze_btn.click(
                    fn=dashboard.dehaze_image,
                    inputs=[input_image],
                    outputs=[output_image]
                )
            
            # Tab 2: Haze Generation
            with gr.TabItem("Generate Haze"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Original Clear Image")
                        clear_image = gr.Image(label="Upload clear image", type="numpy")
                        haze_method = gr.Dropdown(
                            choices=[
                                "atmospheric_scattering",
                                "gaussian_blur",
                                "depth_progressive"
                            ],
                            value="atmospheric_scattering",
                            label="Haze Method"
                        )
                        haze_intensity = gr.Slider(0.1, 1.0, 0.5, label="Haze Intensity")
                    
                    with gr.Column():
                        gr.Markdown("### Generated Hazy Image")
                        hazy_image = gr.Image(label="Hazy image")
                        gen_haze_btn = gr.Button("⚙️ Add Haze", variant="primary")
                
                gen_haze_btn.click(
                    fn=dashboard.add_haze,
                    inputs=[clear_image, haze_intensity, haze_method],
                    outputs=[clear_image, hazy_image]
                )
            
            # Tab 3: Haze Generation + Dehazing
            with gr.TabItem("Haze & Dehaze"):
                with gr.Row():
                    gr.Markdown("### Full Pipeline: Original → Haze → Dehazed")
                
                with gr.Row():
                    with gr.Column():
                        clear_img_input = gr.Image(label="Upload clear image", type="numpy")
                        haze_method_full = gr.Dropdown(
                            choices=[
                                "atmospheric_scattering",
                                "gaussian_blur",
                                "depth_progressive"
                            ],
                            value="atmospheric_scattering",
                            label="Haze Method"
                        )
                        haze_intensity_full = gr.Slider(0.1, 1.0, 0.5, label="Haze Intensity")
                        process_btn = gr.Button("🔄 Process", variant="primary")
                    
                    with gr.Column():
                        clear_img_output = gr.Image(label="Original")
                        hazy_img_output = gr.Image(label="With Haze")
                        dehazed_img_output = gr.Image(label="Dehazed")
                
                process_btn.click(
                    fn=dashboard.dehaze_hazy,
                    inputs=[clear_img_input, haze_intensity_full, haze_method_full],
                    outputs=[clear_img_output, hazy_img_output, dehazed_img_output]
                )
            
            # Tab 4: Side-by-Side Comparison
            with gr.TabItem("Compare"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Original")
                        original_img = gr.Image(label="Original image", type="numpy")
                        gr.Markdown("### Hazy")
                        hazy_img = gr.Image(label="Hazy image", type="numpy")
                    
                    with gr.Column():
                        compare_output = gr.Image(label="Comparison (Original | Hazy | Dehazed)")
                
                compare_btn = gr.Button("📊 Compare", variant="primary")
                
                compare_btn.click(
                    fn=dashboard.compare_dehazed,
                    inputs=[original_img, hazy_img],
                    outputs=[original_img, hazy_img, dehazed_img_output, compare_output]
                )
        
        # Info section
        gr.Markdown(
            f"""
            ## ℹ️ System Information
            - **Model**: AODNet (All-in-One Dehazing Network)
            - **Device**: {device.upper()}
            - **Input Size**: 256×256 (auto-resized)
            
            ## 📖 Usage Guide
            1. **Dehaze**: Upload a hazy image to remove fog/haze
            2. **Generate Haze**: Add synthetic haze to a clear image
            3. **Full Pipeline**: Combine haze generation + dehazing
            4. **Compare**: Side-by-side comparison of results
            """
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True
    )

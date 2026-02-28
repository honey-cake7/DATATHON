"""
Inference pipeline module for image dehazing.
"""

from .dehaze import DehazeInference, create_inference_pipeline

__all__ = ["DehazeInference", "create_inference_pipeline"]

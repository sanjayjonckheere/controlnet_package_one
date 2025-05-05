"""Utility functions for ControlNet."""

import os
import random
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_depth_maps(image_paths: list[Path], output_dir: Path, device: str = "cuda") -> None:
    """
    Generate depth maps for input images using MiDaS and save as grayscale PNGs.
    """
    logger.info("Loading MiDaS model")
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    
    for img_path in image_paths:
        logger.info(f"Processing {img_path.name}")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32)
        logger.info(f"Converted {img_path.name} to NumPy array with shape {img_np.shape}")
        img_tensor = transform(img_np)
        logger.info(f"Transformed image to tensor with shape {img_tensor.shape}")
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            depth = midas(img_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        depth_np = (255 * depth_normalized.cpu().numpy()).astype(np.uint8)
        depth_img = Image.fromarray(depth_np, mode="L")
        
        output_name = img_path.stem.replace("image_", "depth_") + ".png"
        output_path = output_dir / output_name
        depth_img.save(output_path)
        logger.info(f"Saved depth map: {output_path}")


def run_inference(
    pipeline,
    image_path: str,
    prompt: str,
    output_path: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30
) -> None:
    """
    Run inference with a trained ControlNet model.
    """
    # Load input image
    image = Image.open(image_path).convert("RGB")
    
    # Generate the image
    output_image = pipeline(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    # Save the generated image
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    output_image.save(output_path)
    logger.info(f"Generated image saved to: {output_path}")

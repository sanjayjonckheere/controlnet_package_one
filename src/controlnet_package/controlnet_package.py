"""Main module for the ControlNet package."""

import os
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from controlnet_package.config import load_config
from controlnet_package.dataset import DepthDataset
from controlnet_package.model import pipeline_factory
from controlnet_package.training import train_controlnet, clear_memory
from controlnet_package.utils import set_seed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up environment variables for optimized training."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


def run_training(config_path: str) -> None:
    """
    Main entry point for ControlNet fine-tuning with memory optimization.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        logger.info("Initializing ControlNet fine-tuning")
        setup_environment()
        clear_memory()
        
        # Load configuration
        config = load_config(Path(config_path))
        
        # Set device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("GPU not available, using CPU")
            
        # Update config with defaults
        config.update({
            "batch_size": config.get("batch_size", 1),
            "image_size": config.get("image_size", 128),
            "device": device
        })
            
        logger.info(f"Configuration: {config}")
        
        # Set random seed
        set_seed(config.get("seed", 42))
        
        # Create dataset
        dataset = DepthDataset(config["train_data_root"], config["image_size"])
        if len(dataset) == 0:
            raise ValueError("Empty dataset detected")
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 0),
            pin_memory=(device == "cuda")
        )
        
        # Create pipeline
        logger.info("Creating pipeline - this may take a moment...")
        Path(config["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
        pipeline = pipeline_factory(config["pretrained_model"], config["controlnet_model"], "cpu")
        
        # Apply memory optimizations
        if device == "cuda":
            logger.info(f"GPU memory after loading pipeline to CPU: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB reserved")
        
        pipeline.to(torch.float32)
        logger.info("Applying memory optimizations...")
        pipeline.enable_attention_slicing(slice_size=1)
        pipeline.unet.enable_gradient_checkpointing()
        pipeline.controlnet.enable_gradient_checkpointing()
        
        # Start training
        logger.info("Starting training...")
        train_controlnet(pipeline, loader, config, config["ckpt_dir"])
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Cleaning up resources")
        clear_memory()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python -m controlnet_package.controlnet_package <config_path>")
        sys.exit(1)
    run_training(sys.argv[1])

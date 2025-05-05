"""Command-line interface for the ControlNet package."""

import argparse
import sys
import os
import logging
from pathlib import Path
import torch

from controlnet_package.config import load_config
from controlnet_package.dataset import DepthDataset
from controlnet_package.model import pipeline_factory
from controlnet_package.training import train_controlnet, clear_memory
from controlnet_package.utils import generate_depth_maps, run_inference, set_seed


def setup_logging(log_file=None):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    return logging.getLogger(__name__)


def train_command(args):
    """Handle the train subcommand."""
    logger = setup_logging(args.log_file)
    logger.info("Starting ControlNet training")
    
    try:
        # Load configuration
        config_path = Path(args.config)
        config = load_config(config_path)
        
        # Set device
        if torch.cuda.is_available() and not args.cpu:
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for training")
        
        # Update config with CLI arguments
        config.update({
            "device": device,
            "batch_size": args.batch_size or config.get("batch_size", 1),
            "image_size": args.image_size or config.get("image_size", 256),
            "epochs": args.epochs or config.get("epochs", 20),
            "lr": args.learning_rate or config.get("lr", 1e-5)
        })
        
        logger.info(f"Configuration: {config}")
        
        # Load dataset
        dataset = DepthDataset(
            data_root=config["train_data_root"],
            image_size=config["image_size"]
        )
        
        if len(dataset) == 0:
            raise ValueError("Empty dataset detected")
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Create dataloader
        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=args.num_workers or config.get("num_workers", 0),
            pin_memory=(device == "cuda")
        )
        
        # Create pipeline
        logger.info("Creating pipeline - this may take a moment...")
        pipeline = pipeline_factory(
            config["pretrained_model"], 
            config["controlnet_model"], 
            "cpu"  # Initially load to CPU
        )
        
        # Start training
        logger.info("Starting training...")
        train_controlnet(pipeline, loader, config, config["ckpt_dir"])
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def depth_command(args):
    """Handle the depth subcommand."""
    logger = setup_logging(args.log_file)
    logger.info("Starting depth map generation")
    
    try:
        # Set device
        if torch.cuda.is_available() and not args.cpu:
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for depth estimation")
        
        # Find image files
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
        if not image_files:
            logger.error(f"No image files found in {input_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(image_files)} images")
        
        # Generate depth maps
        generate_depth_maps(image_files, output_dir, device)
        logger.info(f"Depth maps generated and saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during depth map generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def inference_command(args):
    """Handle the inference subcommand."""
    logger = setup_logging(args.log_file)
    logger.info("Starting inference")
    
    try:
        # Set device
        if torch.cuda.is_available() and not args.cpu:
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        
        # Load model
        logger.info(f"Loading model from {args.model}")
        pipeline = pipeline_factory(
            args.pretrained_model or "runwayml/stable-diffusion-v1-5",
            args.model,
            device
        )
        
        # Run inference
        logger.info(f"Running inference on {args.image}")
        run_inference(
            pipeline,
            args.image,
            args.prompt,
            args.output,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps
        )
        logger.info(f"Inference completed, output saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """Main entry point for the ControlNet CLI."""
    parser = argparse.ArgumentParser(description="ControlNet depth model fine-tuning and inference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--log-file", type=str, help="Path to log file")
    parent_parser.add_argument("--cpu", action="store_true", help="Force CPU use even if GPU is available")
    
    # Train command
    train_parser = subparsers.add_parser("train", parents=[parent_parser], help="Train a ControlNet model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config file")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--image-size", type=int, help="Image size")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--num-workers", type=int, help="Number of data loader workers")
    train_parser.set_defaults(func=train_command)
    
    # Depth command
    depth_parser = subparsers.add_parser("depth", parents=[parent_parser], help="Generate depth maps")
    depth_parser.add_argument("--input-dir", type=str, required=True, help="Input directory with images")
    depth_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for depth maps")
    depth_parser.set_defaults(func=depth_command)
    
    # Inference command
    infer_parser = subparsers.add_parser("inference", parents=[parent_parser], help="Run inference")
    infer_parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--pretrained-model", type=str, help="Pretrained model name or path")
    infer_parser.add_argument("--image", type=str, required=True, help="Input image path")
    infer_parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    infer_parser.add_argument("--output", type=str, required=True, help="Output image path")
    infer_parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    infer_parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    infer_parser.set_defaults(func=inference_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set environment variables for memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    
    # Call the appropriate command handler
    args.func(args)


if __name__ == "__main__":
    main()

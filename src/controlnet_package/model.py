"""Model implementation for ControlNet."""

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from typing import Any


def pipeline_factory(pretrained_model: str, controlnet_model: str, device: str) -> StableDiffusionControlNetPipeline:
    """
    Create and configure the StableDiffusionControlNetPipeline.
    """
    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float32)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        controlnet=controlnet,
        torch_dtype=torch.float32
    ).to(device)
    return pipeline

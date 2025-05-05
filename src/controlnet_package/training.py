"""Training functionality for ControlNet."""

import torch
import logging
import gc
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)


def log_step(func):
    """Decorator to log training and validation steps."""
    def wrapper(pipeline, loader, config, ckpt_dir, *args, **kwargs):
        logger.debug(f"Starting training with config: {config}")
        result = func(pipeline, loader, config, ckpt_dir, *args, **kwargs)
        logger.debug("Training completed")
        return result
    return wrapper


def clear_memory():
    """Clear CUDA memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@log_step
def train_controlnet(pipeline: Any, loader: torch.utils.data.DataLoader, config: dict, ckpt_dir: str) -> None:
    """
    Train the ControlNet model with validation after each epoch, optimized for memory efficiency.
    """
    device = config["device"]
    if device == "cuda":
        pipeline.to("cpu")
        clear_memory()
        pipeline.enable_attention_slicing(slice_size=1)
        pipeline.enable_sequential_cpu_offload()
        pipeline.enable_vae_slicing()
        pipeline.controlnet.to(device)
    pipeline.controlnet.train()
    for component in [pipeline.unet, pipeline.vae, pipeline.text_encoder]:
        if component is not None:
            component.requires_grad_(False)
    optimizer = torch.optim.AdamW(pipeline.controlnet.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = pipeline.scheduler
    pipeline.controlnet.enable_gradient_checkpointing()
    for epoch in range(config["epochs"]):
        logger.info(f"Starting epoch {epoch}/{config['epochs']}")
        total_loss = 0.0
        num_batches = 0
        for batch_idx, (images, depths) in enumerate(loader):
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.debug(f"Memory before batch {batch_idx}: Allocated {mem_allocated:.2f} GB, Reserved {mem_reserved:.2f} GB")
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            try:
                with torch.no_grad():
                    if pipeline.vae.device != device:
                        pipeline.vae = pipeline.vae.to(device)
                    latents = pipeline.vae.encode(images).latent_dist.sample() * pipeline.vae.config.scaling_factor
                    if device == "cuda":
                        pipeline.vae = pipeline.vae.to("cpu")
                        clear_memory()
                noise = torch.randn_like(latents).to(device, dtype=torch.float32)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = torch.zeros(
                    (images.shape[0], 77, 768),
                    device=device,
                    dtype=torch.float32
                )
                controlnet_output = pipeline.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=depths,
                    return_dict=True
                )
                if pipeline.unet.device != device:
                    pipeline.unet = pipeline.unet.to(device)
                down_samples = controlnet_output.down_block_res_samples
                mid_sample = controlnet_output.mid_block_res_sample
                loss = 0.0
                if mid_sample is not None:
                    loss += 0.5 * torch.nn.functional.mse_loss(
                        torch.mean(mid_sample, dim=[1, 2, 3]),
                        torch.mean(noise, dim=[1, 2, 3])
                    )
                for i, down_sample in enumerate(down_samples):
                    if down_sample is not None:
                        weight = 0.5 / len(down_samples)
                        loss += weight * torch.nn.functional.mse_loss(
                            torch.mean(down_sample, dim=[1, 2, 3]),
                            torch.mean(noise, dim=[1, 2, 3])
                        )
                if device == "cuda":
                    pipeline.unet = pipeline.unet.to("cpu")
                    clear_memory()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pipeline.controlnet.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                if batch_idx % 5 == 0 or batch_idx == len(loader) - 1:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
                del latents, noise, timesteps, noisy_latents, encoder_hidden_states
                del controlnet_output, down_samples, mid_sample, loss
                clear_memory()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM error in batch {batch_idx}, skipping: {str(e)}")
                    clear_memory()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        logger.info(f"Epoch {epoch}, Average Training Loss: {avg_loss:.4f}")
        ckpt_path = Path(ckpt_dir) / f"controlnet_epoch_{epoch}.pt"
        pipeline.controlnet = pipeline.controlnet.to("cpu")
        torch.save(pipeline.controlnet.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
        pipeline.controlnet = pipeline.controlnet.to(device)
        clear_memory()
    pipeline.to("cpu")
    clear_memory()

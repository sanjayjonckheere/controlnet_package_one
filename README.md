# ControlNet Package

HEAD
This project provides a **ControlNet pipeline** for depth-guided image generation, structured as a Python package and optimized for local development using **PyCharm**. It fine-tunes a pre-trained **ControlNet** model (`lllyasviel/sd-controlnet-depth`) with Stable Diffusion to generate images conditioned on depth maps, automatically created from RGB images using **MiDaS** (`DPT_Large`).

This project provides a **ControlNet pipeline** for depth-guided image generation, structured as a Python package and optimized for local development using **PyCharm**. It fine-tunes a pre-trained **ControlNet** model (`lllyasviel/sd-controlnet-depth`) with Stable Diffusion to generate images conditioned on depth maps, automatically created from RGB images using **MiDaS** (`DPT_Large`). 8245f433269d5b781b3981d31209406a0249518c


## Pipeline Overview

### Description
The pipeline fine-tunes ControlNet for depth-guided image generation using two RGB images. It is optimized for local GPUs (e.g., NVIDIA) and structured as a Python package for modularity.

### Components
- **`controlnet_package/`**: Python package:
  - **`config.py`**: Loads and validates `config.json`.
  - **`dataset.py`**: Defines `DepthDataset` for paired RGB and depth images.
  - **`model.py`**: Configures the `StableDiffusionControlNetPipeline`.
  - **`training.py`**: Executes the training loop with memory optimizations.
  - **`controlnet_package.py`**: Main module for the package.
  - **`cli.py`**: Command-line interface for the package.
  - **`utils.py`**: Generates depth maps with MiDaS.
- **`config.json`**: Specifies model paths and hyperparameters.
- **`pyproject.toml`**: Poetry configuration for dependencies.
- **`tests/`**: Contains unit tests (e.g., `test_dataset.py`).
- **`.gitignore`**: Excludes sensitive files.
- **`LICENSE`**: MIT License.

### Functionality
1. **Setup**: Creates a virtual environment and installs dependencies with Poetry.
2. **Data Preparation**: Uses CLI to generate depth maps from RGB images.
3. **Training**: Fine-tunes ControlNet over 20 epochs (~15–25 minutes on a GPU) at 128x128 resolution.
4. **Outputs**: Saves checkpoints (`checkpoints/controlnet_epoch_*.pt`) and logs (`training.log`).

## Role of ControlNet (`lllyasviel/sd-controlnet-depth`)

* **Purpose**: Enhances Stable Diffusion with depth-based conditioning.
* **Implementation**:
  ```python
  controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32)
  ```
* **Benefit**: Enables precise depth-guided image generation.

## Role of MiDaS (`DPT_Large`)

* **Purpose**: Generates depth maps from RGB images.
* **Implementation**:
  ```python
  midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
  ```
* **Benefit**: Automates depth map creation.

## Setup Instructions

### Prerequisites
* **PyCharm**: Professional or Community Edition.
* **Python**: 3.8 or higher.
* **Poetry**: For dependency management.
* **Git**: Configured with GitHub credentials.
* **GPU**: NVIDIA GPU with CUDA (optional but recommended).
* **Images**: Sample RGB images for training.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sanjayjonckheere/controlnet_package.git
   cd controlnet_package
   ```

2. **Set Up Poetry Environment**:
   ```bash
   poetry install
   ```
   Or use PyCharm's Poetry integration.

3. **Prepare Data**:
   * Place your RGB images in an input directory.
   * Generate depth maps:
     ```bash
     poetry run controlnet depth --input-dir ./images --output-dir ./data/train
     ```

4. **Run Training**:
   ```bash
   poetry run controlnet train --config config.json
   ```

5. **Run Tests**:
   ```bash
   poetry run pytest
   ```

6. **Run Inference**:
   ```bash
   poetry run controlnet inference --model ./checkpoints/controlnet_epoch_19.pt --image test.jpg --prompt "A photo" --output result.png
   ```

7. **Push to GitHub**:
   * Use PyCharm's `VCS > Commit` and `VCS > Git > Push`.
   * Or:
     ```bash
     git add .
     git commit -m "Update project"
     git push origin main
     ```

## Requirements

See `pyproject.toml` for dependencies:
* `diffusers==0.29.2`
* `transformers==4.38.2`
* `torch==2.2.1`
* `torchvision==0.17.1`
* `accelerate==0.28.0`
* `timm==0.9.16`
* `tqdm==4.66.2`
* `pytest==8.1.1`

## Licenses

* **ControlNet**: [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
* **MiDaS**: [Apache License 2.0](https://github.com/isl-org/MiDaS/blob/master/LICENSE).
* **Project**: MIT License.

## Acknowledgments

* **ControlNet**: lllyasviel and Hugging Face.
* **MiDaS**: Intel-ISL.
* **Stable Diffusion**: RunwayML.
* **Diffusers**: Hugging Face.

"""Tests for dataset module."""

import pytest
import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
from controlnet_package.dataset import DepthDataset


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory with test images and depth maps."""
    # Create test data directory
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create a test RGB image
    rgb_img = Image.new('RGB', (64, 64), color=(73, 109, 137))
    rgb_path = data_dir / "image_001.jpg"
    rgb_img.save(rgb_path)
    
    # Create a test depth map
    depth_img = Image.new('L', (64, 64), color=128)
    depth_path = data_dir / "depth_001.png"
    depth_img.save(depth_path)
    
    yield data_dir
    
    # Clean up
    shutil.rmtree(data_dir)


def test_dataset_initialization(temp_data_dir):
    """Test dataset initialization with correct paths."""
    dataset = DepthDataset(str(temp_data_dir), 64)
    
    assert dataset.data_root == str(temp_data_dir)
    assert dataset.image_size == 64
    assert len(dataset.image_files) == 1
    assert len(dataset.depth_files) == 1
    assert dataset.image_files[0] == "image_001.jpg"
    assert dataset.depth_files[0] == "depth_001.png"


def test_dataset_len(temp_data_dir):
    """Test __len__ method returns correct number of samples."""
    dataset = DepthDataset(str(temp_data_dir), 64)
    assert len(dataset) == 1


def test_dataset_getitem(temp_data_dir):
    """Test __getitem__ method returns correctly processed tensors."""
    dataset = DepthDataset(str(temp_data_dir), 64)
    
    # Set fixed random seed to make the test deterministic
    torch.manual_seed(42)
    image, depth = dataset[0]
    
    # Check types and shapes
    assert isinstance(image, torch.Tensor)
    assert isinstance(depth, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert depth.shape == (3, 64, 64)  # Depth maps get repeated to 3 channels
    
    # Check normalization (-1 to 1 range)
    assert -1.0 <= image.min() <= 1.0
    assert -1.0 <= image.max() <= 1.0
    assert -1.0 <= depth.min() <= 1.0
    assert -1.0 <= depth.max() <= 1.0


def test_dataset_with_missing_files(temp_data_dir):
    """Test dataset behavior with missing depth files."""
    # Create an extra RGB image without a corresponding depth map
    rgb_img = Image.new('RGB', (64, 64), color=(100, 150, 200))
    rgb_path = temp_data_dir / "image_002.jpg"
    rgb_img.save(rgb_path)
    
    # Dataset should raise an error due to mismatch
    with pytest.raises(ValueError, match="Mismatch in number of images"):
        DepthDataset(str(temp_data_dir), 64)


def test_dataset_empty_directory(tmp_path):
    """Test dataset with an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    # Dataset should initialize but have length 0
    dataset = DepthDataset(str(empty_dir), 64)
    assert len(dataset) == 0
    assert dataset.image_files == []
    assert dataset.depth_files == []


def test_dataset_transformations(temp_data_dir):
    """Test that transformations are correctly applied."""
    dataset = DepthDataset(str(temp_data_dir), 128)  # Different size than original
    
    image, depth = dataset[0]
    
    # Check resizing worked
    assert image.shape == (3, 128, 128)
    assert depth.shape == (3, 128, 128)

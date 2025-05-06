"""Tests for controlnet_package module."""

import pytest
import os
import torch  # noqa
from pathlib import Path  # noqa
from unittest.mock import patch, MagicMock

from controlnet_package import __version__
from controlnet_package.controlnet_package import run_training, setup_environment


def test_version():
    """Test version is a string."""
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_setup_environment():
    """Test environment variables are set properly."""
    setup_environment()
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
    assert os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1"


@pytest.mark.skip(reason="Requires actual config file and GPU")
def test_run_training():
    """Test run_training function with actual config."""
    config_path = "config.json"
    run_training(config_path)
    # This would be an integration test requiring actual files


@patch("controlnet_package.controlnet_package.load_config")
@patch("controlnet_package.controlnet_package.DepthDataset")
@patch("controlnet_package.controlnet_package.DataLoader")
@patch("controlnet_package.controlnet_package.pipeline_factory")
@patch("controlnet_package.controlnet_package.train_controlnet")
def test_run_training_mocked(mock_train, mock_pipeline, mock_loader,
                            mock_dataset, mock_config):
    """Test run_training function with mocks."""
    # Setup mocks
    config_dict = {
        "pretrained_model": "test_model",
        "controlnet_model": "test_controlnet",
        "train_data_root": "data/train",
        "image_size": 128,
        "batch_size": 1,
        "ckpt_dir": "checkpoints",
        "num_workers": 0
    }
    mock_config.return_value = config_dict
    mock_dataset.return_value.__len__.return_value = 2
    mock_pipeline.return_value = MagicMock()

    # Run the function
    run_training("fake_config.json")

    # Assertions
    mock_config.assert_called_once()
    mock_dataset.assert_called_once_with("data/train", 128)
    mock_loader.assert_called_once()
    mock_pipeline.assert_called_once_with("test_model", "test_controlnet", "cpu")
    mock_train.assert_called_once()

"""Tests for model module."""

import pytest
import torch
from unittest.mock import patch, MagicMock
from controlnet_package.model import pipeline_factory


@pytest.mark.skip(reason="Requires downloading large models")
def test_pipeline_factory_with_actual_models():
    """Test pipeline factory function with actual models."""
    pretrained_model = "runwayml/stable-diffusion-v1-5"
    controlnet_model = "lllyasviel/sd-controlnet-depth"
    device = "cpu"
    
    pipeline = pipeline_factory(pretrained_model, controlnet_model, device)
    
    # Check that pipeline is created correctly
    assert pipeline is not None
    assert pipeline.device.type == "cpu"
    assert pipeline.controlnet is not None
    assert pipeline.scheduler is not None
    assert pipeline.unet is not None
    assert pipeline.vae is not None


@patch('controlnet_package.model.ControlNetModel.from_pretrained')
@patch('controlnet_package.model.StableDiffusionControlNetPipeline.from_pretrained')
def test_pipeline_factory_mock(mock_pipeline_fn, mock_controlnet_fn):
    """Test pipeline factory function with mocks."""
    pretrained_model = "runwayml/stable-diffusion-v1-5"
    controlnet_model = "lllyasviel/sd-controlnet-depth"
    device = "cpu"
    
    # Create mock objects
    mock_controlnet = MagicMock()
    mock_pipeline = MagicMock()
    
    # Configure the mocks
    mock_controlnet_fn.return_value = mock_controlnet
    mock_pipeline_fn.return_value = mock_pipeline
    mock_pipeline.to.return_value = mock_pipeline  # This is the key change
    
    # Call the function being tested
    result = pipeline_factory(pretrained_model, controlnet_model, device)
    
    # Verify the mocks were called correctly
    mock_controlnet_fn.assert_called_once_with(controlnet_model, torch_dtype=torch.float32)
    mock_pipeline_fn.assert_called_once_with(
        pretrained_model,
        controlnet=mock_controlnet,
        torch_dtype=torch.float32
    )
    mock_pipeline.to.assert_called_once_with(device)
    
    # Verify result
    assert result is mock_pipeline


@patch('controlnet_package.model.ControlNetModel.from_pretrained')
@patch('controlnet_package.model.StableDiffusionControlNetPipeline.from_pretrained')
def test_pipeline_factory_with_different_devices(mock_pipeline_fn, mock_controlnet_fn):
    """Test pipeline factory with different devices."""
    pretrained_model = "runwayml/stable-diffusion-v1-5"
    controlnet_model = "lllyasviel/sd-controlnet-depth"
    
    # Create mock objects
    mock_controlnet = MagicMock()
    mock_pipeline = MagicMock()
    
    # Configure the mocks
    mock_controlnet_fn.return_value = mock_controlnet
    mock_pipeline_fn.return_value = mock_pipeline
    mock_pipeline.to.return_value = mock_pipeline
    
    # Test with 'cuda' device
    result1 = pipeline_factory(pretrained_model, controlnet_model, "cuda")
    mock_pipeline.to.assert_called_with("cuda")
    assert result1 is mock_pipeline
    
    # Reset the mock to clear call history
    mock_pipeline.reset_mock()
    
    # Test with 'cpu' device
    result2 = pipeline_factory(pretrained_model, controlnet_model, "cpu")
    mock_pipeline.to.assert_called_with("cpu")
    assert result2 is mock_pipeline


@patch('controlnet_package.model.ControlNetModel.from_pretrained')
@patch('controlnet_package.model.StableDiffusionControlNetPipeline.from_pretrained')
def test_pipeline_factory_torch_dtype(mock_pipeline_fn, mock_controlnet_fn):
    """Test that the pipeline factory uses the correct torch dtype."""
    pretrained_model = "runwayml/stable-diffusion-v1-5"
    controlnet_model = "lllyasviel/sd-controlnet-depth"
    device = "cpu"
    
    # Create and configure mock
    mock_pipeline = MagicMock()
    mock_pipeline.to.return_value = mock_pipeline
    mock_pipeline_fn.return_value = mock_pipeline
    
    # Call the function being tested
    result = pipeline_factory(pretrained_model, controlnet_model, device)
    
    # Verify torch dtype is correctly passed
    mock_controlnet_fn.assert_called_once_with(controlnet_model, torch_dtype=torch.float32)
    mock_pipeline_fn.assert_called_once()
    _, kwargs = mock_pipeline_fn.call_args
    assert kwargs.get('torch_dtype') == torch.float32
    assert result is mock_pipeline

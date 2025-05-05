"""Tests for controlnet_package module."""

import pytest
from controlnet_package import __version__


def test_version():
    """Test version is a string."""
    assert isinstance(__version__, str)

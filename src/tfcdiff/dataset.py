"""
Dataset for TFCDiff — reuses PulseNoiseDataset from ddpm.

The DCT transform, truncation, and eta scaling are handled in the
diffusion module, so this dataset returns time-domain signals identical
to the DDPM dataset.

Usage:
    from src.tfcdiff.dataset import PulseNoiseDataset
"""

from src.ddpm.dataset import PulseNoiseDataset  # noqa: F401

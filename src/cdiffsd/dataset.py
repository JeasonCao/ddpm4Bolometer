"""
PyTorch dataset for CDiffSD: paired clean/noisy windows plus a separate
noise sample for the Cold Diffusion forward process.

Returns (x_clean, x_noisy, n_fwd, scale) where n_fwd is an independent
noise sample from the noise library used to corrupt x_clean during training.

Usage:
    from src.cdiffsd.dataset import ColdDiffusionDataset
    ds = ColdDiffusionDataset(clean_dir='...', noise_dir='...')
    x_clean, x_noisy, n_fwd, scale = ds[0]
"""

import os
import glob
import threading
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ColdDiffusionDataset(Dataset):
    """Dataset that pairs clean pulses with noise windows, plus provides
    an independent noise sample for the forward diffusion process.

    Each sample returns:
        x_clean : (1, L) -- clean pulse [normalized]
        x_noisy : (1, L) -- clean + noise [normalized]
        n_fwd   : (1, L) -- independent noise for forward process [normalized]
        scale   : float   -- normalization factor

    Two independent noise permutations are maintained:
        - _noise_perm: pairs noise with clean to form x_noisy (conditioning)
        - _noise_perm_fwd: provides the forward process noise sample

    Parameters
    ----------
    clean_path : str
        Single .h5 file or directory containing clean_*.h5 shards.
    noise_path : str
        Single .h5 file or directory containing noise_*.h5 shards.
    """

    def __init__(self, clean_path: str, noise_path: str):
        self.clean_files = self._resolve_files(clean_path, 'clean_*.h5')
        self.noise_files = self._resolve_files(noise_path, 'noise_*.h5')

        if not self.clean_files:
            raise FileNotFoundError(f"No clean h5 files found at {clean_path}")
        if not self.noise_files:
            raise FileNotFoundError(f"No noise h5 files found at {noise_path}")

        # Build index: (file_idx, window_idx) for each sample
        self.clean_index = []
        self.clean_counts = []
        for fpath in self.clean_files:
            with h5py.File(fpath, 'r') as f:
                n = f.attrs.get('n_windows', f.attrs.get('n_pulses'))
            self.clean_counts.append(n)
            file_idx = len(self.clean_counts) - 1
            for j in range(n):
                self.clean_index.append((file_idx, j))

        self.noise_index = []
        self.noise_counts = []
        for fpath in self.noise_files:
            with h5py.File(fpath, 'r') as f:
                n = f.attrs.get('n_windows', f.attrs.get('n_pulses'))
            self.noise_counts.append(n)
            file_idx = len(self.noise_counts) - 1
            for j in range(n):
                self.noise_index.append((file_idx, j))

        self.n_clean = len(self.clean_index)
        self.n_noise = len(self.noise_index)

        # Two independent noise permutations
        self._noise_perm = np.arange(self.n_noise)
        self._noise_perm_fwd = np.arange(self.n_noise)
        self._epoch_salt = int(time.time()) % 100000

        # Thread-local storage for h5py file handles
        self._local = threading.local()

    @staticmethod
    def _resolve_files(path, pattern):
        if os.path.isfile(path):
            return [path]
        return sorted(glob.glob(os.path.join(path, pattern)))

    def set_epoch(self, epoch: int):
        """Reshuffle both noise permutations for a new epoch."""
        rng1 = np.random.default_rng(epoch + self._epoch_salt)
        rng1.shuffle(self._noise_perm)
        # Use a different seed for the forward noise permutation
        rng2 = np.random.default_rng(epoch + self._epoch_salt + 1_000_000)
        rng2.shuffle(self._noise_perm_fwd)

    def __len__(self):
        return self.n_clean

    def _get_h5(self, kind: str, file_idx: int):
        cache_attr = f'_cache_{kind}'
        if not hasattr(self._local, cache_attr):
            setattr(self._local, cache_attr, {})
        cache = getattr(self._local, cache_attr)
        if file_idx not in cache:
            files = self.clean_files if kind == 'clean' else self.noise_files
            cache[file_idx] = h5py.File(files[file_idx], 'r')
        return cache[file_idx]

    def _get_clean(self, idx):
        file_idx, win_idx = self.clean_index[idx]
        return self._get_h5('clean', file_idx)['waveforms'][win_idx]

    def _get_noise(self, idx):
        idx = idx % self.n_noise
        file_idx, win_idx = self.noise_index[idx]
        return self._get_h5('noise', file_idx)['waveforms'][win_idx]

    def __getitem__(self, idx):
        clean = self._get_clean(idx).astype(np.float32)

        # Noise for conditioning (x_tilde = clean + noise)
        noise_idx = self._noise_perm[idx % self.n_noise]
        noise = self._get_noise(noise_idx).astype(np.float32)

        # Independent noise for forward process
        fwd_idx = self._noise_perm_fwd[idx % self.n_noise]
        noise_fwd = self._get_noise(fwd_idx).astype(np.float32)

        noisy = clean + noise

        # Normalize all by max(|noisy|)
        scale = np.max(np.abs(noisy))
        if scale < 1e-12:
            scale = 1.0
        clean = clean / scale
        noisy = noisy / scale
        noise_fwd = noise_fwd / scale

        x_clean = torch.from_numpy(clean).unsqueeze(0)
        x_noisy = torch.from_numpy(noisy).unsqueeze(0)
        n_fwd = torch.from_numpy(noise_fwd).unsqueeze(0)

        return x_clean, x_noisy, n_fwd, torch.tensor(scale, dtype=torch.float32)

"""
Simple single-signal dataset for training score networks independently.

Each score network trains on one signal type: clean pulses OR noise windows.
Each window is normalized independently to [-1, 1].

Usage:
    from src.jointdiff.dataset import SingleSignalDataset
    ds = SingleSignalDataset('/path/to/clean')   # for Score_x
    ds = SingleSignalDataset('/path/to/noise')   # for Score_n
    x, scale = ds[0]   # x: (1, L), scale: float
"""

import os
import glob
import threading

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SingleSignalDataset(Dataset):
    """Load signal windows from HDF5 shards, normalize each to [-1, 1].

    Parameters
    ----------
    path : str
        Single .h5 file or directory containing *.h5 shards.
    """

    def __init__(self, path: str):
        self.files = self._resolve_files(path)
        if not self.files:
            raise FileNotFoundError(f"No h5 files found at {path}")

        self.index = []
        self.counts = []
        for fpath in self.files:
            with h5py.File(fpath, 'r') as f:
                n = f.attrs.get('n_windows', f.attrs.get('n_pulses'))
            self.counts.append(n)
            file_idx = len(self.counts) - 1
            for j in range(n):
                self.index.append((file_idx, j))

        self.n_total = len(self.index)
        self._local = threading.local()

    @staticmethod
    def _resolve_files(path):
        if os.path.isfile(path):
            return [path]
        patterns = ['*.h5', 'clean_*.h5', 'noise_*.h5']
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(path, p)))
        return sorted(set(files))

    def __len__(self):
        return self.n_total

    def _get_h5(self, file_idx: int):
        cache_attr = '_h5_cache'
        if not hasattr(self._local, cache_attr):
            setattr(self._local, cache_attr, {})
        cache = getattr(self._local, cache_attr)
        if file_idx not in cache:
            cache[file_idx] = h5py.File(self.files[file_idx], 'r')
        return cache[file_idx]

    def __getitem__(self, idx):
        file_idx, win_idx = self.index[idx]
        waveform = self._get_h5(file_idx)['waveforms'][win_idx].astype(np.float32)

        scale = np.max(np.abs(waveform))
        if scale < 1e-12:
            scale = 1.0
        waveform = waveform / scale

        x = torch.from_numpy(waveform).unsqueeze(0)  # (1, L)
        return x, torch.tensor(scale, dtype=torch.float32)

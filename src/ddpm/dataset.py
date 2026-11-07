"""
PyTorch dataset for paired clean/noisy pulse windows from HDF5 shards.

Loads clean pulse windows and noise windows from separate shard directories,
combines them on-the-fly to produce (clean, noisy) pairs.

Usage:
    from src.ddpm.dataset import PulseNoiseDataset
    ds = PulseNoiseDataset(clean_dir='...', noise_dir='...')
    x_clean, x_noisy = ds[0]  # both (1, 10000)

    # With subset filtering (requires preprocess_index.py output):
    ds = PulseNoiseDataset(clean_dir='...', noise_dir='...', subset='pileup_low_100')
"""

import os
import glob
import json
import threading
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_index(h5_path: str) -> dict:
    """Load precomputed index JSON for a clean HDF5 shard.

    Returns dict of category -> list of int indices, or None if not found.
    """
    base = os.path.splitext(h5_path)[0]
    index_path = base + '_index.json'
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        data = json.load(f)
    return data['indices']


class PulseNoiseDataset(Dataset):
    """Dataset that pairs clean pulses with noise windows.

    Each sample returns:
        x_clean : (1, L) — clean pulse waveform [V]
        x_noisy : (1, L) — clean + noise [V]

    Noise pairing uses a shuffled permutation: each epoch, a permutation
    of noise indices is generated so every noise window is used exactly
    once but paired with a different clean pulse. Call set_epoch(e)
    before each epoch to reshuffle. This gives 100K x 100K combinatorial
    augmentation across epochs.

    Parameters
    ----------
    clean_path : str
        Single .h5 file or directory containing clean_*.h5 shards.
    noise_path : str
        Single .h5 file or directory containing noise_*.h5 shards.
    subset : str or None
        If set, filter clean samples to this category from the precomputed
        index file (e.g., 'pileup', 'low_100', 'pileup_low_200').
        Requires running preprocess_index.py first.
    """

    def __init__(self, clean_path: str, noise_path: str, subset: str = None):
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

            if subset is not None:
                idx_data = load_index(fpath)
                if idx_data is None:
                    raise FileNotFoundError(
                        f"Index file not found for {fpath}. "
                        f"Run: python3 -m src.ddpm.preprocess_index --clean_dir {fpath}")
                if subset not in idx_data:
                    raise ValueError(
                        f"Subset '{subset}' not found in index. "
                        f"Available: {list(idx_data.keys())}")
                for j in idx_data[subset]:
                    self.clean_index.append((file_idx, j))
            else:
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

        # Shuffled noise permutation — reshuffle each epoch via set_epoch()
        self._noise_perm = np.arange(self.n_noise)
        self._epoch_salt = int(time.time()) % 100000

        # Thread-local storage for h5py file handles (fork-safe)
        self._local = threading.local()

    @staticmethod
    def _resolve_files(path, pattern):
        """Accept a single .h5 file or a directory of shards."""
        if os.path.isfile(path):
            return [path]
        return sorted(glob.glob(os.path.join(path, pattern)))

    def set_epoch(self, epoch: int):
        """Reshuffle noise permutation for a new epoch."""
        rng = np.random.default_rng(epoch + self._epoch_salt)
        rng.shuffle(self._noise_perm)

    def __len__(self):
        return self.n_clean

    def _get_h5(self, kind: str, file_idx: int):
        """Get or open an h5py file handle, thread-local."""
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
        noise_idx = self._noise_perm[idx % self.n_noise]
        noise = self._get_noise(noise_idx).astype(np.float32)

        noisy = clean + noise

        # Normalize both by max(|noisy|) so signals are in ~[-1, 1]
        scale = np.max(np.abs(noisy))
        if scale < 1e-12:
            scale = 1.0
        clean = clean / scale
        noisy = noisy / scale

        # Shape: (1, L)
        x_clean = torch.from_numpy(clean).unsqueeze(0)
        x_noisy = torch.from_numpy(noisy).unsqueeze(0)

        return x_clean, x_noisy, torch.tensor(scale, dtype=torch.float32)


def _channel_id_of(fpath: str) -> int:
    """Read the channel_id attr from a shard, falling back to the filename int."""
    with h5py.File(fpath, 'r') as f:
        if 'channel_id' in f.attrs:
            return int(f.attrs['channel_id'])
    # Fallback: parse trailing integer in filename (e.g. clean_007.h5 -> 7)
    base = os.path.splitext(os.path.basename(fpath))[0]
    digits = ''.join(ch for ch in base if ch.isdigit())
    return int(digits)


class ChannelBankDataset(Dataset):
    """Pairs clean pulses with noise **within the same channel** (shard).

    For the multi-channel foundation-model denoiser. Clean shard ``clean_NNN.h5``
    and noise shard ``noise_NNN.h5`` share ``channel_id == NNN``; a clean window
    from channel c is only ever combined with a noise window from channel c, and
    the (true) ``channel_id`` is returned so the model can condition on it.

    Each sample returns:
        x_clean    : (1, L) — clean pulse, normalized by max(|noisy|)
        x_noisy    : (1, L) — clean + noise, same normalization
        scale      : ()      — max(|noisy|) before normalization
        channel_id : ()      — long, the channel's true id (shard index)

    Parameters
    ----------
    clean_path : str
        Directory of per-channel clean shards (``clean_*.h5``).
    noise_path : str
        Directory of per-channel noise shards (``noise_*.h5``); point at the
        ``noise_total`` subdir of the noise bank.
    channels : list[int] or None
        Channel ids (shard indices) to include. None = all channels present in
        both banks. Used to split train vs held-out channels.
    window_range : tuple[int, int] or None
        Restrict each channel's *clean* windows to ``[lo, hi)``. Used to split a
        channel's windows into fine-tune vs test sets. None = all windows.
        Noise pairing always draws from the full noise shard.
    max_windows : int or None
        Cap the number of clean windows used per channel (after window_range),
        for the data-efficiency sweep. None = no cap.
    """

    def __init__(self, clean_path: str, noise_path: str,
                 channels=None, window_range=None, max_windows=None):
        clean_files = PulseNoiseDataset._resolve_files(clean_path, 'clean_*.h5')
        noise_files = PulseNoiseDataset._resolve_files(noise_path, 'noise_*.h5')
        if not clean_files:
            raise FileNotFoundError(f"No clean shards at {clean_path}")
        if not noise_files:
            raise FileNotFoundError(f"No noise shards at {noise_path}")

        # Map channel_id -> shard path for each bank, then align on common ids.
        clean_by_ch = {_channel_id_of(p): p for p in clean_files}
        noise_by_ch = {_channel_id_of(p): p for p in noise_files}
        common = sorted(set(clean_by_ch) & set(noise_by_ch))
        if channels is not None:
            want = set(int(c) for c in channels)
            missing = want - set(common)
            if missing:
                raise ValueError(f"Requested channels not found in both banks: "
                                 f"{sorted(missing)}")
            common = [c for c in common if c in want]
        if not common:
            raise ValueError("No channels common to clean and noise banks.")

        # Position p (0..len-1) ↔ channel id; aligned clean/noise file lists.
        self.channel_list = common
        self.clean_files = [clean_by_ch[c] for c in common]
        self.noise_files = [noise_by_ch[c] for c in common]

        self.clean_index = []   # (position, clean_win_idx)
        self.local_pos = []     # window's local position within its channel
        self.noise_counts = []  # noise windows available per position
        self._noise_perm = []   # per-position permutation of noise windows

        for p, fpath in enumerate(self.clean_files):
            with h5py.File(fpath, 'r') as f:
                n_clean = int(f.attrs.get('n_windows', f.attrs.get('n_pulses')))
            with h5py.File(self.noise_files[p], 'r') as f:
                n_noise = int(f.attrs.get('n_windows', f.attrs.get('n_pulses')))

            lo, hi = (0, n_clean) if window_range is None else window_range
            lo = max(0, lo); hi = min(n_clean, hi)
            wins = list(range(lo, hi))
            if max_windows is not None:
                wins = wins[:max_windows]

            for local, w in enumerate(wins):
                self.clean_index.append((p, w))
                self.local_pos.append(local)
            self.noise_counts.append(max(n_noise, 1))
            self._noise_perm.append(np.arange(max(n_noise, 1)))

        self.n_samples = len(self.clean_index)
        self._epoch_salt = int(time.time()) % 100000
        self._local = threading.local()

    def set_epoch(self, epoch: int):
        """Reshuffle each channel's noise permutation for a new epoch."""
        for p in range(len(self._noise_perm)):
            rng = np.random.default_rng(epoch * 100003 + self._epoch_salt + p)
            rng.shuffle(self._noise_perm[p])

    def __len__(self):
        return self.n_samples

    def _get_h5(self, kind: str, p: int):
        cache_attr = f'_cache_{kind}'
        if not hasattr(self._local, cache_attr):
            setattr(self._local, cache_attr, {})
        cache = getattr(self._local, cache_attr)
        if p not in cache:
            files = self.clean_files if kind == 'clean' else self.noise_files
            cache[p] = h5py.File(files[p], 'r')
        return cache[p]

    def __getitem__(self, idx):
        p, clean_win = self.clean_index[idx]
        local = self.local_pos[idx]
        perm = self._noise_perm[p]
        noise_win = int(perm[local % len(perm)])

        clean = self._get_h5('clean', p)['waveforms'][clean_win].astype(np.float32)
        noise = self._get_h5('noise', p)['waveforms'][noise_win].astype(np.float32)

        noisy = clean + noise

        scale = np.max(np.abs(noisy))
        if scale < 1e-12:
            scale = 1.0
        clean = clean / scale
        noisy = noisy / scale

        x_clean = torch.from_numpy(clean).unsqueeze(0)
        x_noisy = torch.from_numpy(noisy).unsqueeze(0)
        channel_id = self.channel_list[p]
        return (x_clean, x_noisy,
                torch.tensor(scale, dtype=torch.float32),
                torch.tensor(channel_id, dtype=torch.long))

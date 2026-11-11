"""Check src/noise/stream.py against the original 10 s window generator.

1. Gaussian part: averaged periodogram of stream noise vs. the original pipeline
   (colored + white + resonances at 10 kHz -> Bessel -> decimate), same parameters.
2. Harmonics: filtered-and-decimated 10 kHz sinusoid vs. the analytic output gain.
3. Total RMS of one hour of stream noise vs. the target.
Writes a PSD comparison figure to results/plots/OFcompare/.
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.signal import sosfilt
from src.noise import stream as S
from src.noise.colored import generate_colored_noise
from src.noise.resonances import generate_resonances
from src.basics.filters import apply_bessel_decimate, make_bessel_sos

OUT = os.path.join(PROJ, 'results', 'plots', 'OFcompare'); os.makedirs(OUT, exist_ok=True)
RMS = 1.382e-3
rng = np.random.default_rng(12345)


def periodogram(x, n=10000):
    w = np.hanning(n); segs = x[:len(x) // n * n].reshape(-1, n)
    X = np.fft.rfft((segs - segs.mean(1, keepdims=True)) * w, axis=1)
    return np.fft.rfftfreq(n, 1e-3), 2 * np.mean(np.abs(X) ** 2, 0) / (1000.0 * np.sum(w ** 2))


for name, p in (('midpoint', S.midpoint_state(rng, RMS)), ('random', S.random_state(rng, RMS))):
    S.finalize_state(p)
    # stream Gaussian part: 2000 s
    g = S.gauss_block(rng, p, 2_000_000)
    f, ps = periodogram(g)
    # reference: original component functions, 400 windows of 11 s (drop the first s of filter start-up)
    ref = []
    for _ in range(400):
        n10 = 110000
        c = generate_colored_noise(rng, n10, 10000.0, p['alpha'], p['a_pink'], p['a_lin'], p['a_white'])
        c = c / c.std() * p['colored_factor'] * p['target_rms']
        w = rng.standard_normal(n10) * p['white_rms']
        r = generate_resonances(rng, n10, 10000.0, p['resonances'])
        ref.append(apply_bessel_decimate(c + w + r)[1000:11000] * p['scale'])
    fr, pr = periodogram(np.concatenate(ref))
    ratio = ps / pr
    for lo, hi in ((0.3, 1), (1, 10), (10, 100), (100, 400)):
        m = (f >= lo) & (f < hi)
        print(f'{name:8s} PSD stream/original {lo:5.1f}-{hi:5.0f} Hz: median {np.median(ratio[m]):.3f}  '
              f'band-power ratio {ps[m].sum() / pr[m].sum():.3f}')
    gw = g.reshape(-1, 10000).std(1).mean(); rw = np.array([r.std() for r in ref]).mean()
    print(f'{name:8s} Gaussian 10 s-window std: stream {gw*1e3:.4f} mV, original-chain {rw*1e3:.4f} mV')
    print(f'{name:8s} Gaussian std: stream {g.std()*1e3:.4f} mV, predicted {np.sqrt(p["var_parts"]["gauss"])*1e3:.4f} mV, '
          f'original-chain {np.concatenate(ref).std()*1e3:.4f} mV')
    # harmonic gain check on three lines
    sos = make_bessel_sos(120.0, 6, 10000.0)
    for fk in (1.4, 50.0, 150.0, 650.0):
        t = np.arange(200000) / 10000.0
        y = sosfilt(sos, np.sin(2 * np.pi * fk * t))[::10][5000:]
        yi = np.sin(2 * np.pi * fk * np.arange(len(y) + 5000)[5000:] / 1000.0 * (1 if fk < 500 else 1) + 0)
        h = S._bessel_h(fk)[0]
        pred = np.abs(h) * np.sin(2 * np.pi * fk * (np.arange(5000, 5000 + len(y)) / 1000.0) + np.angle(h))
        if name == 'midpoint':
            print(f'  harmonic {fk:6.1f} Hz: max |filtered - analytic| = {np.abs(y - pred).max():.2e} (gain {abs(h):.3e})')
    # one hour total
    n = 3_600_000
    env = S.envelope_shape(rng, n)
    tot = (S.gauss_block(rng, p, n) + S.harmonics(p, 0, n)) * (1 + p['envelope_variation'] * S.envelope_at(env, 0, n))
    wstd = tot[:n // 10000 * 10000].reshape(-1, 10000).std(1)
    print(f'{name:8s} 1 h: mean 10 s-window RMS {wstd.mean()*1e3:.4f} mV (spread {wstd.std()/wstd.mean()*100:.1f}%), global {tot.std()*1e3:.4f} mV (target {RMS*1e3:.3f}); parts gauss {np.sqrt(p["var_parts"]["gauss"])*1e3:.3f} '
          f'harm {np.sqrt(p["var_parts"]["harm"])*1e3:.3f} mV')
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
    ax[0].loglog(fr[1:], pr[1:], lw=0.8, label='original window chain (400 x 10 s)')
    ax[0].loglog(f[1:], ps[1:], lw=0.8, alpha=0.8, label='continuous stream (2000 s)')
    ax[0].set_ylabel('PSD [V$^2$/Hz]'); ax[0].legend(frameon=False); ax[0].set_title(f'Gaussian noise part, {name} state, RMS 1.382 mV', loc='left')
    ax[1].semilogx(f[1:], np.convolve(ratio[1:], np.ones(21) / 21, 'same'), lw=0.8, color='k')
    ax[1].axhline(1, color='grey', lw=0.6); ax[1].set_ylim(0.6, 1.4); ax[1].set_ylabel('ratio'); ax[1].set_xlabel('frequency [Hz]')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'stream_noise_psd_check_{name}.png'), dpi=130); plt.close(fig)
print('ALL_DONE')

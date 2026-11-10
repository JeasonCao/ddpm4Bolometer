import sys, h5py, numpy as np
sys.path.insert(0, '.')
from scipy.signal import welch, resample_poly, sosfreqz
from src.basics.filters import apply_bessel_decimate, make_bessel_sos
D = '/mnt/d/storage/research/DDPM4bolometer'; N = 200
def chain(x):
    up = resample_poly(x, 10, 1, axis=1)
    return np.stack([apply_bessel_decimate(r, 10000.0, 1000.0, 120.0, 6) for r in up])[:, :x.shape[1]]
for L in ('1p382mV', '7p5mV'):
    fn = 'eval_efficiency_2615keV.h5'
    with h5py.File(f'{D}/eval_data_holdout_denoised/efficiency_{L}/{fn}') as f:
        dd = f['waveforms_denoised_zeroinit'][:N].astype(float); nz = f['waveforms_noisy'][:N].astype(float)
        cl = np.broadcast_to(f['clean_template'][:].astype(float), dd.shape) if f['clean_template'].ndim == 1 else f['clean_template'][:N].astype(float)
    with h5py.File(f'{D}/eval_data_bessel/bessel_sim/efficiency_{L}/{fn}') as f:
        ddB = f['waveforms_denoised_zeroinit_bessel_sim'][:N].astype(float)
    with h5py.File(f'{D}/eval_data_bessel/bessel_sim_noisy/efficiency_{L}/{fn}') as f:
        nzB = f['waveforms_noisy_bessel_sim'][:N].astype(float)
    clB = chain(cl[:1].copy())
    P = lambda x: welch(x, fs=1000, nperseg=2000, axis=1)
    fr, _ = P(dd[:1])
    res = dict(raw=nz - cl, ddpm=dd - cl, rawB=nzB - clB, ddB=ddB - clB)
    psd = {k: P(v)[1].mean(0) for k, v in res.items()}
    print(f'== {L} 2615 keV, residual PSD (V^2/Hz) at f; and residual variance fraction above 100 Hz')
    for f0 in (100, 150, 200, 300, 450):
        i = np.argmin(abs(fr - f0)); print(f'  {f0:3d} Hz  ' + '  '.join(f'{k} {psd[k][i]:.1e}' for k in psd))
    df = fr[1] - fr[0]
    for k in psd:
        tot = psd[k].sum() * df; hi = psd[k][fr > 100].sum() * df; lo = psd[k][fr < 20].sum() * df
        print(f'  {k:5s} rms {np.sqrt(tot)*1e6:8.2f} uV   >100 Hz: {hi/tot:.2e}   <20 Hz: {lo/tot:.3f}')
    # float32 floor check
    x = ddB[0].astype(np.float32).astype(float); print('  float32 rounding PSD of ddB event:', f"{P((ddB[:1]-ddB[:1].astype(np.float32).astype(float)))[1].mean():.1e}", 'stored dtype float32')

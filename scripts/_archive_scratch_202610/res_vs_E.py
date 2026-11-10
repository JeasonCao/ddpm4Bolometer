import h5py, numpy as np
from scipy.signal import welch
D = '/mnt/d/storage/research/DDPM4bolometer'; N = 300
for L in ('1p382mV', '7p5mV'):
    print(f'== {L}: DDPM residual vs energy (no Bessel)')
    print(f"{'E':>5} {'rms uV':>8} {'rms/peak':>9} {'<20Hz':>6} {'>100Hz':>7} {'amp-like':>8}")
    for E in (1, 15, 50, 239, 727, 1461, 2615, 3034):
        with h5py.File(f'{D}/eval_data_holdout_denoised/efficiency_{L}/eval_efficiency_{E}keV.h5') as f:
            dd = f['waveforms_denoised_zeroinit'][:N].astype(float); ct = f['clean_template'][:].astype(float)
        c = ct if ct.ndim == 1 else ct[0]
        r = dd - c
        fr, p = welch(r, fs=1000, nperseg=2000, axis=1); p = p.mean(0); tot = p.sum()
        # part of the residual that is just a rescaled template: r ~ a*c
        a = r @ c / (c @ c); amp_like = ((a[:, None] * c) ** 2).sum() / (r ** 2).sum()
        print(f"{E:5d} {np.sqrt((r**2).mean())*1e6:8.1f} {np.sqrt((r**2).mean())/c.max():9.2e} {p[fr<20].sum()/tot:6.3f} {p[fr>100].sum()/tot:7.4f} {amp_like:8.3f}")

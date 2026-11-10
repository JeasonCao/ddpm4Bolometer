import sys, h5py, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/of_compare')
from plot_full_of_vs_ddpm import width
EN = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 239, 511, 727, 1461, 2615, 3034]
for L in ('1p382mV', '7p5mV'):
    print(f'== {L}   Gaussian sigma (keV, per-peak calibration), all fits with success; conv = fraction passing chi2/ndf<=10')
    print(f"{'E':>5} {'raw':>8} {'DDPM':>8} {'DDPM+B':>8} {'B/DDPM':>7} {'conv DDPM':>9} {'conv D+B':>8}")
    for E in EN:
        a = h5py.File(f'results/fit_results/{L}_zeroinit/fit_efficiency_{E}keV.h5'); b = h5py.File(f'results/fit_results/{L}_zeroinit_bessel/fit_efficiency_{E}keV.h5')
        def s(f, k):
            x = f[f'amp_{k}'][:].astype(float); x = x[f[f'success_{k}'][:].astype(bool) & np.isfinite(x)]
            return width(x / x.mean() * E)[0]
        r, d, db = s(a, 'noisy'), s(a, 'denoised'), s(b, 'denoised')
        print(f"{E:5d} {r:8.4f} {d:8.4f} {db:8.4f} {db/d:7.3f} {a['converged_denoised'][:].mean():9.3f} {b['converged_denoised'][:].mean():8.3f}")

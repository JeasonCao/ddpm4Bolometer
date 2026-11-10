"""Energy resolution from Octopus OF on the DDPM-output stream (build_ddpm_stream.py).

Triggers sit exactly on the block onsets, so OF rows are matched to blocks by trigger sample.
sigma_E = std / mean x E per peak (and a Gaussian fit within +-3 robust sigma), for half B
(evaluation half: its blocks were not used for the residual/ANPS) and for all events.

Usage: python analyze_ddpm_of.py --run 999996
"""
import os, sys, csv, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plot_full_of_vs_ddpm import width

W = '/mnt/d/storage/research/DDPM4bolometer/OFcompare_ddpm'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(f'{W}/run{a.run}/of_ideal.csv')))
    ts = np.array([int(float(r['module_triggersample'])) for r in rows])
    amp = np.array([float(r['optimumfilter_amplitude']) for r in rows])
    good = np.array([r['optimumfilter_good'] == 'True' for r in rows])
    lim = np.array([r['optimumfilter_Chi2AtLowerLimit'] == 'True' or r['optimumfilter_Chi2AtUpperLimit'] == 'True' for r in rows])
    idx = {t: k for k, t in enumerate(ts)}
    ev = list(csv.DictReader(open(f'{W}/run{a.run}/truth/events.csv')))
    out = {'run': a.run, 'n_rows': len(rows), 'peaks': {}}
    for E in sorted({int(r['energy_keV']) for r in ev}):
        res = {}
        for sel in ('B', 'all'):
            k = np.array([idx[int(r['onset_sample'])] for r in ev if int(r['energy_keV']) == E and (sel == 'all' or r['half'] == sel)])
            ok = good[k] & ~lim[k]; x = amp[k[ok]]
            s, ds, sd = width(x / x.mean() * E)
            res[sel] = dict(n=int(len(k)), n_dropped=int((~ok).sum()), std_keV=float(x.std() / x.mean() * E),
                            gauss_sigma_keV=float(s), gauss_err_keV=float(ds))
        out['peaks'][E] = res
        print(f"run {a.run} {E:4d} keV: half B std {res['B']['std_keV']:.4f} (Gauss {res['B']['gauss_sigma_keV']:.4f}, dropped {res['B']['n_dropped']}) | "
              f"all std {res['all']['std_keV']:.4f} (Gauss {res['all']['gauss_sigma_keV']:.4f})")
    json.dump(out, open(f'{W}/run{a.run}/resolution_ideal.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

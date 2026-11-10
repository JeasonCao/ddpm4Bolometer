"""Energy resolution from a local Octopus OF table (export_of_table.py) matched to the truth.

Per-peak calibration: sigma_E = std(OF amplitude) / mean(OF amplitude) * E, so the 2% gain
non-linearity between 15 and 2615 keV does not enter. Noise windows are converted with the gain of
each peak. Signal triggers are matched to the true onset within +-300 samples.

Usage: python analyze_local_of.py --run 999996 [--variant ideal]
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[3]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import argparse, csv, json
import numpy as np

W = '/mnt/d/storage/research/DDPM4bolometer/OFcompare_local'
TRUTH = '/mnt/d/storage/research/DDPM4bolometer/OFcompare_truth'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, required=True); ap.add_argument('--variant', default='ideal')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(f'{W}/run{a.run}/of_{a.variant}.csv')))
    col = lambda k: np.array([r[k] == 'True' if r[k] in ('True', 'False') else float(r[k]) for r in rows])
    ts = col('module_triggersample').astype(np.int64); isn = col('module_isnoise').astype(bool)
    good = col('optimumfilter_good').astype(bool)
    amp, dly, chi = col('optimumfilter_amplitude'), col('optimumfilter_delay'), col('optimumfilter_Chi2')
    at_lim = col('optimumfilter_Chi2AtLowerLimit').astype(bool) | col('optimumfilter_Chi2AtUpperLimit').astype(bool)

    ev = list(csv.DictReader(open(f'{TRUTH}/run{a.run}/truth/events.csv')))
    on = np.array([int(r['onset_sample']) for r in ev]); E = np.array([int(r['energy_keV']) for r in ev])
    sig = np.flatnonzero(~isn & good)
    j = np.clip(np.searchsorted(on, ts[sig]), 1, len(on) - 1)
    near = np.where(np.abs(ts[sig] - on[j - 1]) < np.abs(ts[sig] - on[j]), j - 1, j)
    matched = np.abs(ts[sig] - on[near]) <= 300
    Etrig = np.zeros(len(ts), int); Etrig[sig[matched]] = E[near[matched]]

    out = {'run': a.run, 'variant': a.variant, 'n_noise_windows': int((isn & good).sum()),
           'n_signal_unmatched': int((~matched).sum()), 'peaks': {}}
    gains = {}
    for e in sorted(set(E.tolist())):
        m = (Etrig == e) & ~at_lim
        x = amp[m]; gains[e] = x.mean() / e
        out['peaks'][int(e)] = {'n_triggered': int((Etrig == e).sum()), 'n_true': int((E == e).sum()),
                           'n_delay_at_limit': int(((Etrig == e) & at_lim).sum()),
                           'sigma_keV': float(x.std() / gains[e]), 'FWHM_keV': float(2.355 * x.std() / gains[e]),
                           'mean_amp': float(x.mean()), 'delay_ms_mean': float(dly[m].mean() * 1e3),
                           'delay_ms_std': float(dly[m].std() * 1e3), 'chi2_median': float(np.median(chi[m]))}
    xn = amp[isn & good]
    out['noise'] = {f'sigma_keV_gain{e}': float(xn.std() / g) for e, g in gains.items()}
    out['noise']['mean_keV_gain2615'] = float(xn.mean() / gains[max(gains)])
    print(json.dumps(out, indent=1))
    json.dump(out, open(f'{W}/run{a.run}/resolution_{a.variant}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

"""Cut the DDPM windows of a pilot run (999996/999997, generated with ddpm=False) from its bin files.

Same layout as the full runs (generate_of_stream.open_ddpm_files): eval_efficiency_{E}keV.h5 with
rows sorted by event_id, window = onset - 1500 .. onset + 8500, values decoded from the bin stream
exactly as Octopus reads them. Output: eval_data_OFcompare/pilot_run{run}/.

Usage: python experiments/of_comparison/cut_ddpm_windows.py --run 999996
"""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, h5py
import generate_of_stream as GEN
from verify_of_stream import read_bin, volts

PRE, POST = 1500, 8500


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, required=True); a = ap.parse_args()
    cfg = GEN.RUNS[a.run]
    root = os.path.join(GEN.TRUTH, f'run{a.run}'); raw_dir = os.path.join(GEN.STAGE, f'run{a.run}', 'raw')
    ev = np.loadtxt(os.path.join(root, 'truth', 'events.csv'), delimiter=',', skiprows=1)
    seg = list(csv.DictReader(open(os.path.join(root, 'truth', 'segments.csv'))))
    with h5py.File(os.path.join(root, 'templates', 'templates.h5')) as f:
        T = {e: f[f'{e}keV'][:PRE + POST] for e in cfg['energies']}
    eid, E, on, part = ev[:, 0].astype(int), ev[:, 1].astype(int), ev[:, 2].astype(np.int64), ev[:, 4].astype(int)
    counts = {e: int((E == e).sum()) for e in cfg['energies']}

    out_cfg = dict(cfg, level=f'pilot_run{a.run}')
    H = GEN.open_ddpm_files(out_cfg, a.run, {e: np.pad(t, (0, 0)) for e, t in T.items()}, counts)
    row = {e: 0 for e in cfg['energies']}
    for s in seg:
        d, s0 = int(s['partial']), int(s['start_sample'])
        _, _, R, data = read_bin(os.path.join(raw_dir, s['file']))
        st = {k: float(s[k]) for k in ('ac_a_base', 'pt_a_base', 'white_rms', 'alpha', 'envelope_variation',
                                       'f_cross', 'scale', 'target_rms')}
        st['resonances'] = [None] * int(s['n_resonances'])     # only the count is used; the json column is not CSV-quoted
        npv = GEN.noise_param_values(st)
        for i in np.flatnonzero(part == d):
            e, o = int(E[i]), int(on[i])
            win = volts(data[o - PRE - s0:o + POST - s0], R)
            f, r = H[e], row[e]; row[e] += 1
            f['waveforms_noisy'][r] = win
            nrms = float(np.std(win - T[e]))
            f['noise_rms'][r] = nrms
            f['snr_db'][r] = 20 * np.log10(T[e].max() / nrms)
            f['event_id'][r], f['onset_sample'][r], f['segment_id'][r], f['noise_state'][r] = eid[i], o, d, d
            f['t_onset_stream'][r] = o / 1000.0
            for k, v in npv.items():
                f['noise_params'][k][r] = v
            f['noise_params']['xfade_weight_neighbour'][r] = 0.0       # stationary run, no crossfades
        print(f'partial {d}: done', flush=True)
    for e, f in H.items():
        assert row[e] == counts[e] and np.all(np.diff(f['event_id'][:]) > 0)
        print(f'{e} keV: {row[e]} windows, mean noise RMS {f["noise_rms"][:].mean() * 1e3:.4f} mV -> {f.filename}')
        f.close()


if __name__ == '__main__':
    main()

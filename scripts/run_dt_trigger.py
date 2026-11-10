"""
Run a blind derivative-threshold (DT) trigger on eval waveforms and compare
trigger efficiency before vs after DDPM denoising.

The trigger decision (src/basics/trigger.dt_trigger_blind) never sees the
true pulse onset — it scans the whole trace with a causal rolling baseline
and dead-time debounce, exactly like a real continuous DAQ trigger would.
The known onset is used ONLY here, downstream, to SCORE the result (truth
matching): a "hit" is any blind trigger that lands within
[onset - match_pre_ms, onset + match_post_ms]. The pre-onset margin absorbs
the small "early leak" effect found in the denoised waveforms (the model
introduces a slight precursor signal a few tens of ms before the labeled
onset — see diagnostics in scripts/test/). Any other blind triggers found
elsewhere in the trace are not part of the same physical pulse — they're
recorded separately (as a dark/background-trigger-rate diagnostic) for
possible use as a cut later, not folded into this pulse's efficiency.

Usage
-----
python -u scripts/run_dt_trigger.py \
    --input_dir /mnt/d/storage/research/DDPM4bolometer/eval_data_holdout_denoised/efficiency_1p2mV \
    --output_dir results/plots/trigger \
    --nsigma 4.0 --n_above 5 --window_ms 300 --dead_ms 300 \
    --match_pre_ms 100 --match_post_ms 200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')

import argparse
import glob
import json
import h5py
import numpy as np

from src.basics.trigger import dt_trigger_blind

ROLLING_WINDOW_MS = 300.0   # causal rolling-baseline window
DEAD_TIME_MS      = 300.0   # debounce gap between accepted triggers
MATCH_PRE_MS       = 100.0   # scoring window: this many ms BEFORE true onset
MATCH_POST_MS      = 200.0   # scoring window: this many ms AFTER true onset


def _score(waveforms, fs, onset_idx, window_n, dead_n, match_pre_n, match_post_n, kw):
    """Blind-scan every event, then truth-match against onset_idx for scoring.

    Returns: matched (bool array), match_time_ms (float array, nan if miss),
    extra_times_ms (list of lists — blind triggers NOT matched to this pulse,
    in ms relative to trace start; kept for future dark-rate cuts),
    n_triggers (int array — TOTAL blind triggers found per event, matched or not).
    """
    n_events = waveforms.shape[0]
    matched        = np.zeros(n_events, dtype=bool)
    match_time_ms  = np.full(n_events, np.nan)
    extra_times_ms = []
    n_triggers     = np.zeros(n_events, dtype=int)

    lo = onset_idx - match_pre_n
    hi = onset_idx + match_post_n

    for i in range(n_events):
        idxs, _, _, _ = dt_trigger_blind(waveforms[i], fs, window_n,
                                         dead_samples=dead_n, **kw)
        n_triggers[i] = len(idxs)
        if len(idxs) == 0:
            extra_times_ms.append([])
            continue
        in_match = (idxs >= lo) & (idxs <= hi)
        if in_match.any():
            matched[i] = True
            match_time_ms[i] = float(idxs[in_match][0]) / fs * 1e3
        extra_times_ms.append((idxs[~in_match] / fs * 1e3).tolist())

    return matched, match_time_ms, extra_times_ms, n_triggers


def process_file(h5_path: str, nsigma: float, n_above: int,
                 use_fixed_threshold: bool, fixed_threshold: float,
                 window_ms: float, dead_ms: float, match_pre_ms: float, match_post_ms: float,
                 denoised_field: str = 'waveforms_denoised'):
    with h5py.File(h5_path, 'r') as f:
        energy_kev = float(f.attrs['energy_kev'])
        fs         = float(f.attrs['f_sample'])
        onset_s    = float(f.attrs['onset_s'])
        noisy      = f['waveforms_noisy'][:]
        if denoised_field not in f:
            raise KeyError(f"{os.path.basename(h5_path)} has no dataset '{denoised_field}' "
                           f"(available: {[k for k in f.keys() if k.startswith('waveforms_denoised')]})")
        denoised   = f[denoised_field][:]
        snr_db     = f['snr_db'][:]

    onset_idx   = int(round(onset_s * fs))
    window_n    = int(round(window_ms * 1e-3 * fs))
    dead_n      = int(round(dead_ms * 1e-3 * fs))
    match_pre_n  = int(round(match_pre_ms * 1e-3 * fs))
    match_post_n = int(round(match_post_ms * 1e-3 * fs))

    kw = dict(n_above=n_above, nsigma=nsigma,
             use_fixed_threshold=use_fixed_threshold, fixed_threshold=fixed_threshold)

    matched_n, mt_n, extra_n, ntrig_n = _score(noisy,    fs, onset_idx, window_n, dead_n, match_pre_n, match_post_n, kw)
    matched_d, mt_d, extra_d, ntrig_d = _score(denoised, fs, onset_idx, window_n, dead_n, match_pre_n, match_post_n, kw)

    eff_n = float(matched_n.mean() * 100)
    eff_d = float(matched_d.mean() * 100)

    jit_n = mt_n[matched_n] - onset_s * 1e3
    jit_d = mt_d[matched_d] - onset_s * 1e3

    # Dark/background-trigger-rate diagnostics: blind triggers elsewhere in
    # the trace, unrelated to this pulse — not used in the efficiency score,
    # kept as a per-event-count summary here; full per-event times saved
    # separately for any future cut.
    n_extra_n = np.array([len(e) for e in extra_n])
    n_extra_d = np.array([len(e) for e in extra_d])

    summary = {
        'energy_kev':       energy_kev,
        'n_total':          int(len(noisy)),
        'snr_mean_db':      round(float(np.nanmean(snr_db)), 2),
        'trigger_eff_before': round(eff_n, 2),
        'trigger_eff_after':  round(eff_d, 2),
        'jitter_ms_before':  round(float(np.mean(jit_n)), 3) if len(jit_n) else None,
        'jitter_ms_after':   round(float(np.mean(jit_d)), 3) if len(jit_d) else None,
        'jitter_std_ms_before': round(float(np.std(jit_n)), 3) if len(jit_n) else None,
        'jitter_std_ms_after':  round(float(np.std(jit_d)), 3) if len(jit_d) else None,
        'extra_trigger_rate_before': round(float(n_extra_n.mean()), 4),
        'extra_trigger_rate_after':  round(float(n_extra_d.mean()), 4),
        'frac_events_with_extra_before': round(float((n_extra_n > 0).mean() * 100), 2),
        'frac_events_with_extra_after':  round(float((n_extra_d > 0).mean() * 100), 2),
        # How often does MORE THAN ONE blind trigger fire within a single
        # event's trace (matched + extra combined)?
        'frac_events_multi_trigger_before': round(float((ntrig_n >= 2).mean() * 100), 2),
        'frac_events_multi_trigger_after':  round(float((ntrig_d >= 2).mean() * 100), 2),
        'mean_n_triggers_before': round(float(ntrig_n.mean()), 4),
        'mean_n_triggers_after':  round(float(ntrig_d.mean()), 4),
        'nsigma':    nsigma,
        'n_above':   n_above,
        'use_fixed_threshold': use_fixed_threshold,
        'fixed_threshold': fixed_threshold if use_fixed_threshold else None,
        'window_ms': window_ms,
        'dead_ms':   dead_ms,
        'match_pre_ms':  match_pre_ms,
        'match_post_ms': match_post_ms,
    }
    return summary, extra_n, extra_d, matched_n, matched_d, ntrig_n, ntrig_d


def run(input_dirs, output_dir, nsigma, n_above, use_fixed_threshold,
       fixed_threshold, window_ms, dead_ms, match_pre_ms, match_post_ms,
       denoised_field='waveforms_denoised'):
    json_dir = os.path.join(output_dir, 'json')
    raw_dir  = os.path.join(output_dir, 'raw')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    all_summaries = {}
    all_ntrig_n, all_ntrig_d = [], []
    for in_dir in input_dirs:
        files = sorted(glob.glob(os.path.join(in_dir, 'eval_efficiency_*.h5')))
        if not files:
            print(f'No eval files found in {in_dir}')
            continue

        ds_tag = os.path.basename(in_dir.rstrip('/')).replace('efficiency_', '')
        print(f'\n=== {ds_tag} ({len(files)} files) ===')
        rows = []
        for fp in files:
            s, extra_n, extra_d, matched_n, matched_d, ntrig_n, ntrig_d = process_file(
                fp, nsigma, n_above, use_fixed_threshold, fixed_threshold,
                window_ms, dead_ms, match_pre_ms, match_post_ms,
                denoised_field=denoised_field)
            rows.append(s)
            all_ntrig_n.append(ntrig_n)
            all_ntrig_d.append(ntrig_d)
            print(f"  {s['energy_kev']:>7.0f} keV  "
                  f"(SNR {s['snr_mean_db']:>6.1f} dB)  "
                  f"trigger: {s['trigger_eff_before']:>6.2f}% -> {s['trigger_eff_after']:>6.2f}%  "
                  f"| extra/evt: {s['extra_trigger_rate_before']:.3f} -> {s['extra_trigger_rate_after']:.3f}  "
                  f"| multi-trig%: {s['frac_events_multi_trigger_before']:.2f} -> {s['frac_events_multi_trigger_after']:.2f}")

            energy_tag = os.path.splitext(os.path.basename(fp))[0]
            energy_tag = energy_tag[len('eval_efficiency_'):]
            json_path = os.path.join(json_dir, f'trigger_{energy_tag}_{ds_tag}.json')
            with open(json_path, 'w') as jf:
                json.dump(s, jf, indent=2)

            # Raw per-event data for future cuts: matched flag + full list of
            # "extra" (unmatched, dark/background) blind trigger times.
            raw_path = os.path.join(raw_dir, f'trigger_raw_{energy_tag}_{ds_tag}.h5')
            vlen_f64 = h5py.special_dtype(vlen=np.dtype('float64'))
            with h5py.File(raw_path, 'w') as rf:
                rf.attrs['energy_kev'] = s['energy_kev']
                rf.create_dataset('matched_before', data=matched_n)
                rf.create_dataset('matched_after',  data=matched_d)
                rf.create_dataset('n_triggers_before', data=ntrig_n)
                rf.create_dataset('n_triggers_after',  data=ntrig_d)
                ds_n = rf.create_dataset('extra_times_ms_before', (len(extra_n),), dtype=vlen_f64)
                ds_d = rf.create_dataset('extra_times_ms_after',  (len(extra_d),), dtype=vlen_f64)
                for i, e in enumerate(extra_n):
                    ds_n[i] = np.array(e, dtype=np.float64)
                for i, e in enumerate(extra_d):
                    ds_d[i] = np.array(e, dtype=np.float64)

        rows.sort(key=lambda r: r['energy_kev'])
        all_summaries[ds_tag] = rows

    if all_ntrig_n:
        cat_n = np.concatenate(all_ntrig_n)
        cat_d = np.concatenate(all_ntrig_d)
        print('\n--- Multi-trigger-per-event summary (all energies/files combined) ---')
        for label, cat in [('before', cat_n), ('after', cat_d)]:
            vals, counts = np.unique(cat, return_counts=True)
            dist = ', '.join(f'{v} trig: {c} ({c/len(cat)*100:.2f}%)' for v, c in zip(vals, counts))
            print(f'  {label}: {dist}')

    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description='Blind DT trigger efficiency, before vs after denoising. '
                    'Onset is used only for scoring (truth-matching), never in the decision.')
    parser.add_argument('--input_dir', nargs='+', required=True,
                        help='One or more directories of eval_efficiency_*.h5 files '
                             '(must contain waveforms_noisy and the field given by --denoised_field)')
    parser.add_argument('--output_dir', default=os.path.join(_RESULTS_DIR, 'plots', 'trigger'))
    parser.add_argument('--nsigma',     type=float, default=4.0,
                        help='Threshold = mu_roll + nsigma*rms_roll (Octopus NRMSList default: 4.0)')
    parser.add_argument('--n_above',    type=int, default=5,
                        help='Consecutive dv samples that must all exceed threshold '
                             '(Octopus NAboveList default: 5)')
    parser.add_argument('--use_fixed_threshold', action='store_true',
                        help='Use threshold = mu_roll + fixed_threshold instead of nsigma*RMS')
    parser.add_argument('--fixed_threshold', type=float, default=10.0,
                        help='Absolute threshold offset [V] when --use_fixed_threshold is set')
    parser.add_argument('--window_ms',  type=float, default=ROLLING_WINDOW_MS,
                        help='Causal rolling-baseline window length [ms]')
    parser.add_argument('--dead_ms',    type=float, default=DEAD_TIME_MS,
                        help='Dead-time debounce gap between accepted triggers [ms]')
    parser.add_argument('--match_pre_ms',  type=float, default=MATCH_PRE_MS,
                        help='Scoring window extends this many ms BEFORE true onset [ms]')
    parser.add_argument('--match_post_ms', type=float, default=MATCH_POST_MS,
                        help='Scoring window extends this many ms AFTER true onset [ms]')
    parser.add_argument('--denoised_field', default='waveforms_denoised',
                        help='Which denoised dataset to score (2026-09-21). Default keeps the '
                             'historical behaviour. Use waveforms_denoised_zeroinit for the '
                             'zero-init method. Each field needs its own --output_dir, since the '
                             'outputs are always labelled before/after regardless of the source.')
    args = parser.parse_args()

    run(args.input_dir, args.output_dir, args.nsigma, args.n_above,
        args.use_fixed_threshold, args.fixed_threshold,
        args.window_ms, args.dead_ms, args.match_pre_ms, args.match_post_ms,
        denoised_field=args.denoised_field)
    print('\nAll done.')


if __name__ == '__main__':
    main()

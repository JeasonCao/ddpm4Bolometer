"""Continuous data streams for the DDPM vs. optimal-filter (Octopus) comparison.

One run = one continuous 1 kHz stream of fixed-template pulses on CUORE-like noise.
It is written in the Octopus "Cupid" bin format (one ~24 h partial per file), together
with the truth tables. For the full-energy runs the same quantised stream is also cut
into 10 s DDPM windows, stored per energy in the eval_efficiency_{E}keV.h5 layout.

Runs
  999996  1.382 mV  15 + 2615 keV   stationary noise (mid-range parameters)   pilot
  999997  7.5 mV    15 + 2615 keV   stationary noise (mid-range parameters)   pilot
  999998  1.382 mV  17 energies     one noise state per partial (day)
  999999  7.5 mV    17 energies     one noise state per partial (day)

Layout rules (samples at 1 kHz)
  * pulse onset on the sample grid; DDPM window = onset - 1500 .. onset + 8500
  * after a pulse: wait max(8.5 s, time for the template to fall below tail_sigma x noise RMS;
    0.1 for the pilots, 0.01 for the full runs, because the OF resolves tails the DDPM does not),
    then a uniform 0-2 s jitter, then the next pulse's 1.5 s pre-trigger
  * roughly every 60 s a 12 s quiet period; its noise trigger sits 2.5 s in, so the
    10 s / 1.5 s Octopus window lies 1 s inside the quiet period
  * partial boundaries ~every 24 h, placed at a point where every earlier pulse has
    decayed below tail_sigma; noise states change there with a 1 h cos/sin crossfade

Usage
  python scripts/of_compare/generate_of_stream.py --run 999996
"""
import os, sys, json, time, argparse, subprocess, csv
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, 'scripts')); os.chdir(PROJ)
import io, contextlib
import numpy as np
import h5py

from src.noise import stream as S
from src.pulse.simulator import simulate_pulse
import generate_eval_dataset as G

FS = 1000
PRE, POST = 1500, 8500                 # DDPM / Octopus window around the onset
WIN = PRE + POST
TEMPLATE_S = 41.5                      # long template: 1.5 s pre + 40 s
MIN_AFTER = 8500                       # samples kept free after an onset at least
JITTER = 2000                          # uniform 0..2 s extra gap
QUIET = 12000                          # quiet period length
NOISE_TRIG_OFFSET = 2500               # noise trigger position inside a quiet period
NOISE_EVERY = 60_000                   # a quiet period at least every 60 s
DAY = 86_400_000
XFADE = 3_600_000                      # noise-state crossfade length (1 h)
MIN_LAST_PARTIAL = 2 * 3_600_000       # a shorter tail partial is merged into the previous one
N_PER_E = 5000
SEED_TAG = 20261016                    # SeedSequence([SEED_TAG, run, stream_id, index])

ALL_E = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 239, 511, 727, 1461, 2615, 3034]
RUNS = {
    999996: dict(level='1p382mV', rms=1.382e-3, energies=[15, 2615], noise='stationary',
                 prefix='20260101T000000', ddpm=False, tail_sigma=0.1),
    999997: dict(level='7p5mV', rms=7.5e-3, energies=[15, 2615], noise='stationary',
                 prefix='20260105T000000', ddpm=False, tail_sigma=0.1),
    999998: dict(level='1p382mV', rms=1.382e-3, energies=ALL_E, noise='daily',
                 prefix='20260110T000000', ddpm=True, tail_sigma=0.01),
    999999: dict(level='7p5mV', rms=7.5e-3, energies=ALL_E, noise='daily',
                 prefix='20260201T000000', ddpm=True, tail_sigma=0.01),
}
DATA = '/mnt/d/storage/research/DDPM4bolometer'
STAGE = os.path.join(DATA, 'OFcompare_octopus_staging')    # what goes to the Octopus side: bin files + README
TRUTH = os.path.join(DATA, 'OFcompare_truth')               # stays here (blind analysis): truth tables, templates
DDPM_DIR = os.path.join(DATA, 'eval_data_OFcompare')

HEADER_MAGIC, ADC_RANGE = 0x00006C20, 10.0


def code_hashes():
    import hashlib
    files = ['scripts/of_compare/generate_of_stream.py', 'src/noise/stream.py', 'src/noise/generator.py',
             'src/noise/resonances.py', 'src/basics/filters.py', 'src/pulse/simulator.py', 'scripts/generate_eval_dataset.py']
    return {f: hashlib.sha256(open(os.path.join(PROJ, f), 'rb').read()).hexdigest()[:16] for f in files}


def rng_for(run, stream_id, index=0):
    return np.random.default_rng(np.random.SeedSequence([SEED_TAG, run, stream_id, index]))


# ── encoding (octopus_convert2.py trans_bin2, Octopus RawDataHandlerCupid) ──
def encode(x):
    if x.min() <= -ADC_RANGE or x.max() >= ADC_RANGE:
        raise ValueError(f'signal outside +-{ADC_RANGE} V: {x.min()} .. {x.max()}')
    return ((x + ADC_RANGE) * (2 ** 24 - 1) / (2 * ADC_RANGE)).astype(np.uint32) << np.uint32(8)


def decode(u):
    return (u >> np.uint32(8)).astype(np.float64) / 0x800000 * ADC_RANGE - ADC_RANGE


def write_bin(path, u):
    with open(path, 'wb') as f:
        f.write(np.array([HEADER_MAGIC], '<u4').tobytes())
        f.write(np.array([FS, ADC_RANGE], '<f4').tobytes())
        f.write(u.astype('<u4').tobytes())


# ── templates ────────────────────────────────────────────────────────────────
def make_templates(energies, sigma, tail_sigma):
    with contextlib.redirect_stdout(io.StringIO()):
        params, eq = G._find_stable_params([energies[0]], seed=1)   # the eval-set detector (seed 1)
    T, clear = {}, {}
    for E in energies:
        _, v = simulate_pulse(float(E), params, eq, duration=TEMPLATE_S, t_onset=PRE / FS, f_sample=FS)
        v = v[:int(TEMPLATE_S * FS)].astype(np.float64)
        T[E] = v
        above = np.nonzero(np.abs(v) > tail_sigma * sigma)[0]
        clear[E] = int(above.max() - PRE + 1) if len(above) else 0
        if v[-1] > tail_sigma * sigma:
            raise RuntimeError(f'{E} keV template still above {tail_sigma} sigma at {TEMPLATE_S} s')
    return T, clear, dict(R0=params.R0)


# ── schedule ─────────────────────────────────────────────────────────────────
def make_schedule(run, energies, clear):
    rng = rng_for(run, 0)
    order = rng.permutation(np.repeat(np.asarray(energies), N_PER_E))
    events, noise, bounds = [], [], [0]
    cur = 0

    def quiet(cur):
        noise.append(cur + NOISE_TRIG_OFFSET)
        return cur + QUIET

    cur = quiet(cur)
    last_quiet = 0
    for E in order:
        onset = cur + int(rng.integers(0, JITTER + 1)) + PRE
        events.append((int(E), onset))
        cur = onset + max(MIN_AFTER, clear[int(E)])           # every pulse so far is < tail_sigma here
        if cur - last_quiet >= NOISE_EVERY:
            last_quiet = cur
            cur = quiet(cur)
        if cur - bounds[-1] >= DAY:
            bounds.append(cur)
    cur = quiet(cur)                                          # lead-out
    end = cur
    if end - bounds[-1] < MIN_LAST_PARTIAL and len(bounds) > 1:
        bounds.pop()
    bounds.append(end)
    return events, noise, bounds


# ── noise ────────────────────────────────────────────────────────────────────
class NoiseRun:
    """Noise states per partial; Gaussian blocks generated once and cached."""

    def __init__(self, run, cfg, bounds):
        self.run, self.cfg, self.b = run, cfg, bounds
        n_part = len(bounds) - 1
        if cfg['noise'] == 'stationary':
            st = S.finalize_state(S.midpoint_state(rng_for(run, 1, 0), cfg['rms']))
            self.states = [st] * n_part
            self.shared_harm = True
        else:
            self.states = [S.finalize_state(S.random_state(rng_for(run, 1, d), cfg['rms'])) for d in range(n_part)]
            self.shared_harm = False
        self.env = S.envelope_shape(rng_for(run, 3), bounds[-1])
        self._g = {}
        h = XFADE // 2
        self.span = [(max(0, bounds[d] - h), min(bounds[-1], bounds[d + 1] + h)) for d in range(n_part)]

    def gauss(self, d):
        if d not in self._g:
            a, z = self.span[d]
            self._g = {k: v for k, v in self._g.items() if k >= d - 1}
            self._g[d] = S.gauss_block(rng_for(self.run, 2, d), self.states[d], z - a)
        return self._g[d]

    def segment(self, d):
        """Noise over partial d: [b_d, b_{d+1})."""
        s0, s1 = self.b[d], self.b[d + 1]
        n, h = s1 - s0, XFADE // 2
        a, _ = self.span[d]
        out = self.gauss(d)[s0 - a:s1 - a].copy()
        harm_own = S.harmonics(self.states[d], s0, n)
        out += harm_own
        v = np.full(n, self.states[d]['envelope_variation'])
        for nb, lo, hi in ((d - 1, s0, min(s1, s0 + h)), (d + 1, max(s0, s1 - h), s1)):
            if nb < 0 or nb >= len(self.states) or lo >= hi:
                continue
            b = s0 if nb == d - 1 else s1                     # the boundary shared with the neighbour
            w_old, w_new = S.crossfade_weights(XFADE)
            pos = np.arange(lo, hi) - (b - h)                 # position inside the 1 h crossfade
            w_self = w_new[pos] if nb == d - 1 else w_old[pos]
            w_nb = w_old[pos] if nb == d - 1 else w_new[pos]
            an, _ = self.span[nb]
            g_nb = self.gauss(nb)[lo - an:hi - an]
            g_self = out[lo - s0:hi - s0] - harm_own[lo - s0:hi - s0]
            if self.shared_harm:
                h_mix = harm_own[lo - s0:hi - s0]
            else:
                h_mix = w_self * harm_own[lo - s0:hi - s0] + w_nb * S.harmonics(self.states[nb], lo, hi - lo)
            out[lo - s0:hi - s0] = w_self * g_self + w_nb * g_nb + h_mix
            lam = w_nb ** 2
            v[lo - s0:hi - s0] = (1 - lam) * self.states[d]['envelope_variation'] + lam * self.states[nb]['envelope_variation']
        out *= 1.0 + v * S.envelope_at(self.env, s0, n)
        return out


def state_row(d, st, cfg, run):
    r = {k: st[k] for k in ('target_rms', 'ac_a_base', 'ac_p_decay', 'pt_freq', 'pt_a_base', 'pt_p_decay',
                            'alpha', 'f_cross', 'a_white', 'white_rms', 'envelope_variation', 'colored_factor', 'scale')}
    r.update(noise_state=d, n_resonances=len(st['resonances']),
             resonances=json.dumps([{k: round(float(v), 6) for k, v in x.items()} for x in st['resonances']]),
             rms_gauss=float(np.sqrt(st['var_parts']['gauss'])), rms_harm=float(np.sqrt(st['var_parts']['harm'])),
             seed_params=f'[{SEED_TAG},{run},1,{0 if cfg["noise"] == "stationary" else d}]',
             seed_gauss=f'[{SEED_TAG},{run},2,{d}]')
    return r


# ── DDPM windows ─────────────────────────────────────────────────────────────
OLD_NP_KEYS = ['ac_a_base', 'ac_frac', 'alpha', 'colored_frac', 'envelope_variation', 'f_cross', 'n_resonances',
               'pt_a_base', 'pt_frac', 'scale_factor', 'target_rms', 'white_frac', 'white_rms']


def open_ddpm_files(cfg, run, T, counts):
    out_dir = os.path.join(DDPM_DIR, cfg['level']); os.makedirs(out_dir, exist_ok=True)
    H = {}
    for E in cfg['energies']:
        f = h5py.File(os.path.join(out_dir, f'eval_efficiency_{E}keV.h5'), 'w')
        n = counts[E]
        f.attrs.update(energy_kev=float(E), n_events=n, f_sample=float(FS), duration=WIN / FS, onset_s=PRE / FS,
                       run=run, source='OFcompare continuous stream (scripts/of_compare/generate_of_stream.py)')
        f.create_dataset('clean_template', data=T[E][:WIN], compression='gzip')
        f.create_dataset('waveforms_noisy', shape=(n, WIN), dtype='f8', chunks=(1, WIN), compression='gzip')
        for k in ('snr_db', 'noise_rms'):
            f.create_dataset(k, shape=(n,), dtype='f4')
        for k in ('event_id', 'onset_sample', 'segment_id', 'noise_state'):
            f.create_dataset(k, shape=(n,), dtype='i8')
        f.create_dataset('t_onset_stream', shape=(n,), dtype='f8')
        g = f.create_group('noise_params')
        for k in OLD_NP_KEYS + ['xfade_weight_neighbour']:
            g.create_dataset(k, shape=(n,), dtype='f4')
        H[E] = f
    return H


def noise_param_values(st):
    ac, pt, white = st['ac_a_base'] ** 2, st['pt_a_base'] ** 2, st['white_rms'] ** 2
    tot = ac + pt + white                       # same proxy as generate_eval_dataset._noise_fracs
    return dict(ac_a_base=st['ac_a_base'], ac_frac=ac / tot, alpha=st['alpha'], colored_frac=0.0,
                envelope_variation=st['envelope_variation'], f_cross=st['f_cross'],
                n_resonances=len(st['resonances']), pt_a_base=st['pt_a_base'], pt_frac=pt / tot,
                scale_factor=st['scale'], target_rms=st['target_rms'], white_frac=white / tot,
                white_rms=st['white_rms'])


def neighbour_weight(o, s0, s1, end):
    """Power fraction of the neighbouring noise state at sample o (0 outside the crossfades)."""
    h = XFADE // 2
    w_old, w_new = S.crossfade_weights(XFADE)
    if s1 < end and o >= s1 - h:                 # fading into the next state
        return float(w_new[o - (s1 - h)] ** 2)
    if s0 > 0 and o < s0 + h:                    # still fading out of the previous state
        return float(w_old[o - (s0 - h)] ** 2)
    return 0.0


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=int, required=True, choices=sorted(RUNS))
    ap.add_argument('--max_partials', type=int, default=None, help='stop after this many partials (testing)')
    args = ap.parse_args()
    run, cfg = args.run, RUNS[args.run]
    sigma = cfg['rms']
    t0 = time.time()
    root = os.path.join(TRUTH, f'run{run}'); [os.makedirs(os.path.join(root, s), exist_ok=True) for s in ('truth', 'templates')]
    raw_dir = os.path.join(STAGE, f'run{run}', 'raw'); os.makedirs(raw_dir, exist_ok=True)

    T, clear, det = make_templates(cfg['energies'], sigma, cfg['tail_sigma'])
    events, noise_trig, bounds = make_schedule(run, cfg['energies'], clear)
    n_part = len(bounds) - 1
    print(f'run {run}: {len(events)} events, {len(noise_trig)} noise windows, {bounds[-1] / 3.6e6:.1f} h, {n_part} partials', flush=True)
    for E in cfg['energies']:
        print(f'  {E:5d} keV: tail below {cfg["tail_sigma"]} sigma {clear[E] / FS:5.1f} s after onset')

    part_of = lambda s: int(np.searchsorted(bounds, s, side='right') - 1)
    ev = np.array([(i, E, o, part_of(o)) for i, (E, o) in enumerate(events)], dtype=np.int64)
    nz = np.array([(j, t, part_of(t)) for j, t in enumerate(noise_trig)], dtype=np.int64)
    # every window must lie inside one partial
    assert all(part_of(o - PRE) == part_of(o + POST - 1) for o in ev[:, 2])
    assert all(part_of(t - PRE) == part_of(t + POST - 1) for t in nz[:, 1])

    NR = NoiseRun(run, cfg, bounds)
    seg_rows = [state_row(d, NR.states[d], cfg, run) for d in range(n_part)]

    # truth tables
    with open(os.path.join(root, 'truth', 'events.csv'), 'w') as f:
        f.write('event_id,energy_keV,onset_sample,onset_time_s,partial,onset_sample_in_partial,window_start_sample\n')
        for i, E, o, d in ev:
            f.write(f'{i},{E},{o},{o / FS:.3f},{d},{o - bounds[d]},{o - PRE}\n')
    with open(os.path.join(root, 'truth', 'noise_windows.csv'), 'w') as f:
        f.write('noise_id,trigger_sample,trigger_time_s,partial,window_start_sample,window_end_sample\n')
        for j, t, d in nz:
            f.write(f'{j},{t},{t / FS:.3f},{d},{t - PRE},{t + POST}\n')
    files = [f'{run:06d}_{cfg["prefix"]}_001_{d:03d}.bin' for d in range(n_part)]
    with open(os.path.join(root, 'truth', 'segments.csv'), 'w', newline='') as f:
        # csv module: the resonances (JSON) and seed columns contain commas and quotes
        keys = list(seg_rows[0]); w = csv.writer(f, lineterminator='\n')
        w.writerow(['partial', 'file', 'start_sample', 'n_samples', 'start_time_s'] + keys)
        for d in range(n_part):
            w.writerow([d, files[d], bounds[d], bounds[d + 1] - bounds[d], f'{bounds[d] / FS:.3f}']
                       + [seg_rows[d][k] for k in keys])
    with h5py.File(os.path.join(root, 'templates', 'templates.h5'), 'w') as f:
        f.attrs.update(f_sample=FS, onset_sample=PRE, detector_seed=1, detector_R0=det['R0'],
                       note='Clean pulse injected at every event, index PRE = onset. Baseline 0 V.')
        for E in cfg['energies']:
            dset = f.create_dataset(f'{E}keV', data=T[E]); dset.attrs.update(energy_kev=E, clear_samples=clear[E])
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJ, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', 'src', 'scripts/of_compare'], cwd=PROJ, text=True).strip())
    except Exception:
        commit, dirty = 'unknown', True
    info = dict(run=run, **{k: v for k, v in cfg.items()}, f_sample=FS, adc_range_v=ADC_RANGE, header_magic=hex(HEADER_MAGIC),
                n_events=len(events), n_per_energy=N_PER_E, n_noise_windows=len(noise_trig), n_samples=bounds[-1],
                duration_h=bounds[-1] / 3.6e6, partials=files, window_pre=PRE, window_post=POST, quiet=QUIET,
                noise_trigger_offset=NOISE_TRIG_OFFSET, noise_every=NOISE_EVERY, jitter=JITTER, xfade=XFADE,
                clear_samples={str(k): v for k, v in clear.items()}, seed_tag=SEED_TAG,
                seeds='SeedSequence([seed_tag, run, s, i]): s=0 schedule, 1 state params (i=partial; stationary i=0), '
                      '2 Gaussian block (i=partial), 3 envelope',
                low_freq_rolloff_hz=S.F_LOW, rms_definition='std inside a 10 s window, mean removed',
                git_commit=commit, git_dirty=dirty, code_sha256=code_hashes())
    json.dump(info, open(os.path.join(root, 'truth', 'run_info.json'), 'w'), indent=1)

    # DDPM files
    H = None
    if cfg['ddpm']:
        counts = {E: int((ev[:, 1] == E).sum()) for E in cfg['energies']}
        H = open_ddpm_files(cfg, run, T, counts)
        row_of = {}
        nxt = {E: 0 for E in cfg['energies']}
        for i, E, o, d in ev:
            row_of[i] = nxt[E]; nxt[E] += 1

    qerr, stats = 0.0, []
    for d in range(n_part if args.max_partials is None else min(n_part, args.max_partials)):
        tp = time.time()
        s0, s1 = bounds[d], bounds[d + 1]
        x = NR.segment(d)
        # pulses (tails continue across partial boundaries)
        L = len(next(iter(T.values())))
        sel = (ev[:, 2] - PRE < s1) & (ev[:, 2] - PRE + L > s0)
        for i, E, o, _ in ev[sel]:
            a, z = o - PRE, o - PRE + L
            lo, hi = max(a, s0), min(z, s1)
            x[lo - s0:hi - s0] += T[int(E)][lo - a:hi - a]
        u = encode(x)
        write_bin(os.path.join(raw_dir, files[d]), u)
        # noise RMS in this partial's quiet windows (noise + residual tails < tail_sigma)
        w, wr = [], []
        for t in nz[nz[:, 2] == d][:, 1]:
            v = decode(u[t - PRE - s0:t + POST - s0])
            qerr = max(qerr, float(np.abs(v - x[t - PRE - s0:t + POST - s0]).max()))
            w.append(v); wr.append(np.std(v))
        wr = np.array(wr)
        mine = ev[ev[:, 3] == d]
        if H is not None:
            for i, E, o, _ in mine:
                win = decode(u[o - PRE - s0:o + POST - s0])
                qerr = max(qerr, float(np.abs(win - x[o - PRE - s0:o + POST - s0]).max()))
                f = H[int(E)]; r = row_of[int(i)]
                f['waveforms_noisy'][r] = win
                nrms = float(np.std(win - T[int(E)][:WIN]))
                f['noise_rms'][r] = nrms
                f['snr_db'][r] = 20 * np.log10((T[int(E)][:WIN].max()) / nrms)
                f['event_id'][r], f['onset_sample'][r], f['segment_id'][r], f['noise_state'][r] = i, o, d, d
                f['t_onset_stream'][r] = o / FS
                xf = neighbour_weight(o, s0, s1, bounds[-1])
                for k, v in noise_param_values(NR.states[d]).items():
                    f['noise_params'][k][r] = v
                f['noise_params']['xfade_weight_neighbour'][r] = xf
        stats.append(dict(partial=d, hours=(s1 - s0) / 3.6e6, events=len(mine), noise_windows=len(w),
                          noise_window_rms_mV=float(wr.mean() * 1e3), noise_window_rms_spread=float(wr.std() / wr.mean()),
                          vmin=float(x.min()), vmax=float(x.max()), seconds=time.time() - tp))
        print(f'  partial {d:3d}: {(s1 - s0) / 3.6e6:5.2f} h, {len(mine):5d} events, quiet-window RMS '
              f'{wr.mean() * 1e3:.4f} mV (spread {wr.std() / wr.mean() * 100:.1f}%), {time.time() - tp:.0f} s', flush=True)
        del x, u
    if H is not None:
        for f in H.values():
            f.close()
    json.dump(dict(partials=stats, max_quantisation_error_v=qerr, wall_s=time.time() - t0),
              open(os.path.join(root, 'truth', 'generation_log.json'), 'w'), indent=1)
    print(f'done in {(time.time() - t0) / 60:.1f} min; max quantisation error {qerr:.2e} V')
    print('ALL_DONE')


if __name__ == '__main__':
    main()

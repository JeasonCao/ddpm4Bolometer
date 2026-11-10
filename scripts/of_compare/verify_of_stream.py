"""Independent checks of a generated OFcompare run (reads only the files on disk).

  1. bin files: header (magic, 1 kHz, 10 V), size, low byte = 0, sample counts vs segments.csv
  2. schedule: 5000 events per energy, spacing / tail rule, windows inside one partial,
     quiet windows free of pulses
  3. data: RMS of quiet windows and of (event window - template) vs. the target RMS
  4. reproducibility: partial 0 regenerated from the seeds is byte-identical to the file
  5. DDPM windows (full-energy runs): each row equals the bin stream slice

Usage: python scripts/of_compare/verify_of_stream.py --run 999996 [--no_regen]
"""
import os, sys, json, argparse, csv
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, 'scripts')); sys.path.insert(0, os.path.dirname(__file__)); os.chdir(PROJ)
import numpy as np, h5py
import generate_of_stream as GEN

PRE, POST = 1500, 8500


def read_bin(path):
    raw = np.fromfile(path, dtype='<u4')
    hdr = raw[:3].tobytes()
    fs, rng_v = np.frombuffer(hdr[4:], '<f4')
    data = raw[3:]
    return hdr, float(fs), float(rng_v), data


def volts(data, R):
    return (data >> np.uint32(8)).astype(np.float64) / 2 ** 23 * R - R      # Octopus RawDataHandlerCupid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, required=True); ap.add_argument('--no_regen', action='store_true')
    a = ap.parse_args()
    root = os.path.join(GEN.TRUTH, f'run{a.run}'); raw_dir = os.path.join(GEN.STAGE, f'run{a.run}', 'raw')
    info = json.load(open(os.path.join(root, 'truth', 'run_info.json')))
    ev = np.loadtxt(os.path.join(root, 'truth', 'events.csv'), delimiter=',', skiprows=1)
    nz = np.loadtxt(os.path.join(root, 'truth', 'noise_windows.csv'), delimiter=',', skiprows=1, ndmin=2)
    seg = list(csv.DictReader(open(os.path.join(root, 'truth', 'segments.csv'))))
    sigma = info['rms']; ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(('  PASS ' if cond else '  FAIL ') + msg)

    print(f'== run {a.run}: {info["level"]}, {info["noise"]}, {len(seg)} partials, {info["duration_h"]:.1f} h')
    # 1. files
    tot, starts = 0, []
    data_by_part = {}
    for s in seg:
        hdr, fs, R, d = read_bin(os.path.join(raw_dir, s['file']))
        good = (hdr[:4] == bytes([0x20, 0x6C, 0, 0]) and fs == 1000.0 and R == 10.0 and len(d) == int(s['n_samples'])
                and not np.any(d & np.uint32(0xFF)))
        check(good, f'{s["file"]}: header 20 6C 00 00 / 1000 Hz / 10 V, {len(d)} samples, low byte 0')
        starts.append(tot); tot += len(d)
        data_by_part[int(s['partial'])] = d
    check(tot == info['n_samples'], f'total samples {tot} == run_info {info["n_samples"]}')
    check(all(int(s['start_sample']) == st for s, st in zip(seg, starts)), 'partial start samples are contiguous')
    bounds = starts + [tot]

    # 2. schedule
    E, on, part = ev[:, 1].astype(int), ev[:, 2].astype(np.int64), ev[:, 4].astype(int)
    cnt = {e: int((E == e).sum()) for e in info['energies']}
    check(all(c == 5000 for c in cnt.values()), f'events per energy: {cnt}')
    check(np.all(np.diff(on) > 0), 'onsets strictly increasing')
    clear = {int(k): v for k, v in info['clear_samples'].items()}
    busy_end = on + np.array([max(8500, clear[e]) for e in E])
    check(np.all(on[1:] - PRE >= busy_end[:-1]), f'next pre-trigger starts after the previous pulse is < {info["tail_sigma"]} sigma')
    pb = lambda x: np.searchsorted(bounds, x, side='right') - 1
    check(np.all(pb(on - PRE) == pb(on + POST - 1)) and np.all(pb(on) == part), 'event windows inside one partial, partial column right')
    nt = nz[:, 1].astype(np.int64)
    k = np.searchsorted(on, nt)                      # first event after each noise trigger
    prev_busy = np.where(k > 0, busy_end[np.maximum(k - 1, 0)], -1)
    next_pre = np.where(k < len(on), on[np.minimum(k, len(on) - 1)] - PRE, tot + 1)
    check(np.all(nt - PRE >= prev_busy) and np.all(nt + POST <= next_pre), f'{len(nt)} quiet windows free of pulses')
    gaps = np.diff(nt) / 1000
    print(f'       quiet-window spacing: median {np.median(gaps):.1f} s, max {gaps.max():.1f} s')

    # 3. RMS
    with h5py.File(os.path.join(root, 'templates', 'templates.h5')) as f:
        T = {e: f[f'{e}keV'][:] for e in info['energies']}
    qr = []
    for t in nt:
        d = pb(t); v = volts(data_by_part[d][t - PRE - bounds[d]:t + POST - bounds[d]], 10.0); qr.append(v.std())
    qr = np.array(qr)
    check(abs(qr.mean() / sigma - 1) < 0.02, f'quiet-window RMS {qr.mean() * 1e3:.4f} mV vs target {sigma * 1e3:.3f} mV (spread {qr.std() / qr.mean() * 100:.1f}%)')
    rr = {e: [] for e in info['energies']}
    for i in np.linspace(0, len(on) - 1, 2000).astype(int):
        d = part[i]; v = volts(data_by_part[d][on[i] - PRE - bounds[d]:on[i] + POST - bounds[d]], 10.0)
        rr[E[i]].append((v - T[E[i]][:PRE + POST]).std())
    for e, r in rr.items():
        check(abs(np.mean(r) / sigma - 1) < 0.03, f'{e:5d} keV: RMS of (window - template) {np.mean(r) * 1e3:.4f} mV')

    # 5. DDPM windows
    if info.get('ddpm'):
        for e in info['energies']:
            with h5py.File(os.path.join(GEN.DDPM_DIR, info['level'], f'eval_efficiency_{e}keV.h5')) as f:
                ids = f['event_id'][:]; n = len(ids)
                sel = np.linspace(0, n - 1, 20).astype(int)
                bad = 0
                for r in sel:
                    i = ids[r]; d = part[i]
                    v = volts(data_by_part[d][on[i] - PRE - bounds[d]:on[i] + POST - bounds[d]], 10.0)
                    bad += not np.array_equal(v, f['waveforms_noisy'][r])
                same_tmpl = np.array_equal(f['clean_template'][:], T[e][:PRE + POST])
                check(n == 5000 and bad == 0 and same_tmpl and np.all(np.diff(ids) > 0),
                      f'DDPM {e} keV file: 5000 rows sorted by event_id, sampled rows equal the bin stream, template ok')

    # 4. reproducibility
    if not a.no_regen:
        cfg = GEN.RUNS[a.run]
        Tg, clear_g, _ = GEN.make_templates(cfg['energies'], cfg['rms'], cfg['tail_sigma'])
        events, noise_trig, bnds = GEN.make_schedule(a.run, cfg['energies'], clear_g)
        check(bnds == bounds and [o for _, o in events] == list(on), 'schedule regenerated from seeds is identical')
        NR = GEN.NoiseRun(a.run, cfg, bnds)
        x = NR.segment(0)
        L = len(Tg[cfg['energies'][0]])
        for e_, o in events:
            if o - PRE < bnds[1] and o - PRE + L > 0:
                lo, hi = max(o - PRE, 0), min(o - PRE + L, bnds[1])
                x[lo:hi] += Tg[e_][lo - (o - PRE):hi - (o - PRE)]
        check(np.array_equal(GEN.encode(x), data_by_part[0]), 'partial 000 regenerated from seeds is byte-identical')
    print('ALL_PASS' if ok else 'SOME_FAILED')


if __name__ == '__main__':
    main()

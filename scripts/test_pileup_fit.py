"""
Generate N random pileup waveforms (clean + noisy) and fit them with fit_pulse,
then save a summary figure of the fits for visual inspection.

Random ranges (uniform):
    E1, E2     : 100 - 3000 keV
    dt = t2-t1 : 0.1 - 2.0 s

Usage (Windows):
    python scripts/test_pileup_fit.py
    python scripts/test_pileup_fit.py --n 10 --output results/pileup_fit_test.png --seed 42
    python scripts/test_pileup_fit.py --tau_constraint 0.3   # relax τ constraint to ±30%
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.pulse.simulator import (
    sample_params, find_equilibrium, is_valid_equilibrium,
    simulate_pileup,
)
from src.noise.generator import generate_noise, sample_noise_params
from src.basics.fit import fit_pulse, biexp_single


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def draw_valid_detector(rng, max_attempts=200):
    """Sample detector params until one with a valid equilibrium is found."""
    for _ in range(max_attempts):
        params = sample_params(rng)
        try:
            eq = find_equilibrium(params)
        except Exception:
            continue
        if is_valid_equilibrium(params, eq):
            return params, eq
    raise RuntimeError(f"No valid detector params after {max_attempts} attempts")


def make_pileup_window(rng, duration, f_sample,
                       E_min=100.0, E_max=3000.0,
                       dt_min=0.5,  dt_max=2.0,
                       t1_margin=0.5):
    """Generate one (clean, noisy, meta) pileup waveform with random E1, E2, dt.

    t1 is placed so that t2 = t1 + dt fits inside the window with some margin.
    """
    E1 = float(rng.uniform(E_min, E_max))
    E2 = float(rng.uniform(E_min, E_max))
    dt = float(rng.uniform(dt_min, dt_max))

    # Place t1 so that t2 has at least 1 s of decay tail before window ends.
    t1_max = duration - dt - 1.0
    t1_max = max(t1_max, t1_margin + 0.01)
    t1 = float(rng.uniform(t1_margin, t1_max))
    t2 = t1 + dt

    params, eq = draw_valid_detector(rng)
    t, v_clean = simulate_pileup(
        [E1, E2], [t1, t2], params, eq,
        duration=duration, f_sample=f_sample,
    )

    noise_params = sample_noise_params(rng)
    noise = generate_noise(rng, noise_params, duration=duration, f_sample=f_sample)

    v_noisy = v_clean + noise
    snr = v_clean.max() / (np.std(noise) + 1e-30)

    meta = dict(E1=E1, E2=E2, dt=dt, t1=t1, t2=t2, snr=float(snr))
    return t, v_clean, v_noisy, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10, help='Number of pileup windows')
    p.add_argument('--output', default='results/pileup_fit_test.png')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--duration', type=float, default=10.0)
    p.add_argument('--f_sample', type=float, default=1000.0)
    p.add_argument('--tau_constraint', type=float, default=0.20,
                   help='Max fractional tau difference between pulses (default 0.20 = ±20%%)')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Generate + fit
    # ------------------------------------------------------------------
    entries = []
    for k in range(args.n):
        print(f"[{k+1}/{args.n}] generating ...", end=' ', flush=True)
        t, v_clean, v_noisy, meta = make_pileup_window(
            rng, args.duration, args.f_sample,
        )
        print(f"E1={meta['E1']:6.0f} keV  E2={meta['E2']:6.0f} keV  "
              f"dt={meta['dt']:.3f} s  SNR={meta['snr']:6.1f}", end=' ', flush=True)

        fr_clean = fit_pulse(v_clean, n_pulses=2, fs=args.f_sample,
                             tau_constraint=args.tau_constraint)
        fr_noisy = fit_pulse(v_noisy, n_pulses=2, fs=args.f_sample,
                             tau_constraint=args.tau_constraint)

        print(f"  clean ok={fr_clean.success}  noisy ok={fr_noisy.success}")

        entries.append(dict(
            t=t, v_clean=v_clean, v_noisy=v_noisy, meta=meta,
            fr_clean=fr_clean, fr_noisy=fr_noisy,
        ))

    # ------------------------------------------------------------------
    # Output paths: inject timestamp before extension
    # ------------------------------------------------------------------
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base, ext = os.path.splitext(args.output)
    out_png  = f"{base}_{ts}{ext}"
    out_npz  = f"{base}_{ts}.npz"

    out_dir = os.path.dirname(os.path.abspath(out_png))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot: rows = windows, cols = (clean full | clean residual | noisy full | noisy residual)
    # ------------------------------------------------------------------
    n_rows = len(entries)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 3.2 * n_rows), squeeze=False)
    col_titles = ['Clean — full signal', 'Clean — residual (P2)',
                  'Noisy — full signal', 'Noisy — residual (P2)']

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight='bold', pad=4)

    for row, e in enumerate(entries):
        t_arr, m = e['t'], e['meta']
        fs_plot = args.f_sample

        for side, (sig, fr) in enumerate([
            (e['v_clean'], e['fr_clean']),
            (e['v_noisy'], e['fr_noisy']),
        ]):
            col_full  = side * 2       # 0 (clean) or 2 (noisy)
            col_resid = side * 2 + 1   # 1 (clean) or 3 (noisy)

            ax_full  = axes[row, col_full]
            ax_resid = axes[row, col_resid]

            # ---- full-signal panel ----
            ax_full.plot(t_arr, sig * 1e3, 'k-', lw=0.5, alpha=0.8, label='data')

            # Shaded regions: seg1 (blue) and seg2 (orange)
            if fr.success and not np.isnan(fr.seg1_end_t):
                ax_full.axvspan(0, fr.seg1_end_t,
                                color='tab:blue', alpha=0.08, label='seg1 (P1 fit)')
                ax_full.axvspan(fr.seg2_start_t, t_arr[-1],
                                color='tab:orange', alpha=0.08, label='seg2 (P2 fit)')
                # Boundary lines
                ax_full.axvline(fr.seg1_end_t,   color='tab:blue',   ls='--', lw=0.8)
                ax_full.axvline(fr.seg2_start_t, color='tab:orange', ls='--', lw=0.8)

            # Pre-fit detected peak times (program's estimate, not ground truth)
            if len(fr.detected_peak_times) >= 2:
                ax_full.axvline(fr.detected_peak_times[0], color='tab:green', ls='--',
                                lw=0.9, alpha=0.9, label='det. peak P1')
                ax_full.axvline(fr.detected_peak_times[1], color='tab:red',   ls='--',
                                lw=0.9, alpha=0.9, label='det. peak P2')

            # Fit overlay
            if fr.success and fr.model_fn is not None:
                fit_curve = fr.model_fn(t_arr, *fr.params)
                ax_full.plot(t_arr, fit_curve * 1e3, color='red', lw=1.0,
                             alpha=0.9, label='fit (total)')
                chi2_str = f"χ²/ndf={fr.chi2_per_ndf:.2e}"
            else:
                chi2_str = f"FAILED: {fr.message[:35]}"

            ax_full.set_ylabel('mV', fontsize=7)
            ax_full.set_xlabel('Time [s]', fontsize=7)
            ax_full.tick_params(labelsize=6)
            ax_full.grid(True, alpha=0.25)
            ax_full.legend(fontsize=5, loc='upper right', ncol=2)

            # param text box
            if len(fr.detected_peak_times) >= 2:
                det_dt = fr.detected_peak_times[1] - fr.detected_peak_times[0]
                det_str = (f"det  pk1={fr.detected_peak_times[0]:.2f}s"
                           f"  pk2={fr.detected_peak_times[1]:.2f}s  dt={det_dt:.2f}s")
            else:
                det_str = "det  (no peaks)"
            param_lines = [
                det_str,
                f"truth  t1={m['t1']:.2f}s  dt={m['dt']:.2f}s",
                f"       E1={m['E1']:.0f}  E2={m['E2']:.0f} keV",
            ]
            if fr.success and len(fr.peak_amps) >= 2:
                param_lines.append(f"fit  B={fr.baseline*1e3:+.2f}mV  {chi2_str}")
                for i, (A, t0, tr, al, td, td2) in enumerate(zip(
                        fr.peak_amps, fr.onset_times, fr.tau_r, fr.alpha,
                        fr.tau_d, fr.tau_d2)):
                    param_lines.append(
                        f"P{i+1}: A={A*1e3:+.1f}mV  t0={t0:.3f}s"
                        f"  τr={tr*1e3:.0f}ms  α={al:.2f}"
                        f"  τd1={td*1e3:.0f}ms  τd2={td2*1e3:.0f}ms"
                    )
            else:
                param_lines.append(chi2_str)
            ax_full.text(0.98, 0.02, '\n'.join(param_lines),
                         transform=ax_full.transAxes, fontsize=5.5,
                         ha='right', va='bottom', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   alpha=0.80, edgecolor='gray', lw=0.3))

            if col_full == 0:
                ax_full.text(-0.14, 0.5, f"#{row+1}",
                             transform=ax_full.transAxes, fontsize=10,
                             rotation=90, va='center', ha='center', fontweight='bold')

            # ---- residual panel ----
            if fr.success and len(fr.pulse1_popt) == 7 and not np.isnan(fr.seg2_start_t):
                # residual = data − pulse-1 model (over full window)
                resid_full = sig - biexp_single(t_arr, *fr.pulse1_popt)

                # mask to seg2 window only
                mask = t_arr >= fr.seg2_start_t
                t_r  = t_arr[mask]
                r_r  = resid_full[mask]

                ax_resid.plot(t_arr, resid_full * 1e3,
                              color='gray', lw=0.4, alpha=0.5, label='residual (full)')
                ax_resid.plot(t_r, r_r * 1e3,
                              'k-', lw=0.6, alpha=0.9, label='residual (seg2)')

                # pulse-2 fit: biexp_single with P2 params, B ≈ 0 (B2 small)
                # params[7:13] = [A2, t0_2, tau_r2, alpha2, tau_d1_2, tau_d2_2]
                p2 = fr.params
                p2_single = np.array([p2[0] - fr.pulse1_popt[0],  # B2 ≈ B_combined - B1
                                      p2[7], p2[8], p2[9], p2[10], p2[11], p2[12]])
                fit_r = biexp_single(t_arr, *p2_single)
                ax_resid.plot(t_arr, fit_r * 1e3, color='red', lw=0.9, label='P2 fit')

                # seg2 boundary and detected peak
                ax_resid.axvline(fr.seg2_start_t, color='tab:orange', ls='--', lw=0.8)
                if len(fr.detected_peak_times) >= 2:
                    ax_resid.axvline(fr.detected_peak_times[1], color='tab:red',
                                     ls='--', lw=0.9, alpha=0.9, label='det. peak P2')
                ax_resid.axhline(0, color='k', lw=0.4, alpha=0.4)
                ax_resid.set_ylabel('mV', fontsize=7)
                ax_resid.set_xlabel('Time [s]', fontsize=7)
                ax_resid.tick_params(labelsize=6)
                ax_resid.grid(True, alpha=0.25)
                ax_resid.legend(fontsize=5, loc='upper right')
            else:
                ax_resid.text(0.5, 0.5, 'fit failed\nno residual',
                              transform=ax_resid.transAxes,
                              ha='center', va='center', fontsize=8, color='red')
                ax_resid.set_axis_off()

    fig.suptitle(
        f"Pileup fit test — {args.n} windows, "
        f"E∈[100,3000] keV, Δt∈[0.1,2.0]s  |  {ts}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure: {out_png}")

    # Optional: dump waveforms + truth for later inspection
    np.savez(
        out_npz,
        t=entries[0]['t'],
        v_clean=np.stack([e['v_clean'] for e in entries]),
        v_noisy=np.stack([e['v_noisy'] for e in entries]),
        E1=np.array([e['meta']['E1'] for e in entries]),
        E2=np.array([e['meta']['E2'] for e in entries]),
        dt=np.array([e['meta']['dt'] for e in entries]),
        t1=np.array([e['meta']['t1'] for e in entries]),
        t2=np.array([e['meta']['t2'] for e in entries]),
        f_sample=args.f_sample,
    )
    print(f"Saved waveforms:  {out_npz}")


if __name__ == '__main__':
    main()
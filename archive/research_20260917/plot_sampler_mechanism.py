"""
2026-10-08: why DDIM eta=0 and --no_noise differ, as a figure.

Both samplers are fully deterministic (neither draws a random number inside
the loop). The difference is arithmetic: DDIM recomputes each step for the
no-noise case and so keeps x_t on the marginal q(x_t|x_0); --no_noise just
deletes DDPM's stochastic term, leaving a step that was calibrated assuming
that term would be added back, so it overshoots and the trajectory runs a
variance deficit that compounds.

The consequence is what the figure shows: the random x_T is pure variance
with no signal in it, so DDIM carries it to the output while --no_noise
squeezes it out.
"""
import os, sys, importlib.util
PROJ = "/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#"
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, h5py, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = "/tmp/claude-1000/-home-wsl-0vbb/4159fbbb-99dd-4ed5-b7cf-493a0436b6d5/scratchpad"
spec = importlib.util.spec_from_file_location("nn2", f"{SP}/nonoise_2x2.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

C_DDIM, C_NN, C_REF = '#2a78d6', '#eb6834', '#8a8780'
N_EV, BATCH = 16, 16


@torch.no_grad()
def trajectory(diffusion, x_tilde, rule, seed):
    """Returns (T+1, B, L): x_t after every transition, index 0 = x_T."""
    B, _, L = x_tilde.shape
    dev = x_tilde.device
    g = torch.Generator(device=dev); g.manual_seed(seed)
    x_t = torch.randn(B, 1, L, device=dev, generator=g)
    sch, traj = diffusion.schedule, [x_t.squeeze(1).float().cpu().numpy().copy()]
    order = (list(range(diffusion.T - 1, -1, -1)) if rule == 'ddim'
             else list(reversed(range(diffusion.T))))
    for i, t_cur in enumerate(order):
        cond = torch.full((B,), t_cur + 1, device=dev, dtype=torch.long)
        with torch.autocast(device_type=dev.type, dtype=torch.float16,
                            enabled=dev.type == 'cuda'):
            eps = diffusion._predict_noise(x_t, x_tilde, cond, None)
        if rule == 'ddim':
            x0 = (x_t - sch.sqrt_one_minus_alpha_bar[t_cur] * eps) / sch.sqrt_alpha_bar[t_cur]
            if i < len(order) - 1:
                tn = order[i + 1]
                x_t = sch.sqrt_alpha_bar[tn] * x0 + sch.sqrt_one_minus_alpha_bar[tn] * eps
            else:
                x_t = x0
        else:
            coeff = sch.beta[t_cur] / sch.sqrt_one_minus_alpha_bar[t_cur]
            x_t = (1.0 / sch.sqrt_alpha[t_cur]) * (x_t - coeff * eps)
        traj.append(x_t.squeeze(1).float().cpu().numpy().copy())
    return np.stack(traj)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = m.load_model(m.MODEL, dev)
    with h5py.File(os.path.join(m.EVAL, 'eval_efficiency_2615keV.h5')) as f:
        noisy = f['waveforms_noisy'][:N_EV].astype(np.float32)
        tmpl = f['clean_template'][:].astype(np.float32)
    sc = np.abs(noisy).max(axis=1, keepdims=True)
    x_tilde = torch.from_numpy(noisy / sc).unsqueeze(1).to(dev)
    x0_norm = tmpl[None, :] / sc                      # true clean signal, normalised

    tr = {}
    for rule in ('ddim', 'nonoise'):
        for seed in (1, 2):
            tr[(rule, seed)] = trajectory(diffusion, x_tilde, rule, seed)
            print(f'  captured {rule} seed={seed}', flush=True)

    T = diffusion.T
    steps = np.arange(T + 1)                          # 0 = x_T, T = x_0
    ab = diffusion.schedule.alpha_bar.cpu().numpy()
    # Marginal-preserving prediction: var(x_t) = ab_t*var(x0) + (1-ab_t)
    v0 = float(np.var(x0_norm))
    ab_at = np.concatenate([[ab[T - 1]], ab[::-1][1:], [1.0]])
    pred = np.sqrt(ab_at * v0 + (1.0 - ab_at))

    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.45, 1, 1], hspace=0.55, wspace=0.32)
    a = fig.add_subplot(gs[0, :3])
    b = fig.add_subplot(gs[0, 3:])

    # (a) variance budget
    a.plot(steps, pred, color=C_REF, ls='--', lw=2,
           label='Marginal-preserving target')
    for rule, c, lab in (('ddim', C_DDIM, 'DDIM η=0'), ('nonoise', C_NN, '--no_noise')):
        sd = tr[(rule, 1)].std(axis=(1, 2))
        a.plot(steps, sd, color=c, lw=2, label=lab)
    a.set_title('(a)  Noise budget along the trajectory', fontsize=11, loc='left')
    a.set_xlabel('Reverse step  (0 = start from $x_T$,  50 = output)')
    a.set_ylabel('std of $x_t$')
    a.legend(fontsize=8.5, frameon=False)
    a.grid(alpha=0.25)

    # (b) how much of x_T survives
    for rule, c, lab in (('ddim', C_DDIM, 'DDIM η=0'), ('nonoise', C_NN, '--no_noise')):
        d = np.sqrt(((tr[(rule, 1)] - tr[(rule, 2)]) ** 2).mean(axis=(1, 2)))
        b.semilogy(steps, d, color=c, lw=2, label=lab)
    b.set_title('(b)  Memory of the random start\n(two seeds, same input)',
                fontsize=11, loc='left')
    b.set_xlabel('Reverse step')
    b.set_ylabel('RMS difference between seeds')
    b.legend(fontsize=8.5, frameon=False)
    b.grid(alpha=0.25, which='both')

    # (c,d) snapshots: one panel per step, shared y within a column so the two
    # samplers are directly comparable at equal step.
    SHOW = [0, 10, 25, 40, 50]
    t = np.arange(noisy.shape[1]) / 1000.0
    for col, k in enumerate(SHOW):
        lim = 3.2 * max(tr[('ddim', 1)][k, 0].std(), tr[('nonoise', 1)][k, 0].std())
        for row, (rule, c) in enumerate((('ddim', C_DDIM), ('nonoise', C_NN))):
            p = fig.add_subplot(gs[1 + row, col])
            p.plot(t, tr[(rule, 1)][k, 0], lw=0.6, color=c)
            p.set_ylim(-lim, lim)
            p.set_xticks([0, 5, 10])
            p.tick_params(labelsize=7.5)
            if row == 0:
                p.set_title(f'step {k}', fontsize=9.5)
            if col == 0:
                p.set_ylabel('DDIM η=0' if row == 0 else '--no_noise',
                             fontsize=9.5, color=c, fontweight='medium')
            else:
                p.tick_params(labelleft=False)
            if row == 1:
                p.set_xlabel('t (s)', fontsize=8)
            else:
                p.tick_params(labelbottom=False)
            p.grid(alpha=0.2)
    fig.text(0.5, 0.593, '(c)  $x_t$ at five steps, one event — y scale shared '
             'within each column', fontsize=11, ha='center')

    out = os.path.join(PROJ, 'results', 'plots', 'paper_zeroinit',
                       'sampler_mechanism_ddim_vs_nonoise.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved -> {out}')

    print('\nstd(x_t) at the output:  '
          f"DDIM {tr[('ddim',1)][-1].std():.4f}   "
          f"no_noise {tr[('nonoise',1)][-1].std():.4f}   target {pred[-1]:.4f}")
    for rule in ('ddim', 'nonoise'):
        d = np.sqrt(((tr[(rule,1)] - tr[(rule,2)]) ** 2).mean(axis=(1,2)))
        print(f'  {rule:8s} seed divergence: start {d[0]:.3f} -> output {d[-1]:.3e} '
              f'({d[-1]/d[0]:.1e} of initial)')
    print('MECHANISM_FIG_DONE')


if __name__ == '__main__':
    main()

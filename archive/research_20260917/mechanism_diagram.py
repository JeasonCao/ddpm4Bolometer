"""
2026-09-17: one summary figure for the group-meeting slides -- the causal
chain, drawn rather than described. White background, screenshot-friendly.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = '/tmp/claude-1000/-home-wsl-0vbb/946b7f43-5893-49fc-8189-f0872f2eda18/scratchpad'

NOISE = '#5b7fa6'
FLOOR = '#c0562c'
GOOD = '#2f6b57'
GREY = '#8b9490'


def diagram():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis('off')
    fig.patch.set_facecolor('white')

    boxes = [
        (0.15, 'Sampler invents\n$x_T \\sim N(0,I)$', 'unit-variance random field\nnothing to do with the data', NOISE),
        (3.35, 'Network must\ncancel it', 'it manages\n$\\sim$0.1% relative accuracy', GREY),
        (6.55, 'Leftover is fixed\nin NORMALIZED space', '$5.6\\times10^{-4}$, same at every energy', FLOOR),
        (9.75, 'Multiply back by\nscale $\\propto$ energy', 'physical error $\\propto E$\n$\\Rightarrow$ FWHM/E $\\approx$ 0.105%', FLOOR),
    ]
    for x, title, sub, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.15), 3.0, 1.5, boxstyle='round,pad=0.08',
                                    facecolor=col, alpha=0.11, edgecolor=col, linewidth=1.6))
        ax.text(x + 1.5, 2.22, title, ha='center', va='center', fontsize=12.5,
                fontweight='bold', color=col)
        ax.text(x + 1.5, 1.58, sub, ha='center', va='center', fontsize=9.4, color='#333')
    for x in (3.18, 6.38, 9.58):
        ax.add_patch(FancyArrowPatch((x - 0.02, 1.9), (x + 0.16, 1.9),
                                     arrowstyle='-|>', mutation_scale=17, color='#666', linewidth=1.6))

    ax.text(6.5, 0.52, 'The measurement noise we actually wanted to remove is only 1.31 mV — '
                       'at 2615 keV that is SMALLER than the residue the sampler leaves behind.',
            ha='center', va='center', fontsize=10.5, color=FLOOR, style='italic')
    ax.text(6.5, 3.25, 'Where the post-denoising noise floor comes from',
            ha='center', va='center', fontsize=14.5, fontweight='bold', color='#1c2220')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_chain.png'), dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def crossover():
    """The race: normalized input noise (falling) vs the fixed floor."""
    E = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 239, 511, 727, 1461, 2615, 3034])
    sig = np.array([.269023, .239153, .203103, .149673, .087484, .061451, .047291, .038423,
                    .03235, .024574, .019806, .004236, .001989, .001401, .0007, .000394, .00034])
    cold = np.array([.007453, .006143, .004546, .003061, .001825, .001386, .001165, .001028,
                     .000941, .000838, .000776, .0006, .000579, .000573, .000567, .000564, .000564])
    cold5 = np.array([.005091, .003521, .002704, .001885, .001121, .000831, .000682, .000587,
                      .000544, .000468, .000422, .00024, .000208, .000199, .000186, .00018, .000179])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    ax.loglog(E, sig, 'o-', color=NOISE, lw=2.4, ms=5, label='Real noise, normalized (what the net sees)')
    ax.loglog(E, cold, 'o-', color=FLOOR, lw=2.4, ms=5, label='Network error floor - cold (production)')
    ax.loglog(E, cold5, 'o-', color=GOOD, lw=1.8, ms=4, label='Network error floor - cold5shot')

    # crossover shading
    ax.axvspan(1400, 4000, color=FLOOR, alpha=0.07)
    ax.text(2100, 0.12, 'denoising\nHURTS', ha='center', fontsize=11, color=FLOOR, fontweight='bold')
    ax.text(60, 0.12, 'denoising helps', ha='center', fontsize=11, color=GOOD, fontweight='bold')

    ax.set_xlabel('True energy (keV)', fontsize=11.5)
    ax.set_ylabel('Amplitude in normalized units', fontsize=11.5)
    ax.set_title('The floor never moves - the data drops below it\n(1.382 mV noise)', fontsize=12.5, fontweight='bold')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=9.5, loc='lower left')
    ax.set_xlim(0.8, 4200)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_crossover.png'), dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def decomposition():
    """Total residual vs the part that actually costs resolution."""
    meth = ['no denoise', 'cold\n(production)', 'cold5shot', 'eta1_1shot', 'zero-init']
    tot = np.array([1.38e-3, 2.44e-3, 7.50e-4, 9.16e-4, 9.01e-4]) * 1e3   # mV
    std_a = np.array([0.0405, 0.400, 0.157, 0.365, 0.0219])                # keV
    common = np.array([0.3, 11.7, 85.5, 55.7, 93.7])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    fig.patch.set_facecolor('white')
    x = np.arange(len(meth))
    cols = [GREY, FLOOR, GOOD, '#d08c3a', GOOD]

    axes[0].bar(x, tot, color=cols, alpha=.55, edgecolor='none')
    for i, v in enumerate(tot):
        axes[0].text(i, v * 1.03, f'{v:.2f}', ha='center', fontsize=9.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(meth, fontsize=9.5)
    axes[0].set_ylabel('Total residual rms (mV)', fontsize=11)
    axes[0].set_title('Total residual: zero-init $\\approx$ eta1_1shot', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=.25, axis='y')
    for i, c in enumerate(common):
        axes[0].text(i, tot[i] * 0.45, f'common\n{c:.0f}%', ha='center', fontsize=9,
                     color='white' if c > 50 else '#333', fontweight='bold')

    axes[1].bar(x, std_a, color=cols, alpha=.85, edgecolor='none')
    for i, v in enumerate(std_a):
        axes[1].text(i, v * 1.04 + 0.006, f'{v:.3f}', ha='center', fontsize=9.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(meth, fontsize=9.5)
    axes[1].set_ylabel('Event-to-event amplitude scatter std(a)  (keV)', fontsize=11)
    axes[1].set_title('The part that sets resolution: 17x apart', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=.25, axis='y')

    fig.suptitle('2615 keV - same total residual, but the common-mode part is absorbed by calibration',
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_decomposition.png'), dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def final_fwhm():
    E = np.array([15, 239, 511, 1461, 2615, 3034])
    noisy = np.array([0.918, 0.940, 0.933, 0.935, 0.922, 0.913])
    cold = np.array([0.098, 0.279, 0.496, 1.461, 2.519, 2.793])
    cold5 = np.array([0.077, 0.155, 0.262, 0.601, 1.091, 1.165])
    warm5 = np.array([0.593, 0.163, 0.261, 0.578, 0.888, 0.969])
    zero = np.array([0.071, 0.146, 0.205, 0.340, 0.459, 0.441])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    ax.plot(E, noisy, 's--', color=GREY, lw=1.8, ms=5, label='no denoising')
    ax.plot(E, cold, 'o-', color=FLOOR, lw=2.2, ms=5, label='cold (production, 50 evals)')
    ax.plot(E, cold5, 'o-', color='#8a6fb0', lw=1.8, ms=4.5, label='cold5shot (250 evals)')
    ax.plot(E, warm5, 'o-', color='#d08c3a', lw=1.8, ms=4.5, label='warm5shot (5-30 evals, energy-dependent)')
    ax.plot(E, zero, 'o-', color=GOOD, lw=2.8, ms=6, label='zero-init (50 evals)')
    ax.set_xscale('log')
    ax.set_xlabel('True energy (keV)', fontsize=11.5)
    ax.set_ylabel('Energy resolution FWHM (keV)', fontsize=11.5)
    ax.set_title('Real tri-exp fits, 500 identical events per point\nzero-init wins at every energy, and always beats no-denoising', fontsize=12.5, fontweight='bold')
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9.5)
    ax.set_ylim(0, 3.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_final_fwhm.png'), dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    diagram(); crossover(); decomposition(); final_fwhm()
    print('figures written to', OUT)

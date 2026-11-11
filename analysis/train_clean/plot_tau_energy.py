"""Time constants against true deposited energy, exact blue-only selection."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from analysis.train_clean.plot_after_fit import DEST, TAUS, read_all, quality, draw_panel, save, atomic_json, now, np, plt


def main():
    _,a,fingerprint=read_all(False)
    good,flags,_,_=quality(a)
    mask=(a['n_pulses_fit']==1)&good&~flags
    with np.load(DEST/'blue_only/selected_windows.npz') as old:
        np.testing.assert_array_equal(old['source_shard'],a['shard'][mask])
        np.testing.assert_array_equal(old['window_idx'],a['window_idx'][mask])
    energy=a['energies_1'][mask]
    assert np.all(np.isfinite(energy)) and np.all(energy>0)
    dest=DEST/'blue_only/energy'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        raise FileExistsError('Energy scatter outputs already complete')
    count=int(mask.sum())
    for log_x in [False,True]:
        fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
        for ax,(tau,scale,label) in zip(axes,TAUS):
            draw_panel(ax,energy,a[tau][mask,0]*scale,flags[mask],
                       'True deposited energy (keV)',label)
            if log_x:
                ax.set_xscale('log')
        fig.suptitle(f'Selected single events | N={count:,} | all points, no downsampling\n'
                     +('Logarithmic energy axis' if log_x else 'Linear energy axis'),fontsize=12)
        save(fig,dest/('time_constants_vs_energy'+('_logx' if log_x else '')))
    report=dict(created_at=now(),source_fingerprint=fingerprint,selected=count,
                x='energies_1: true deposited energy, keV',
                y=['t_rise (ms)','t_fall1 (s)','t_fall2 (s)'],
                selection='Exact prior blue_only source_shard/window_idx selection',
                energy_min_kev=float(energy.min()),energy_max_kev=float(energy.max()),
                no_downsampling=True)
    atomic_json(dest/'COMPLETE.json',report)
    (dest/'README.md').write_text(
        '# 波形时间常数 vs 真实能量\n\n'
        f'严格沿用61225个蓝色筛选单事件，每面板每窗口一个点，不抽样。'
        '横轴为源HDF5 energies_1的真实沉积能量（keV），非拟合幅度或反推能量。'
        '三个纵轴依次为上升时间（ms）、快下降时间（s）、慢下降时间（s）。\n\n'
        '提供线性能量轴与对数能量轴两套PNG/PDF，点集合完全相同。'
        '未控制T_base及其他模拟参数；散点关系不能直接解释为固定探测器参数下的纯能量效应。'
        '原始clean、fit文件及原图不变。\n',encoding='utf-8')
    print(report,flush=True)


if __name__=='__main__':
    main()

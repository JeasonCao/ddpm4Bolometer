"""Wait for the authorized fit run; assess pileup and plot every successful single."""

# Repository entry bootstrap; no data or model is loaded here.
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_ROOT = _RepoPath(__file__).resolve().parents[2]
_repo_sys.path.insert(0, str(_REPO_ROOT))
_repo_sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from pathlib import Path
import os
import sys

from experiments.train_clean_20261107.paths import ROOT, PLOTS, diary_path, acquire_lock
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'

import argparse
import ctypes
import json
import time
import traceback
import hashlib
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.train_clean_20261107.fit_train_clean import NOTES,OUTPUT,load_fitter,atomic_json,now

DEST=PLOTS
PARAMS=[('R0',1,r'$R_0$ ($\Omega$)'),('T0',1,r'$T_0$ (K)'),
        ('lambda0',1e9,r'$\lambda_0$ ($10^{-9}$ m K$^{0.5}$)'),('q',1,r'$q$'),
        ('g_ec',1,r'$g_{ec}$ (W K$^{-a_{ec}}$)'),('a_ec',1,r'$a_{ec}$'),
        ('C_p',1e12,r'$C_p$ (pF)'),('T_base',1e3,r'$T_{base}$ (mK)')]
TAUS=[('t_rise',1e3,r'$\tau_r$ (ms)'),('t_fall1',1,r'$\tau_{d1}$ (s), fast'),
      ('t_fall2',1,r'$\tau_{d2}$ (s), slow')]
FIELDS=['window_idx','n_pulses_fit','success','relative_rms','onset_error','tau_boundary',
        'decay_degenerate','t_rise','t_fall1','t_fall2','onsets_1','onsets_2','energies_1','energies_2']


def record(state,**kwargs):
    atomic_json(ROOT/'plot_status.json',dict(time=now(),pid=os.getpid(),state=state,**kwargs))
    print(now(),state,kwargs,flush=True)


def diary(text):
    diary_file=diary_path()
    diary_file.parent.mkdir(parents=True,exist_ok=True)
    with diary_file.open('a',encoding='utf-8') as f:
        f.write('\n- ['+now()+'] clean拟合后处理：'+text+'\n')


def read_all(pilot=False):
    folder=OUTPUT/'pilot' if pilot else OUTPUT
    files=sorted(folder.glob('fit_clean_*.h5'))
    assert len(files)==10, 'Expected ten complete fit shards'
    arrays={k:[] for k in FIELDS}
    params={k:[] for k,_,_ in PARAMS}
    shards=[];rows=[]
    fingerprint=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))['fingerprint']
    for shard,path in enumerate(files):
        with h5py.File(path,'r') as f:
            assert bool(f.attrs['complete']) and np.all(f['done'][:])
            assert f.attrs['fingerprint']==fingerprint
            n=len(f['window_idx'])
            for k in arrays:
                arrays[k].append(f[k][:])
            for k in params:
                params[k].append(f['params'][k][:])
            shards.extend([shard]*n)
            rows.extend(range(n))
    a={k:np.concatenate(v) for k,v in arrays.items()}
    a['params']={k:np.concatenate(v) for k,v in params.items()}
    a['shard']=np.asarray(shards,dtype=np.int16)
    a['fit_row']=np.asarray(rows,dtype=np.int64)
    if not pilot:
        assert len(a['success'])==90500
    return files,a,fingerprint


def quality(a):
    finite=np.isfinite(a['relative_rms'])
    for key,_,_ in TAUS:
        finite &= np.isfinite(a[key][:,0]) & (a[key][:,0]>0)
        second=a['n_pulses_fit']==2
        finite[second] &= np.isfinite(a[key][second,1]) & (a[key][second,1]>0)
    good=a['success'] & finite
    onset=np.abs(a['onset_error'])
    error=np.where(a['n_pulses_fit']==1,onset[:,0],np.max(onset,axis=1))
    # Diagnostics, not a calibrated physical-quality or statistical-confidence test.
    flagged=(a['tau_boundary']|a['decay_degenerate']|(a['relative_rms']>.01)|(error>.05)|~np.isfinite(error))
    report={}
    for name,n in [('single',1),('pileup',2)]:
        mask=a['n_pulses_fit']==n
        valid=mask & good
        count=int(valid.sum())
        rel=a['relative_rms'][valid]
        er=error[valid]
        report[name]=dict(total=int(mask.sum()),success_finite=count,
            success_finite_fraction=float(count/max(mask.sum(),1)),
            tau_boundary_fraction=float(a['tau_boundary'][valid].mean()) if count else None,
            decay_degenerate_fraction=float(a['decay_degenerate'][valid].mean()) if count else None,
            relative_rms_median=float(np.median(rel)) if count else None,
            relative_rms_p95=float(np.quantile(rel,.95)) if count else None,
            onset_error_p95_seconds=float(np.quantile(er,.95)) if count and np.all(np.isfinite(er)) else None,
            onset_error_over_50ms_fraction=float(np.mean((er>.05)|~np.isfinite(er))) if count else None,
            diagnostic_flag_fraction=float(flagged[valid].mean()) if count else None)
    p=report['pileup']
    reliable=(p['success_finite_fraction']>=.95 and
              p['diagnostic_flag_fraction'] is not None and p['diagnostic_flag_fraction']<=.10)
    report.update(pileup_reliable_for_separate_parameter_plots=bool(reliable),
        decision='single main figures; separate P1/P2 supplement' if reliable else 'single-event parameter figures only; pileup excluded',
        diagnostic_criteria=dict(relative_rms_above=.01,onset_abs_error_seconds_above=.05,
                                tau_boundary=True,near_equal_decays=True,
                                pileup_allowed_flag_fraction=.10,pileup_required_success_fraction=.95),
        warning='Fit time constants describe an approximate waveform model, not uniquely identified ODE thermal eigenmodes. Marginal scatter does not control energy or parameter correlations.')
    return good,flagged,error,report


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=180,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),dpi=180,facecolor='white')
    plt.close(fig)


def draw_panel(ax,x,y,flag,xlabel,ylabel,small=False):
    regular=~flag
    ax.scatter(x[regular],y[regular],s=2 if small else 3,alpha=.12,c='#2367a4',
               linewidths=0,rasterized=True)
    ax.scatter(x[flag],y[flag],s=5 if small else 8,alpha=.45,c='#d25830',
               linewidths=0,rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8 if small else 9)


def scatter_figures(a,mask,flags,dest,prefix,pulse=0):
    count=int(mask.sum())
    assert count>0
    # Every selected window is plotted, with no downsampling or energy filtering.
    fig,axes=plt.subplots(3,8,figsize=(25,10),layout='constrained')
    for col,(key,scale,xlabel) in enumerate(PARAMS):
        x=a['params'][key][mask]*scale
        assert np.all(np.isfinite(x))
        for row,(tau,yscale,ylabel) in enumerate(TAUS):
            draw_panel(axes[row,col],x,a[tau][mask,pulse]*yscale,flags[mask],xlabel,ylabel,True)
    fig.suptitle(f'{prefix}: {count:,} windows, one point per window per panel\n'
                 'Blue: no diagnostic flag | Orange: boundary / degeneracy / residual / onset flag (retained)',fontsize=14)
    save(fig,dest/(prefix+'_overview'))
    for key,scale,xlabel in PARAMS:
        fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
        for ax,(tau,yscale,ylabel) in zip(axes,TAUS):
            draw_panel(ax,a['params'][key][mask]*scale,a[tau][mask,pulse]*yscale,
                       flags[mask],xlabel,ylabel)
        fig.suptitle(f'{prefix} | N={count:,} | orange: diagnostic flag, retained',fontsize=12)
        save(fig,dest/(prefix+'_vs_'+key))


def qa_figures(files,a,good,flags,error,dest):
    single=(a['n_pulses_fit']==1)&good
    pile=(a['n_pulses_fit']==2)&good
    gap=a['onsets_2']-a['onsets_1']
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    bins=np.logspace(-7,0,65)
    for mask,label,color in [(single,'single','#2367a4'),(pile,'pileup','#d25830')]:
        axes[0,0].hist(np.maximum(a['relative_rms'][mask],1e-12),bins=bins,
                       histtype='step',density=True,label=label,color=color)
    axes[0,0].set(xscale='log',yscale='log',xlabel='Full-window relative residual RMS',ylabel='Density')
    axes[0,0].legend()
    for j,color in [(0,'#2367a4'),(1,'#d25830')]:
        axes[0,1].scatter(gap[pile],np.maximum(np.abs(a['onset_error'][pile,j]),1e-6),
                          s=3,alpha=.12,color=color,label=f'pulse {j+1}',rasterized=True)
    axes[0,1].axhline(.05,color='black',ls='--',lw=1)
    axes[0,1].set(yscale='log',xlabel='True pulse separation (s)',ylabel='Absolute fitted onset error (s)')
    axes[0,1].legend()
    edges=np.linspace(.01,2.,16)
    centers=(edges[1:]+edges[:-1])/2
    for field,label in [(a['tau_boundary'],'tau boundary'),(flags,'any diagnostic flag')]:
        rates=[]
        for left,right in zip(edges[:-1],edges[1:]):
            mask=pile&(gap>=left)&(gap<=right if right==edges[-1] else gap<right)
            rates.append(float(field[mask].mean()) if mask.any() else np.nan)
        axes[1,0].plot(centers,rates,'o-',label=label)
    axes[1,0].set(xlabel='True pulse separation (s)',ylabel='Fraction of successful pileup fits',ylim=(0,1))
    axes[1,0].legend()
    axes[1,1].scatter(a['energies_2'][pile]/a['energies_1'][pile],
                      np.maximum(a['relative_rms'][pile],1e-12),s=3,alpha=.12,rasterized=True)
    axes[1,1].set(xscale='log',yscale='log',xlabel='True E2 / E1',ylabel='Pileup relative residual RMS')
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle('Pileup diagnostics: solver success alone does not establish reliable decomposition')
    save(fig,dest/'pileup_quality')
    candidates=np.flatnonzero(pile)
    if not len(candidates):
        return
    ranked=candidates[np.argsort(a['relative_rms'][candidates])]
    selected=[int(ranked[int(q*(len(ranked)-1))]) for q in [.1,.5,.9,.99]]
    selected += [int(candidates[np.argmax(error[candidates])]),int(ranked[-1])]
    selected=list(dict.fromkeys(selected))
    mod=load_fitter()
    fig,axes=plt.subplots(len(selected),2,figsize=(14,3*len(selected)),layout='constrained')
    examples=[]
    for axpair,i in zip(axes,selected):
        file=files[a['shard'][i]]
        row=int(a['fit_row'][i]);idx=int(a['window_idx'][i])
        with h5py.File(file,'r') as fit:
            source=fit.attrs['source_path']
            p=fit['raw_params'][row,:int(fit['n_params'][row])]
            fs=float(fit.attrs['f_sample'])
        with h5py.File(source,'r') as src:
            wave=src['waveforms'][idx]
        t=np.arange(len(wave))/fs
        model=mod.biexp_double_free(t,*p)
        axpair[0].plot(t,wave*1e3,label='clean',lw=1)
        axpair[0].plot(t,model*1e3,label='fit',ls='--',lw=1)
        axpair[0].set(xlabel='Time (s)',ylabel='Voltage (mV)',
                      title=f'{Path(source).name}, row {idx}; gap={gap[i]:.3f}s; flagged={bool(flags[i])}')
        axpair[0].legend()
        axpair[1].plot(t,(wave-model)*1e3,lw=1,color='#d25830')
        axpair[1].set(xlabel='Time (s)',ylabel='Residual (mV)',
                      title=f'Rel.RMS={a["relative_rms"][i]:.3g}; max onset error={error[i]:.3g}s')
        examples.append(dict(source=str(source),window_idx=idx,relative_rms=float(a['relative_rms'][i]),
                             max_onset_error_seconds=float(error[i])))
    save(fig,dest/'pileup_example_waveforms')
    atomic_json(dest/'pileup_examples.json',examples)


def render(pilot=False):
    files,a,fingerprint=read_all(pilot)
    good,flags,error,report=quality(a)
    dest=ROOT/'plot_smoke' if pilot else DEST
    dest.mkdir(parents=True,exist_ok=True)
    if not pilot:
        assert not (dest/'COMPLETE.json').exists(),'Completed plot run exists; refuse overwrite'
    record('plotting_smoke' if pilot else 'plotting',directory=str(dest))
    mask=(a['n_pulses_fit']==1)&good
    scatter_figures(a,mask,flags,dest,'single')
    if report['pileup_reliable_for_separate_parameter_plots']:
        for pulse in [0,1]:
            scatter_figures(a,(a['n_pulses_fit']==2)&good,flags,dest,f'pileup_pulse{pulse+1}',pulse)
    qa_figures(files,a,good,flags,error,dest)
    np.savez_compressed(dest/'plot_index.npz',source_shard=a['shard'],window_idx=a['window_idx'],
                        single_plotted=mask,diagnostic_flag=flags,success_finite=good)
    report.update(created_at=now(),source_fingerprint=fingerprint,pilot_only=pilot,
                  single_plotted=int(mask.sum()),no_downsampling=True,
                  matplotlib_version=matplotlib.__version__,
                  plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files=[str(p) for p in files])
    atomic_json(dest/'quality_report.json',report)
    p=report['pileup']
    (dest/'README.md').write_text(
        '# Clean拟合时间常数与ODE参数\n\n'
        +('**这是500窗绘图接口测试，不是全量结果。**\n\n' if pilot else '')
        +f'主图含 {int(mask.sum()):,} 个成功且时间常数有限的单事件窗口，不抽样、不按能量删点。每面板每窗口一个点。\n\n'
        +'`single_overview.png/pdf` 是3行×8列总览；`single_vs_参数.png/pdf` 每张有三个时间常数面板。'
        +'横轴为原HDF5实际随机参数；上升时间为ms，快/慢下降时间为s。\n\n'
        +'蓝色为未触发诊断标记，橙色为贴边、近等衰减、相对残差>1%或起点偏差>50ms，仍保留在图中。'
        +'拟合失败或非有限参数不能绘点，数量见quality_report.json。\n\n'
        +f'Pileup成功且参数有限：{p["success_finite"]}/{p["total"]}；tau贴边比例={p["tau_boundary_fraction"]}；'
        +f'任一诊断标记比例={p["diagnostic_flag_fraction"]}。决定：{report["decision"]}。\n\n'
        +'Pileup若通过保守诊断，只输出独立的pulse1/pulse2补充图，不把两个脉冲混成单窗单参数。'
        +'不通过则不输出pileup参数关系图，仅保留质量诊断与代表波形。\n\n'
        +'时间常数为三指数近似模型参数，不能直接等同ODE物理热时间常数；散点是边际关系，未控制能量或参数间相关性。'
        +'绘图与门控不改变任何fit或clean文件。plot_index.npz保存所有源分片编号、行号、主图选择和诊断标记。\n',encoding='utf-8')
    atomic_json(dest/'COMPLETE.json',dict(time=now(),pilot_only=pilot,source_fingerprint=fingerprint))
    if not pilot:
        diary(f'全量后处理完成。单事件主图{int(mask.sum())}窗，8个ODE参数×3个时间常数，所有有效点保留；'
              f'pileup贴边比例={p["tau_boundary_fraction"]:.3%}，诊断标记比例={p["diagnostic_flag_fraction"]:.3%}；'
              f'决定：{report["decision"]}。图与质量报告：{dest}')
    record('smoke_complete' if pilot else 'complete',directory=str(dest),decision=report['decision'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        render(True)
        return
    lock=acquire_lock(ROOT/'plot.lock')
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        started=time.monotonic()
        record('waiting_for_all_fits')
        while True:
            state=json.loads((ROOT/'status.json').read_text(encoding='utf-8'))
            if state['state']=='complete':
                break
            if state['state'] in ['paused_for_scientific_review','stopped_error']:
                record('blocked_by_fit',fit_state=state)
                diary('拟合任务暂停或失败，未冒充全量绘图完成；见plot_status.json。')
                return
            if time.monotonic()-started>13*3600:
                raise RuntimeError('Wait exceeded 13 hours; fit status may be stale')
            time.sleep(30)
        render(False)
    except Exception:
        message=traceback.format_exc()
        record('stopped_error',error=message)
        diary('绘图任务异常停止：'+message[-2000:])
        raise
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':
    main()

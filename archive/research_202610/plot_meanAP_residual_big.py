# One large figure per case: DDPM output minus mean AP in the time domain (pilot runs).
import numpy as np, h5py, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':13})
D='/mnt/d/storage/research/DDPM4bolometer'
OUTD='/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#/results/plots/OFcompare'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
t=np.arange(10000)/1000
cols=['#1f77b4','#2ca02c','#9467bd']
for run,lab,E in [(999996,'1p382mV',15),(999996,'1p382mV',2615),(999997,'7p5mV',15),(999997,'7p5mV',2615)]:
    with h5py.File(f'{D}/nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5') as f:
        den=bsub(f['waveforms_denoised_nonoise'][:].astype(float)); noisy=bsub(f['waveforms_noisy'][:].astype(float)); tm=bsub(f['clean_template'][:].astype(float))
    AP=den.mean(0); R=(den-AP)*1e3; sR=R.std(0); sN=((noisy-tm)*1e3)[:,:1400].std()
    ex=np.random.default_rng(0).choice(len(R),3,replace=False)
    fig,axs=plt.subplots(2,1,figsize=(16,11))
    for ax,(lo,hi) in zip(axs,[(0,10),(1.4,1.9)]):
        m=(t>=lo)&(t<hi)
        for k,i in enumerate(ex): ax.plot(t[m],R[i,m],color=cols[k],lw=0.9,alpha=0.85,label=f'event {i}: DDPM output − mean AP')
        ax.plot(t[m],sR[m],'k',lw=2,label='±1σ over 5000 events'); ax.plot(t[m],-sR[m],'k',lw=2)
        ax.axhline(0,color='0.5',lw=0.6)
        ax2=ax.twinx(); ax2.plot(t[m],AP[m]*1e3,color='#c2603c',ls='--',lw=1.2); ax2.set_ylabel('mean AP (mV), dashed',color='#c2603c')
        ax2.tick_params(colors='#c2603c')
        lim=1.15*max(np.abs(R[ex][:,m][:,10 if lo==0 else 0:]).max(),sR[m][10 if lo==0 else 0:].max())
        ax.set_ylim(-lim,lim); ax.set_xlim(lo,hi); ax.grid(alpha=0.3)
        ax.set_xlabel('time (s)'); ax.set_ylabel('residual (mV)')
        ax.legend(fontsize=11,loc='upper right')
    axs[0].set_title(f'run {run}, {lab.replace("p",".")} noise, {E} keV (pulse peak {AP.max()*1e3:.0f} mV); raw input noise σ = {sN:.2f} mV\n'
                     f'residual σ: before pulse {sR[:1400].mean()*1e3:.0f} µV, rising edge max {sR[1400:1700].max()*1e3:.0f} µV, tail 2.5–9.5 s {sR[2500:9500].mean()*1e3:.0f} µV')
    axs[1].set_title('zoom 1.4–1.9 s (onset at 1.5 s)')
    fig.tight_layout(); fn=f'{OUTD}/ddpm_meanAP_residual_{lab}_{E}keV.png'; fig.savefig(fn,dpi=120); plt.close(fig); print(fn)

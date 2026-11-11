# Time-domain residual of DDPM output minus mean AP (the DDPM->OF "noise window" substitute), pilot runs.
import numpy as np, h5py, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='/mnt/d/storage/research/DDPM4bolometer'
OUT='/mnt/d/坚果云存储/复旦事宜/科研/CUPID/paperwriting/Denoising/ddpm4Bolometer#/results/plots/OFcompare/ddpm_meanAP_residual_time.png'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
t=np.arange(10000)/1000
cases=[(999996,'1.382 mV',15),(999996,'1.382 mV',2615),(999997,'7.5 mV',15),(999997,'7.5 mV',2615)]
fig,axs=plt.subplots(2,4,figsize=(19,8))
rng=np.random.default_rng(0)
for j,(run,lab,E) in enumerate(cases):
    with h5py.File(f'{D}/nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5') as f:
        den=bsub(f['waveforms_denoised_nonoise'][:].astype(float)); noisy=bsub(f['waveforms_noisy'][:].astype(float))
        tmpl=bsub(f['clean_template'][:].astype(float))
    AP=den.mean(0); R=(den-AP)*1e3; Nn=(noisy-tmpl)*1e3      # mV
    sR, sN = R.std(0), Nn.std(0)
    pk=AP.max()*1e3
    for row,(lo,hi) in enumerate([(0,10),(1.3,2.6)]):
        ax=axs[row,j]; m=(t>=lo)&(t<hi)
        ax.fill_between(t[m],-sN[m],sN[m],color='0.85',label=f'noisy input − true pulse, ±1σ ({sN[:1400].mean():.2f} mV)')
        for k,i in enumerate(rng.choice(len(R),3,replace=False)):
            ax.plot(t[m],R[i,m],lw=0.7,alpha=0.9,label='DDPM output − mean AP (examples)' if k==0 else None)
        ax.plot(t[m],sR[m],'k',lw=1.2,label='DDPM residual ±1σ over 5000 events'); ax.plot(t[m],-sR[m],'k',lw=1.2)
        ax2=ax.twinx(); ax2.plot(t[m],AP[m]*1e3,color='#c2603c',ls='--',lw=0.8); ax2.set_ylabel('mean AP (mV)',color='#c2603c',fontsize=8)
        ax2.tick_params(labelsize=7,colors='#c2603c')
        lim=max(3*sR.max(),1.2*sN.max()) if row==1 else 1.5*max(sN.max(),3*sR.max())
        ax.set_ylim(-lim,lim); ax.set_xlim(lo,hi); ax.grid(alpha=0.3)
        ax.set_xlabel('time (s)'); ax.set_ylabel('residual (mV)')
        if row==0:
            ax.set_title(f'run {run} ({lab}), {E} keV  pulse peak {pk:.0f} mV\nresidual σ: baseline {sR[:1400].mean()*1e3:.0f} µV, rising edge {sR[1400:1700].max()*1e3:.0f} µV, tail 2.5–9.5 s {sR[2500:9500].mean()*1e3:.0f} µV',fontsize=9)
            ax.legend(fontsize=7,loc='lower right')
        else:
            ax.set_title('zoom 1.3–2.6 s',fontsize=9)
        print(run,E,'baseline %.0f uV max %.0f uV at %.3f s; noisy %.0f uV; peak %.1f mV, max resid/peak %.2e'%(sR[:1400].mean()*1e3,sR.max()*1e3,t[sR.argmax()],sN[:1400].mean()*1e3,pk,sR.max()/pk))
fig.tight_layout(); fig.savefig(OUT,dpi=130); print(OUT)

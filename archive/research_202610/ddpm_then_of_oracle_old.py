# Same oracle test on the old eval set (PT frequency varies per window): DDPM -> OF vs OF on raw.
import sys; sys.path.insert(0,'.')
import numpy as np, h5py
from scripts.of_compare.plot_of_vs_ddpm_15keV import of_amplitudes
D='/mnt/d/storage/research/DDPM4bolometer'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
L=[('1p382mV',f'{D}/nonoise_out/efficiency_1p382mV','waveforms_denoised_nonoise','results/fit_results/1p382mV_nonoise'),
   ('7p5mV',f'{D}/zeroinit_out/efficiency_7p5mV','waveforms_denoised_zeroinit','results/fit_results/7p5mV_zeroinit')]
for tag,dd,field,fit in L:
    for E in (15,2615):
        with h5py.File(f'{D}/eval_data_holdout/efficiency_{tag}/eval_efficiency_{E}keV.h5') as f:
            T=f['clean_template'][:]; noisy=f['waveforms_noisy'][:]
        with h5py.File(f'{dd}/eval_efficiency_{E}keV.h5') as f: den=f[field][:].astype(float)
        with h5py.File(f'{fit}/fit_efficiency_{E}keV.h5') as f: tri=f['amp_denoised'][:]
        A,B=slice(0,2500),slice(2500,5000); S=np.fft.rfft(T); sh=np.arange(-10,10.01,0.5); res=[]
        for name,X in (('noisy',noisy),('ddpm',den)):
            R=np.fft.rfft(bsub(X[A]-T)); XB=np.fft.rfft(bsub(X[B]))
            for w,Dn in (('|N|',np.abs(R).mean(0)),('|N|^2',(np.abs(R)**2).mean(0))):
                a,_=of_amplitudes(XB,S,Dn,sh); res.append(f'{name}+OF{w} {a.std()/a.mean()*E:.4f}')
        t=tri[B]; t=t[np.isfinite(t)]; res.append(f'ddpm+triexp {t.std()/t.mean()*E:.4f}')
        print(tag,E,' | '.join(res),flush=True)

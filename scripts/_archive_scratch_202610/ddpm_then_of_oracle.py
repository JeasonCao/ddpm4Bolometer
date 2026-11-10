# Oracle test: OF applied to DDPM outputs, residual spectrum from truth (den - template) of the other half.
import sys; sys.path.insert(0,'.')
import numpy as np, h5py
from scripts.of_compare.plot_of_vs_ddpm_15keV import of_amplitudes
D='/mnt/d/storage/research/DDPM4bolometer'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
for run in (999996,999997):
    for E in (15,2615):
        with h5py.File(f'{D}/nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5') as f:
            den=f['waveforms_denoised_nonoise'][:]; T=f['clean_template'][:]; noisy=f['waveforms_noisy'][:]
        with h5py.File(f'results/fit_results/pilot_run{run}_nonoise/fit_efficiency_{E}keV.h5') as f:
            tri=f['amp_denoised'][:]
        A,B=slice(0,2500),slice(2500,5000)
        S=np.fft.rfft(T); sh=np.arange(-10,10.01,0.5)
        res=[]
        for name,X,src in (('noisy',noisy,noisy),('ddpm',den,den)):
            R=np.fft.rfft(bsub(src[A]-T))
            XB=np.fft.rfft(bsub(X[B]))
            for w,Dn in (('|N|',np.abs(R).mean(0)),('|N|^2',(np.abs(R)**2).mean(0))):
                a,_=of_amplitudes(XB,S,Dn,sh); res.append(f'{name}+OF{w} {a.std()/a.mean()*E:.4f}')
        t=tri[B]; res.append(f'ddpm+triexp {t.std()/t.mean()*E:.4f}')
        print(run,E,' | '.join(res),flush=True)

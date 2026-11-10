# DDPM -> OF: how much does the ANPS choice matter? Pilot runs, half A builds AP/ANPS, half B is evaluated.
import sys; sys.path.insert(0,'.')
import numpy as np, h5py
from scripts.of_compare.plot_of_vs_ddpm_15keV import of_amplitudes
D='/mnt/d/storage/research/DDPM4bolometer'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
sh=np.arange(-10,10.01,0.5)
def of(XB_t, AP, Dn):
    a,_=of_amplitudes(np.fft.rfft(XB_t),np.fft.rfft(AP),Dn,sh); return a
def interp_psd(seg, L):
    """mean |FFT| of short segments, interpolated onto the frequency grid of length-L windows"""
    P=np.abs(np.fft.rfft(seg)).mean(0); f=np.fft.rfftfreq(seg.shape[1]); return np.interp(np.fft.rfftfreq(L),f,P)
for run in (999996,999997):
    for E in (15,2615):
        with h5py.File(f'{D}/nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5') as f: den=bsub(f['waveforms_denoised_nonoise'][:].astype(float))
        A,B=den[:2500],den[2500:]; AP=A.mean(0); RA=A-AP; L=den.shape[1]
        res={}
        res['1 current |N|']=of(B,AP,np.abs(np.fft.rfft(RA)).mean(0))
        res['2 current |N|^2']=of(B,AP,(np.abs(np.fft.rfft(RA))**2).mean(0))
        res['3 white']=of(B,AP,np.ones(L//2+1))
        res['4 pre-pulse 1.4 s only']=of(B,AP,interp_psd(RA[:,:1400],L))
        res['5 2 s segments (Welch-like)']=of(B,AP,interp_psd(RA.reshape(-1,2000),L))
        APB=B.mean(0); res['6 cheat: AP+ANPS from B itself']=of(B,APB,np.abs(np.fft.rfft(B-APB)).mean(0))
        c=slice(1000,4000); APc=A[:,c].mean(0)
        res['7 crop 1-4 s (3 s window)']=of(B[:,c],APc,np.abs(np.fft.rfft(A[:,c]-APc)).mean(0))
        print(f'run {run} {E} keV: '+'  '.join(f'{k}: {v.std()/v.mean()*E:.4f}' for k,v in res.items()),flush=True)

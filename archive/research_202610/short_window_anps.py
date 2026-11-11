# Can the ANPS come from shorter windows than the 10 s AP/signal windows? Pilot runs, halves A -> B.
# Short-segment ANPS = mean |FFT| of the segments, interpolated onto the 10 s frequency grid
# (bins below 1/T_seg take the lowest available value). Overall scale does not matter for the OF.
import sys; sys.path.insert(0,'.')
import numpy as np, h5py
from scripts.of_compare.plot_of_vs_ddpm_15keV import of_amplitudes
D='/mnt/d/storage/research/DDPM4bolometer'
bsub=lambda x: x-x[...,:1400].mean(-1,keepdims=True)
sh=np.arange(-10,10.01,0.5); L=10000; F=np.fft.rfftfreq(L)
def seg_anps(segs):
    segs=segs-segs.mean(1,keepdims=True)
    P=np.abs(np.fft.rfft(segs)).mean(0); f=np.fft.rfftfreq(segs.shape[1])
    # the segment DC bin is ~0 after mean removal: drop it, bins below 1/T_seg take the lowest nonzero bin
    return np.interp(F,f[1:],P[1:])
def cut(x,a,b,n):   # consecutive segments of n samples from x[:, a:b]
    y=x[:,a:a+(b-a)//n*n]; return y.reshape(-1,n)
def res(XB,AP,Dn,E):
    a,_=of_amplitudes(np.fft.rfft(XB),np.fft.rfft(AP),Dn,sh); return a.std()/a.mean()*E
for run in (999996,999997):
    for E in (15,2615):
        with h5py.File(f'{D}/nonoise_out/pilot_run{run}/eval_efficiency_{E}keV.h5') as f:
            den=bsub(f['waveforms_denoised_nonoise'][:].astype(float)); noisy=bsub(f['waveforms_noisy'][:].astype(float)); tm=bsub(f['clean_template'][:].astype(float))
        out=[]
        # raw data: pure noise = noisy - true template (stationary pilot noise)
        N=noisy-tm; AP=tm if tm.ndim==1 else tm.mean(0)
        out.append(('raw 10s', res(noisy[2500:],AP,np.abs(np.fft.rfft(N[:2500])).mean(0),E)))
        out.append(('raw 5s', res(noisy[2500:],AP,seg_anps(cut(N[:2500],0,10000,5000)),E)))
        out.append(('raw 1s', res(noisy[2500:],AP,seg_anps(cut(N[:2500],0,10000,1000)),E)))
        out.append(('raw pre-pulse 1.4s', res(noisy[2500:],AP,seg_anps(N[:2500,:1400]),E)))
        # DDPM output: residual about the mean AP
        A=den[:2500]; APd=A.mean(0); R=A-APd; B=den[2500:]
        out.append(('ddpm 10s', res(B,APd,np.abs(np.fft.rfft(R)).mean(0),E)))
        out.append(('ddpm 5s (whole window)', res(B,APd,seg_anps(cut(R,0,10000,5000)),E)))
        out.append(('ddpm 1s (whole window)', res(B,APd,seg_anps(cut(R,0,10000,1000)),E)))
        out.append(('ddpm pre-pulse 1.4s', res(B,APd,seg_anps(R[:,:1400]),E)))
        out.append(('ddpm tail 5-10s', res(B,APd,seg_anps(R[:,5000:10000]),E)))
        print(f'run {run} {E:4d} keV | '+' | '.join(f'{k} {v:.4f}' for k,v in out),flush=True)

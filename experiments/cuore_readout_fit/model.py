"""Causal effective-readout models for CUORE. NOT a calibrated thermal inversion.
Bessel Wn is explicitly -3 dB (norm='mag'). Voltage units are normalized ADC.
The nonlinear electrical stage uses the constant-current (RB/RL -> 0) limit;
its nonlinearity is an effective coefficient, not a measured temperature.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.signal import bessel,sosfilt
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from experiments.cuore_fit_ranges.stable_fitter import evaluate as thermal, make_bounds, CONFIGS, seed

def electrical(x,J,fs,rc,k,nonlinear):
    """Midpoint, piecewise-constant solution of voltage circuit and sensitivities.
    du/dt=(expm1(k*x)/k-exp(k*x)*u)/rc; k->0 gives (x-u)/rc.
    Returned derivatives include log(rc), log(k). No acausal filtfilt.
    """
    xm=(x+np.r_[0.,x[:-1]])*.5
    jm=(J+np.vstack([np.zeros((1,J.shape[1])),J[:-1]]))*.5
    if nonlinear:
        q=k*xm
        target=-np.expm1(-q)/k
        target_x=np.exp(-q)
        target_k=xm*target_x-target
        rate=np.exp(q)/rc
    else:
        target=xm;target_x=np.ones_like(xm);target_k=np.zeros_like(xm);rate=np.full_like(xm,1/rc)
    a=np.exp(-rate/fs)
    band=np.zeros((2,len(x)));band[0]=1.;band[1,:-1]=-a[1:]
    rhs=(1-a)*target;rhs[0]=0.
    u=solve_banded((1,0),band,rhs,check_finite=False)
    prev=np.r_[0.,u[:-1]]
    ax=-a*rate/fs*k if nonlinear else np.zeros_like(xm)
    common=(1-a)*target_x+ax*(prev-target)
    rhs_j=common[:,None]*jm
    rhs_rc=(a*rate/fs)*(prev-target)
    rhs_k=(1-a)*target_k-(a*rate/fs*k*xm)*(prev-target) if nonlinear else np.zeros_like(xm)
    rhs_all=np.column_stack([rhs_j,rhs_rc,rhs_k]);rhs_all[0]=0.
    grad=solve_banded((1,0),band,rhs_all,check_finite=False)
    return u,grad[:,:J.shape[1]],grad[:,-2],grad[:,-1]

class Forward:
    def __init__(self,n,kind='bessel',fc=None,rc=None,oversample=2):
        if kind not in ('bare','bessel','rc','nonlinear'): raise ValueError('Unknown model kind')
        if not isinstance(oversample,int) or oversample<1: raise ValueError('oversample must be a positive integer')
        if n<2: raise ValueError('At least two samples required')
        if fc is not None and not 0<fc<500*oversample: raise ValueError('fc must lie below internal Nyquist')
        if rc is not None and rc<=0: raise ValueError('rc must be positive')
        self.n=n;self.kind=kind;self.fixed_fc=fc;self.fixed_rc=rc
        self.os=oversample;self.fs=1000.*oversample;self.t=np.arange(n*oversample)/self.fs
        self.extra=[]
        if kind!='bare' and fc is None:self.extra.append('log_fc')
        if kind in ('rc','nonlinear') and rc is None:self.extra.append('log_rc')
        if kind=='nonlinear':self.extra.append('log_k')
    def settings(self,p):
        d=dict(zip(self.extra,p[7:]))
        fc=self.fixed_fc if self.fixed_fc is not None else np.exp(d.get('log_fc',np.log(12)))
        rc=self.fixed_rc if self.fixed_rc is not None else np.exp(d.get('log_rc',np.log(.02)))
        k=np.exp(d.get('log_k',np.log(.03)))
        return fc,rc,k
    def evaluate(self,p):
        b=p[0];q=np.array(p[:7]);q[0]=0.
        x,J=thermal(self.t,q);J[:,0]=0.
        fc,rc,k=self.settings(p);parts={}
        if self.kind in ('rc','nonlinear'):
            x,J,drc,dk=electrical(x,J,self.fs,rc,k,self.kind=='nonlinear')
            parts['log_rc']=drc;parts['log_k']=dk
        if self.kind!='bare':
            sos=bessel(6,fc,fs=self.fs,norm='mag',output='sos')
            pre=x
            x=sosfilt(sos,x);J=sosfilt(sos,J,axis=0)
            parts={key:sosfilt(sos,v) for key,v in parts.items()}
            if 'log_fc' in self.extra:
                eps=1e-4
                plus=sosfilt(bessel(6,fc*np.exp(eps),fs=self.fs,norm='mag',output='sos'),pre)
                minus=sosfilt(bessel(6,fc*np.exp(-eps),fs=self.fs,norm='mag',output='sos'),pre)
                parts['log_fc']=(plus-minus)/(2*eps)
        pred=x[::self.os]+b
        J=J[::self.os];J[:,0]=1.
        if self.extra:J=np.column_stack([J]+[parts[name][::self.os] for name in self.extra])
        return pred,J
    def bounds(self,window):
        lo,hi=make_bounds(CONFIGS['moderate'],[window])
        limits={'log_fc':(1.,150.),'log_rc':(.0005,.3),'log_k':(1e-5,1.)}
        return np.r_[lo,[np.log(limits[k][0]) for k in self.extra]],np.r_[hi,[np.log(limits[k][1]) for k in self.extra]]

def fit(y,kind='bessel',fc=None,rc=None,oversample=2,max_nfev=300,nstarts=3,initial=None):
    if nstarts not in (0,1,2,3) or (nstarts==0 and initial is None): raise ValueError('Need 1-3 seeds or an explicit initial vector')
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not np.all(np.isfinite(y)): raise ValueError('Expected a finite 1-D waveform')
    t=np.arange(len(y))/1000.;tp=float(t[np.argmax(y)])
    window=(max(0,tp-.95),min(t[-1],tp-.001))
    fw=Forward(len(y),kind,fc,rc,oversample);lo,hi=fw.bounds(window)
    candidates=[]
    starts=[(.035,.30,1.1,8.),(.08,.25,1.3,15.),(.15,.22,1.1,35.)]
    for tr,tf,ts,f0 in starts[:nstarts]:
        pp=seed(y,t,[tp],[window],CONFIGS['moderate'],(tr,tf,ts))
        delay=2.7/(2*np.pi*(fc or f0)) # initial guess only, not fixed alignment
        pp[3]=np.clip(tp-.18-delay,window[0]+1e-8,window[1]-1e-8)
        extras={'log_fc':np.log(f0),'log_rc':np.log(.02),'log_k':np.log(.05)}
        candidates.append(np.r_[pp,[extras[k] for k in fw.extra]])
    if initial is not None:candidates.insert(0,np.array(initial))
    results=[]
    for pp in candidates:
        last=None;pred=None;J=None
        def calc(v):
            nonlocal last,pred,J
            if last is None or not np.array_equal(last,v):
                pred,J=fw.evaluate(v);last=v.copy()
            return pred,J
        start=time.perf_counter()
        sol=least_squares(lambda v:calc(v)[0]-y,np.clip(pp,lo+1e-9,hi-1e-9),
            jac=lambda v:calc(v)[1],bounds=(lo,hi),x_scale='jac',max_nfev=max_nfev,
            ftol=2e-9,xtol=2e-9,gtol=1e-8)
        fitted,jac=fw.evaluate(sol.x);res=y-fitted
        C,D,tr,gap=np.exp(sol.x[[1,2,4,6]]);delta=sol.x[5];tf=tr*np.exp(delta);ts=tf+gap
        mask=(t>tp-.3)&(t<tp+.65)
        fc_v,rc_v,k=fw.settings(sol.x)
        norm=np.linalg.norm(jac,axis=0);sv=np.linalg.svd(jac/np.maximum(norm,1e-30),compute_uv=False)
        # delta=0 is a separate repeated-pole diagnostic; its upper bound still counts.
        boundary=[int(i) for i in range(len(sol.x)) if
            (i!=5 and min((sol.x[i]-lo[i])/(hi[i]-lo[i]),(hi[i]-sol.x[i])/(hi[i]-lo[i]))<.001)
            or (i==5 and (hi[i]-sol.x[i])/(hi[i]-lo[i])<.001)]
        results.append(dict(p=sol.x.tolist(),kind=kind,fixed_fc=fc,fixed_rc=rc,
            oversample=oversample,success=bool(sol.success),message=sol.message,nfev=sol.nfev,
            nrmse=float(np.sqrt(np.mean(res**2))),peak_region_rms=float(np.sqrt(np.mean(res[mask]**2))),
            max_abs_residual=float(np.max(abs(res))),peak_time_value_error=float(fitted[np.argmax(y)]-1),
            tau=[float(tr),float(tf),float(ts)],delta=float(delta),alpha=float(C/(C+D*delta)),
            fc_hz=float(fc_v) if kind!='bare' else None,
            rc_s=float(rc_v) if kind in ('rc','nonlinear') else None,
            k=float(k) if kind=='nonlinear' else None,
            condition=float(sv[0]/max(sv[-1],1e-30)),boundary=boundary,elapsed=time.perf_counter()-start))
    return sorted(results,key=lambda r:(not r['success'],r['nrmse']))

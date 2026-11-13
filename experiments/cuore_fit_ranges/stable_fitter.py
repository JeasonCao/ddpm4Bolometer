"""Stable, ordered, positive-coefficient version of the SAME tri-exponential family.
C*(exp(-u*exp(-delta))-exp(-u))/delta + D*(exp(-t/ts)-exp(-u)).
tr=exp(lr), tf=tr*exp(delta), ts=tf+exp(lgap).
For delta>0: A=C/delta+D, alpha=(C/delta)/A.
At delta=0 the exact repeated-pole limit is retained explicitly.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import time
CONFIGS={
 'compact':dict(rise=(.01,.25),ratio=20.,gap=(.01,10.)),
 'moderate':dict(rise=(.002,.6),ratio=100.,gap=(.002,15.)),
 'wide':dict(rise=(.001,2.),ratio=1000.,gap=(.001,30.))
}
SEEDS=[(.19,.21,1.1),(.04,.30,1.2),(.10,.40,2.)]

def evaluate(t,p):
    y=np.full_like(t,p[0]); J=np.zeros((len(t),len(p))); J[:,0]=1
    for j in range(1,len(p),6):
        lc,ld,t0,lr,delta,lg=p[j:j+6]
        C,D,tr,gap=np.exp([lc,ld,lr,lg]); tf=tr*np.exp(delta); ts=tf+gap
        dt=np.maximum(t-t0,0.); active=t>t0; u=dt/tr
        er=np.exp(-u); es=np.exp(-dt/ts)
        if delta<1e-4:
            H=u+.5*(u*u-u)*delta+(u**3-3*u*u+u)*delta**2/6
            Hp=1+(u-.5)*delta+(3*u*u-6*u+1)*delta**2/6
            h=er*H
            hlr=er*u*(H-Hp)
            ht0=er*(H-Hp)/tr
            hd=er*(.5*(u*u-u)+(u**3-3*u*u+u)*delta/3)
        else:
            v=np.exp(-delta); ef=np.exp(-u*v)
            diff=ef*(-np.expm1(-u*(1-v)))
            h=diff/delta
            hlr=u*(v*ef-er)/delta
            ht0=(v*ef-er)/(tr*delta)
            hd=(u*v*ef*delta-diff)/delta**2
        k=es-er
        klr=es*dt/ts*tf/ts-er*u
        kd=es*dt/ts*tf/ts
        kg=es*dt/ts*gap/ts
        kt0=es/ts-er/tr
        y+=(C*h+D*k)*active
        J[:,j]=C*h*active; J[:,j+1]=D*k*active
        J[:,j+2]=(C*ht0+D*kt0)*active
        J[:,j+3]=(C*hlr+D*klr)*active
        J[:,j+4]=(C*hd+D*kd)*active
        J[:,j+5]=D*kg*active
    return y,J

def make_bounds(cfg,windows):
    lo=[-.1]; hi=[.1]
    for left,right in windows:
        lo += [np.log(1e-6),np.log(1e-6),left,np.log(cfg['rise'][0]),0.,np.log(cfg['gap'][0])]
        hi += [np.log(100.),np.log(10.),right,np.log(cfg['rise'][1]),np.log(cfg['ratio']),np.log(cfg['gap'][1])]
    return np.array(lo),np.array(hi)

def peaks(y,label):
    sm=gaussian_filter1d(y,10.)
    pp,props=find_peaks(sm,prominence=.025,distance=120)
    # Keep dominant pulse for singles; pileup supports up to three observed peaks.
    order=np.argsort(sm[pp])[::-1]
    if not label: return [int(np.argmax(sm))/1000.],len(pp),False
    selected=list(pp[order[:3]])
    if sm[-1]>.05 and sm[-1]-sm[-101]>.025 and all(abs(v-9999)>200 for v in selected):
        selected.append(9999)
    missing=len(selected)<2
    if not selected: selected=[int(np.argmax(sm))]
    if len(selected)<2: selected.append(min(selected[0]+3000,9900))
    return sorted(set(v/1000. for v in selected[:3])),len(pp),missing

def seed(y,t,tps,windows,cfg,taus):
    tr,tf,ts=taus; delta=np.log(tf/tr); gap=ts-tf
    p=[float(np.median(y[t<.5]))]
    for tp,window in zip(tps,windows):
        amp=max(float(np.interp(tp,t,y)-p[0]),.02)
        p += [np.log(2.5*amp),np.log(.12*amp),np.clip(tp-.21,*window),np.log(tr),delta,np.log(gap)]
    lo,hi=make_bounds(cfg,windows)
    return np.clip(p,lo+1e-9,hi-1e-9)

def solve(t,y,p,cfg,windows,max_nfev=400):
    lo,hi=make_bounds(cfg,windows)
    last=None; val=None; jac=None
    def calc(p):
        nonlocal last,val,jac
        if last is None or not np.array_equal(last,p):
            val,jac=evaluate(t,p);last=p.copy()
        return val,jac
    start=time.perf_counter()
    r=least_squares(lambda p:calc(p)[0]-y,np.clip(p,lo+1e-10,hi-1e-10),
       jac=lambda p:calc(p)[1],bounds=(lo,hi),x_scale='jac',max_nfev=max_nfev,
       ftol=1e-9,xtol=1e-9,gtol=1e-8)
    pred,J=evaluate(t,r.x)
    # Condition number diagnostic, not a claim of calibrated uncertainties.
    norms=np.linalg.norm(J,axis=0); s=np.linalg.svd(J/np.maximum(norms,1e-30),compute_uv=False)
    flags=[]; taus=[]; alphas=[]; peaks_out=[]; collapsed=[]; boundary=[]
    for j in range(1,len(r.x),6):
        lc,ld,t0,lr,d,lg=r.x[j:j+6]
        C,D,tr,gap=np.exp([lc,ld,lr,lg]); tf=tr*np.exp(d); ts=tf+gap
        taus.append([float(tr),float(tf),float(ts)])
        alphas.append(float(C/(C+D*d)))
        collapsed.append(bool(d<.05)) # resolved ratio <1.051 is a diagnostic, not deletion.
        inds=[j,j+1,j+3,j+5] # delta=0 is the mathematical limit, separately flagged
        near=any(min((r.x[k]-lo[k])/(hi[k]-lo[k]),(hi[k]-r.x[k])/(hi[k]-lo[k]))<.001 for k in inds)
        near |= (hi[j+4]-r.x[j+4])/(hi[j+4]-lo[j+4])<.001
        boundary.append(bool(near))
        tt=np.linspace(t0,min(t0+10,20),2000)
        pp=np.r_[0.,r.x[j:j+6]]
        peaks_out.append(float(np.max(evaluate(tt,pp)[0])))
    return dict(p=r.x.tolist(),success=bool(r.success),nfev=int(r.nfev),
                nrmse=float(np.sqrt(np.mean((pred-y)**2))),elapsed=time.perf_counter()-start,
                tau=taus,alpha=alphas,peak_amp=peaks_out,collapsed=collapsed,
                boundary=boundary,condition=float(s[0]/max(s[-1],1e-30)),message=r.message,
                dominant=int(np.argmax(peaks_out)))

def fit(y,label,config='moderate',stride=5,nseeds=3,max_nfev=400):
    t=np.arange(len(y))/1000.
    tps,detected,missing=peaks(y,label)
    windows=[]
    for i,tp in enumerate(tps):
        left=max(0,tp-.9)
        if i: left=max(left,tps[i-1]-.05)
        windows.append((left,min(9.999,tp-.001)))
    cfg=CONFIGS[config]
    rr=[]
    for taus in SEEDS[:nseeds]:
        p=seed(y,t,tps,windows,cfg,taus)
        r=solve(t[::stride],y[::stride],p,cfg,windows,max_nfev)
        r.update(detected_peaks=detected,missing_peak=missing,windows=windows,seed=list(taus),n_pulses=len(tps))
        rr.append(r)
    return sorted(rr,key=lambda r:r['nrmse'])

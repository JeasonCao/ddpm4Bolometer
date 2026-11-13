"""Independent CUORE tri-exponential range study. Original inputs/core fitter untouched."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import time

RANGES={
 'legacy':dict(tr=(.01,.2),d1=(.0670820393249937,10.),d2=(.45,10.),alpha=(.01,.99),amax=1000.),
 'alpha_open':dict(tr=(.01,.2),d1=(.0670820393249937,10.),d2=(.45,10.),alpha=(1e-6,1-1e-6),amax=10000.),
 'expanded':dict(tr=(.002,.6),d1=(.002,5.),d2=(.01,15.),alpha=(1e-6,1-1e-6),amax=10000.),
 'negative_prefactor':dict(tr=(.001,.6),d1=(.002,5.),d2=(.01,15.),alpha=(1.000001,100.),amax=10000.,sign=-1),
 'wide':dict(tr=(.001,2.),d1=(.001,10.),d2=(.001,30.),alpha=(1e-8,1-1e-8),amax=100000.)
}
SEEDS=[(.045,.3,3.,.5),(.12,.16,1.,.95),(.25,.06,1.,.25),(.06,.25,1.,.8)]

def model_jac(t,p,sign=1):
    # p = baseline + repeated(log A, t0, log tr, alpha, log td1, log td2).
    y=np.full_like(t,p[0]); jac=np.zeros((len(t),len(p))); jac[:,0]=1.
    for j in range(1,len(p),6):
        la,t0,lr,al,l1,l2=p[j:j+6]
        A,tr,d1,d2=np.exp([la,lr,l1,l2]); A*=sign
        dt=np.maximum(t-t0,0.); active=(t>t0)
        er,e1,e2=np.exp(-dt/tr),np.exp(-dt/d1),np.exp(-dt/d2)
        shape=(-er+al*e1+(1-al)*e2)*active
        y+=A*shape
        jac[:,j]=A*shape
        jac[:,j+1]=A*(-er/tr+al*e1/d1+(1-al)*e2/d2)*active
        jac[:,j+2]=-A*er*dt/tr
        jac[:,j+3]=A*(e1-e2)*active
        jac[:,j+4]=A*al*e1*dt/d1
        jac[:,j+5]=A*(1-al)*e2*dt/d2
    return y,jac

def bounds(config,windows):
    lo=[-.2]; hi=[.2]
    for left,right in windows:
        lo.extend([np.log(1e-6),left,np.log(config['tr'][0]),config['alpha'][0],np.log(config['d1'][0]),np.log(config['d2'][0])])
        hi.extend([np.log(config['amax']),right,np.log(config['tr'][1]),config['alpha'][1],np.log(config['d1'][1]),np.log(config['d2'][1])])
    return np.array(lo),np.array(hi)

def seed_params(y,t,peak_times,windows,config,seed):
    tr,d1,d2,alpha=seed
    tr=np.clip(tr,*config['tr']); d1=np.clip(d1,*config['d1']); d2=np.clip(d2,*config['d2'])
    alpha=np.clip(alpha,*config['alpha'])
    tt=np.linspace(0,2.,1000)
    shape=config.get('sign',1)*(-np.exp(-tt/tr)+alpha*np.exp(-tt/d1)+(1-alpha)*np.exp(-tt/d2))
    ip=int(np.argmax(shape)); peak=float(max(shape[ip],.001))
    p=[float(np.median(y[t<.5]))]
    for tp,window in zip(peak_times,windows):
        observed=float(np.interp(tp,t,y)-p[0])
        p.extend([np.log(max(observed,.02)/peak),np.clip(tp-tt[ip],*window),np.log(tr),alpha,np.log(d1),np.log(d2)])
    lo,hi=bounds(config,windows)
    return np.clip(p,lo+1e-9,hi-1e-9)

def solve(t,y,p,config,windows,max_nfev=500):
    lo,hi=bounds(config,windows)
    cached_p=None; cached_y=None; cached_j=None
    def calc(p):
        nonlocal cached_p,cached_y,cached_j
        if cached_p is None or not np.array_equal(cached_p,p):
            cached_y,cached_j=model_jac(t,p,config.get('sign',1)); cached_p=p.copy()
        return cached_y,cached_j
    start=time.perf_counter()
    res=least_squares(lambda p:calc(p)[0]-y,np.clip(p,lo+1e-10,hi-1e-10),
                      jac=lambda p:calc(p)[1],bounds=(lo,hi),method='trf',x_scale='jac',
                      max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    yhat,J=model_jac(t,res.x,config.get('sign',1))
    nrmse=float(np.sqrt(np.mean((yhat-y)**2)))
    s=np.linalg.svd(J,compute_uv=False)
    cond=float(s[0]/max(s[-1],1e-30))
    cov=np.linalg.pinv(J.T@J,rcond=1e-13)*np.sum((yhat-y)**2)/max(1,len(y)-len(res.x))
    return dict(sign=config.get('sign',1),p=res.x.tolist(),success=bool(res.success),nfev=res.nfev,nrmse=nrmse,
                optimality=float(res.optimality),condition=cond,stderr=np.sqrt(np.maximum(0,np.diag(cov))).tolist(),
                elapsed=time.perf_counter()-start,message=res.message)

def describe(r):
    r=dict(r); p=np.array(r['p']); tau=[]; alpha=[]; amp=[]
    for j in range(1,len(p),6):
        tr,d1,d2=np.exp(p[[j+2,j+4,j+5]]); a=p[j+3]
        if d1>d2: d1,d2,a=d2,d1,1-a
        tau.append([tr,d1,d2]); alpha.append(a); amp.append(float(r.get('sign',1)*np.exp(p[j])))
    r.update(tau=tau,alpha=alpha,A=amp)
    return r

def single(y,config_name,stride=5,all_seeds=True,max_nfev=500):
    t=np.arange(len(y))/1000.
    tp=float(np.argmax(gaussian_filter1d(y,10))/1000.)
    windows=[(max(0.,tp-.9),min(9.999,tp-.005))]
    config=RANGES[config_name]
    results=[]
    seeds=SEEDS if config.get('sign',1)>0 else [(.03,.2,1.,8.),(.08,.2,1.,3.),(.1,.15,1.,1.6),(.01,.3,1.,40.)]
    for seed in (seeds if all_seeds else seeds[:1]):
        p=seed_params(y,t,[tp],windows,config,seed)
        r=solve(t[::stride],y[::stride],p,config,windows,max_nfev)
        r['seed']=list(seed); results.append(describe(r))
    return sorted(results,key=lambda r:r['nrmse'])

"""Numerical checks and fixed-separation profiles, not statistical confidence intervals."""
import argparse,json
from pathlib import Path
import numpy as np,h5py
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from stable_fitter import evaluate,fit,make_bounds,CONFIGS

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);t=np.arange(10000)/1000.
 checks=[]
 for delta in [0.,1e-6,.01,1.]:
  pp=np.array([0,np.log(2),np.log(.1),2.9354,np.log(.1),delta,np.log(1.)])
  J=evaluate(t[::20],pp)[1]
  fd=approx_derivative(lambda x:evaluate(t[::20],x)[0],pp,method='3-point',bounds=([-np.inf]*5+[0,-np.inf],[np.inf]*7))
  err=float(np.max(abs(J-fd))/np.max(abs(fd)));assert err<1e-5
  checks.append(dict(kind='analytic_jacobian',delta=delta,relative_error=err))
 for tr,tf,ts in [(.03,.3,1.2),(.08,.25,2.0)]:
  C,D=2.,.2;delta=np.log(tf/tr)
  pp=np.array([0,np.log(C),np.log(D),3.,np.log(tr),delta,np.log(ts-tf)])
  y=evaluate(t,pp)[0]; dt=np.maximum(t-3.,0)
  A=C/delta+D; alpha=(C/delta)/A
  old=A*(-np.exp(-dt/tr)+alpha*np.exp(-dt/tf)+(1-alpha)*np.exp(-dt/ts))*(t>3.)
  err=float(np.max(abs(y-old)));assert err<1e-12
  r=fit(y/np.max(y),0,stride=1,nseeds=3)[0]
  terr=float(np.max(abs(np.array(r['tau'][0])/[tr,tf,ts]-1)))
  assert terr<1e-3 and r['nrmse']<1e-6,(terr,r['nrmse'])
  checks.append(dict(kind='synthetic_recovery',tau=[tr,tf,ts],equivalence_max_error=err,tau_relative_error=terr,nrmse=r['nrmse']))
 profiles=[]
 with h5py.File(Path(a.data)/'cuoreTraining.h5','r') as f:
  amps=f['normalizationScale'][:4500,0];ids=np.argsort(amps)[(np.array([.1,.5,.9])*4499).astype(int)]
  for row in ids:
   y=(f['Waveform'][row]-f['normalizationOffset'][row,0])/amps[row]
   best=fit(y,0,stride=1,nseeds=3)[0];orig=np.array(best['p'])
   lo,hi=make_bounds(CONFIGS['moderate'],best['windows']);free=np.array([0,1,2,3,4,6])
   points=[]
   for delta in [0.,.001,.005,.01,.02,.05,.1,.2,.4,.8,1.2]:
    base=orig.copy();base[4]-=(delta-base[5])/2;base[5]=delta
    def pack(x):
     pp=base.copy();pp[free]=x;return pp
    res=least_squares(lambda x:evaluate(t,pack(x))[0]-y,np.clip(base[free],lo[free]+1e-10,hi[free]-1e-10),
       jac=lambda x:evaluate(t,pack(x))[1][:,free],bounds=(lo[free],hi[free]),x_scale='jac',max_nfev=500,ftol=1e-10,xtol=1e-10,gtol=1e-9)
    points.append(dict(delta=delta,ratio=float(np.exp(delta)),nrmse=float(np.sqrt(np.mean(res.fun**2))),success=bool(res.success)))
   profiles.append(dict(row=int(row),event_id=int(f['eventId'][row]),amplitude_adc=float(amps[row]),points=points))
 (out/'numerical_checks_and_profiles.json').write_text(json.dumps(dict(checks=checks,profiles=profiles),indent=2),encoding='utf-8')
 print(json.dumps(dict(checks=checks,profile_rows=[r['row'] for r in profiles])),flush=True)
if __name__=='__main__':main()

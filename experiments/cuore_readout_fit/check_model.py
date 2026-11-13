"""Numerical implementation checks; these do not validate CUORE hardware settings."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.cuore_readout_fit.model import Forward,fit
from experiments.cuore_fit_ranges.stable_fitter import evaluate as bare

def check():
    # Off-grid onset: the unfiltered piecewise model is not differentiable exactly at onset.
    p0=np.array([.001,np.log(2),np.log(.2),2.900237,np.log(.04),np.log(6.),np.log(.9)])
    values={'log_fc':np.log(12.),'log_rc':np.log(.02),'log_k':np.log(.1)}
    report={}
    for kind in ['bare','bessel','rc','nonlinear']:
        f=Forward(10000,kind);p=np.r_[p0,[values[k] for k in f.extra]]
        y,J=f.evaluate(p);errors=[]
        for j in range(len(p)):
            a=p.copy();b=p.copy();a[j]+=1e-6;b[j]-=1e-6
            fd=(f.evaluate(a)[0]-f.evaluate(b)[0])/2e-6
            errors.append(float(np.max(abs(fd-J[:,j]))/max(np.max(abs(fd)),1e-7)))
        assert max(errors)<1e-5,(kind,errors)
        assert np.max(abs(y[:2900]-p0[0]))<1e-12
        report[kind+'_jacobian_relative_max']=max(errors)
    ybare=Forward(10000,'bare').evaluate(p0)[0]
    assert np.max(abs(ybare-bare(np.arange(10000)/1000,p0)[0]))<1e-12
    for kind in ['bessel','rc','nonlinear']:
        pp=np.r_[p0,np.log(.1)] if kind=='nonlinear' else p0
        ys=[Forward(10000,kind,6.,.08,os_).evaluate(pp)[0] for os_ in [2,4,8]]
        d24=float(np.max(abs(ys[0]-ys[1])));d48=float(np.max(abs(ys[1]-ys[2])))
        assert d24<1e-4 and d48<d24
        report[kind+'_oversample_max_2vs4']=d24
        report[kind+'_oversample_max_4vs8']=d48
    linear=Forward(10000,'rc',6.,.08).evaluate(p0)[0]
    nonlinear=Forward(10000,'nonlinear',6.,.08).evaluate(np.r_[p0,np.log(1e-8)])[0]
    report['linear_limit_max_error']=float(np.max(abs(linear-nonlinear)))
    assert report['linear_limit_max_error']<1e-7
    for kind in ['bessel','rc','nonlinear']:
        pp=np.r_[p0,np.log(.1)] if kind=='nonlinear' else p0
        yy=Forward(10000,kind,6.,.08).evaluate(pp)[0]
        r=fit(yy,kind,6.,.08,nstarts=3)[0]
        report[kind+'_noiseless_recovery_rms']=r['nrmse']
        report[kind+'_noiseless_tau_relative_error']=float(np.max(abs(np.asarray(r['tau'])/[.04,.24,1.14]-1)))
        assert r['nrmse']<1e-6,(kind,r)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=check();Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

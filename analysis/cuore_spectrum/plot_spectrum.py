"""Raw CUORE single-event spectra and a conditional, illustrative two-line energy scale."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np,h5py
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import ndtr
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False})
# Reference assignments are hypotheses checked against multiple observed peaks.
LINES=[
 ('annihilation',510.999,5687.5,115.,'511：湮没'),
 ('Tl208',583.191,6487.5,150.,'583：Tl-208'),
 ('Bi212_candidate',727.330,8062.5,160.,'727候选'),
 ('Ac228',911.204,10062.5,170.,'911：Ac-228'),
 ('Ac228_blend',968.971,10687.5,150.,'969：Ac-228'),
 ('near1590_blend_candidate',2614.511-2*510.999,17387.5,200.,'~1590：混峰候选'),
 ('single_escape_candidate',2614.511-510.999,22787.5,250.,'2104：单逃逸候选'),
 ('Tl208_high',2614.511,28112.5,250.,'2615：Tl-208')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def peakfit(values,seed,halfwidth,binwidth=10.):
    edges=np.arange(seed-halfwidth,seed+halfwidth+binwidth*.5,binwidth)
    counts,_=np.histogram(values,edges)
    bg=max(np.median(np.r_[counts[:3],counts[-3:]])/binwidth,.001)
    area=max(counts.sum()-bg*(edges[-1]-edges[0]),5.)
    def expectation(p):
        n,mu,sigma,b=np.exp(p[0]),seed+p[1]*halfwidth,np.exp(p[2]),np.exp(p[3])
        return n*np.diff(ndtr((edges-mu)/sigma))+b*np.diff(edges)
    def objective(p):
        lam=expectation(p)
        return float(np.sum(lam-counts*np.log(lam)))
    sol=minimize(objective,[np.log(area),0.,np.log(30.),np.log(bg)],method='L-BFGS-B',
        bounds=[(np.log(.01),np.log(max(20,3*counts.sum()))),(-.65,.65),(np.log(3.),np.log(halfwidth*.7)),(np.log(1e-6),np.log(max(counts.sum()/binwidth,1.)))])
    n,mu,sigma,b=np.exp(sol.x[0]),seed+sol.x[1]*halfwidth,np.exp(sol.x[2]),np.exp(sol.x[3])
    return dict(mu_adc=float(mu),sigma_adc=float(sigma),signal_area=float(n),background_per_adc=float(b),
                fit_success=bool(sol.success),events_in_window=int(counts.sum()),bin_width_adc=binwidth,
                window_low=float(edges[0]),window_high=float(edges[-1]),mu_bound=bool(abs(sol.x[1])>.64))
def main(a):
    data=Path(a.data);out=Path(a.output);plots=Path(a.plots);out.mkdir(parents=True,exist_ok=True);plots.mkdir(parents=True,exist_ok=True)
    amp=[];base=[];ident=[];sources={}
    for name,n in [('cuoreTraining.h5',4500),('cuoreTest.h5',500)]:
        sources[name]=sha(data/name)
        with h5py.File(data/name) as h:
            assert int(h.attrs['cleanPulseCount'])==n
            if 'Labels' in h:assert np.all(h['Labels'][:n]==0)
            else:assert h.attrs['rowOrder']=='clean rows followed by pileup rows'
            amp.extend(h['normalizationScale'][:n,0]);base.extend(h['normalizationOffset'][:n,0])
            ident.extend([dict(source_file=name,source_row=i,event_id=int(np.asarray(h['eventId'][i]).item())) for i in range(n)])
    amp=np.array(amp);base=np.array(base);group=base>-70000
    assert len(amp)==5000
    candidates=[];histrows=[]
    for width in [25,50,100]:
        edges=np.arange(5000,30000+width,width);centers=(edges[1:]+edges[:-1])/2
        for name,mask in [('all',np.ones(5000,dtype=bool)),('A',group),('B',~group)]:
            hist,_=np.histogram(amp[mask],edges);sm=gaussian_filter1d(hist.astype(float),1.)
            peaks,pr=find_peaks(sm,prominence=3,distance=max(2,int(250/width)))
            for i,k in enumerate(peaks):
                candidates.append(dict(group=name,bin_width_adc=width,center_adc=float(centers[k]),
                    smoothed_prominence=float(pr['prominences'][i]),bin_count=int(hist[k])))
            if width==50:
                histrows.extend(dict(group=name,low_adc=float(edges[k]),high_adc=float(edges[k+1]),count=int(v)) for k,v in enumerate(hist))
    fitted=[]
    for name,E,seed,window,label in LINES:
        r=peakfit(amp[group],seed,window)
        fitted.append(dict(line=name,reference_energy_kev=E,label=label,**r,used_as_anchor=name in ['Tl208','Tl208_high']))
    anchors=[r for r in fitted if r['used_as_anchor']]
    assert all(r['fit_success'] and not r['mu_bound'] for r in anchors)
    # Scale amplitude by 10,000 to condition the zero-intercept quadratic.
    x=np.array([r['mu_adc']/10000 for r in anchors]);E=np.array([r['reference_energy_kev'] for r in anchors])
    c=np.linalg.solve(np.column_stack([x,x*x]),E)
    energy=lambda v:c[0]*(v/10000)+c[1]*(v/10000)**2
    for r in fitted:
        r['mapped_energy_kev']=float(energy(r['mu_adc']))
        r['residual_kev']=r['mapped_energy_kev']-r['reference_energy_kev']
    monotone=float(np.min(c[0]+2*c[1]*np.array([amp[group].min(),amp[group].max()])/10000))>0
    assert monotone
    # Independent cross-checks at 511, 911 and 969; these are not calibration anchors.
    checks=[r for r in fitted if r['line'] in ['annihilation','Ac228','Ac228_blend']]
    accepted=all(r['fit_success'] and not r['mu_bound'] and abs(r['residual_kev'])<10 for r in checks)
    rows=[]
    for i,r in enumerate(ident):
        rows.append(dict(**r,amplitude_adc=float(amp[i]),baseline_adc=float(base[i]),baseline_group='A' if group[i] else 'B',
                         reference_energy_group_A_kev=float(energy(amp[i])) if group[i] and accepted else ''))
    csvwrite(out/'events.csv',rows);csvwrite(out/'histogram_adc.csv',histrows)
    csvwrite(out/'peak_candidates.csv',candidates);csvwrite(out/'peak_assignments_group_A.csv',fitted)
    meta=dict(n=5000,n_A=int(group.sum()),n_B=int((~group).sum()),source_sha256=sources,
        group_rule='A: normalizationOffset > -70000 ADC; B: <= -70000; empirical empty-gap split, NOT run IDs',
        baseline_gap_events=int(np.sum((base>-70000)&(base<-65000))),amplitude='normalizationScale, raw maximum minus pretrigger mean',
        reference_energy_available=accepted,calibration=dict(formula='E_keV=c1*(A/10000)+c2*(A/10000)^2',c1=float(c[0]),c2=float(c[1]),
            anchors=[583.191,2614.511],zero_intercept_assumed=True,applies_only_to='group A',official_calibration=False,
            max_independent_peak_residual_kev=max(abs(r['residual_kev']) for r in checks)),
        warnings=['Not gain-stabilized or OF amplitudes. No official calibration/energy fields supplied.',
                  'Group B has shifted peak positions; no energy scale is assigned to B.',
                  'Peak labels inferred from pattern; source deployment details are not supplied for this release.',
                  '969-keV feature may include the nearby 965-keV line; Gaussian centroids are descriptive.',
                  'The approximately 1590-keV structure is not a resolved double escape peak; a neighboring 1588-keV line can contribute.',
                  'Counts are from a selected classification sample, not exposure-normalized event rates.'],
        sources=['https://arxiv.org/html/2607.02548v1','https://arxiv.org/html/2510.25720v1',
                 'https://cuore.lngs.infn.it/sites/default/files/2021_Huang_search_0nbb_CUORE-CUPID_compressed.pdf',
                 'https://www.nndc.bnl.gov/ensnds/212/Po/beta_decay_60.55_m.pdf'])
    dump(out/'summary.json',meta)
    fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True)
    bins=np.arange(5000,30050,50)
    axs[0].hist(amp,bins=bins,histtype='step',color='#314e69',lw=1.2,label='全部5000个单事件')
    axs[0].set_ylabel('事件数 / 50 ADC');axs[0].legend()
    for mask,label,color in [(group,'基线组A：2716例','#337db4'),(~group,'基线组B：2284例','#d68c34')]:
        axs[1].hist(amp[mask],bins=bins,histtype='step',color=color,lw=1,label=label)
    axs[1].set_yscale('log');axs[1].set_ylim(.7,400);axs[1].set_ylabel('事件数 / 50 ADC（对数）')
    axs[1].set_xlabel('原始幅值（ADC，尚未做增益稳定）');axs[1].legend()
    for r in fitted:
        if r['line'] in ['Tl208','Ac228','Ac228_blend','Tl208_high']:
            axs[0].axvline(r['mu_adc'],color='#8a6551',ls='--',alpha=.6)
            y={'Tl208':175,'Ac228':145,'Ac228_blend':105,'Tl208_high':100}[r['line']]
            axs[0].annotate(f"{r['reference_energy_kev']:.0f} keV候选",xy=(r['mu_adc'],y),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10,rotation=0)
    for ax in axs:ax.grid(alpha=.18);ax.set_xlim(5000,30000)
    axs[0].set_ylim(0,215)
    fig.suptitle('CUORE公开数据的幅值谱：可见多组标定峰，部分峰随基线组移动',fontsize=15)
    fig.tight_layout();save(fig,plots,'01_adc_spectrum')
    if accepted:
        edges=np.arange(450,2704,4);hist,_=np.histogram(energy(amp[group]),edges)
        csvwrite(out/'histogram_reference_energy_A.csv',[dict(low_kev=float(edges[k]),high_kev=float(edges[k+1]),count=int(v)) for k,v in enumerate(hist)])
        fig,ax=plt.subplots(figsize=(14,5.4))
        ax.stairs(hist,edges,color='#337db4',lw=1.2)
        for i,r in enumerate(fitted):
            E=r['reference_energy_kev'];anchor=r['used_as_anchor']
            ax.axvline(E,color='#be6f25' if anchor else '#71818e',ls='--',lw=.8,alpha=.8)
            yy=[55,145,35,90,45,35,25,85][i]
            label=r['label']
            ax.text(E+8,.97,label,transform=ax.get_xaxis_transform(),rotation=90,va='top',ha='left',fontsize=9,color='#95551c' if anchor else '#344b5c',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
        ax.set_yscale('log');ax.set_ylim(.7,360);ax.set_xlim(450,2700)
        ax.set_xlabel('参考能量（keV；两峰暂定标度，仅适用于基线组A）');ax.set_ylabel('事件数 / 4 keV')
        ax.grid(alpha=.2);ax.set_title('2716个单事件的参考能谱：583/2615两锚点；其余峰用于交叉检查',fontsize=14)
        fig.tight_layout();save(fig,plots,'02_reference_energy_spectrum_A')
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    axs[0].hist(base,bins=np.arange(-80000,-61000,200),histtype='step',color='#314e69')
    axs[0].axvline(-70000,ls='--',color='#777');axs[0].set_xlabel('原始基线（ADC）');axs[0].set_ylabel('事件数 / 200 ADC');axs[0].set_title('基线存在明显空隙，可作描述性分组')
    for m,label,col in [(group,'组A','#337db4'),(~group,'组B','#d68c34')]:
        sel=m&(amp>27500)
        axs[1].scatter(base[sel],amp[sel],s=12,alpha=.65,color=col,label=label)
    axs[1].set_xlabel('原始基线（ADC）');axs[1].set_ylabel('原始幅值（ADC）');axs[1].set_title('高幅值峰区域：幅值移动与基线相关');axs[1].legend()
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();save(fig,plots,'03_baseline_groups')
    assert sum(r['count'] for r in histrows if r['group']=='all')==5000
    assert sources=={f:sha(data/f) for f in sources}
    dump(out/'verification.json',dict(passed=True,n_unique_events=len({(r['source_file'],r['source_row']) for r in rows}),
        adc_hist_count=5000,group_counts=[int(group.sum()),int((~group).sum())],reference_scale_monotonic=monotone,
        independent_peak_check=accepted,source_hashes_unchanged=True))
    print(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(fitted,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['data','output','plots']:p.add_argument('--'+k,required=True)
    main(p.parse_args())

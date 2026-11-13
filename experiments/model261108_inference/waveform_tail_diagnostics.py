"""Read-only waveform / tail diagnostics for the original evaluation set.

Saved fit parameters are reconstructed, not refitted (only noiseless templates
are fitted as controls). Gaussian mixtures describe shapes, not physical classes.
Energy-defined groups are visual diagnostics and never remove production events.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.basics.fit import fit_pulse
from experiments.model261108_inference.full_comparison import digest,json_safe,save_figure

CASES=[('1p382mV',2),('1p382mV',5),('7p5mV',10),('7p5mV',15),
       ('7p5mV',20),('7p5mV',25),('7p5mV',30),('7p5mV',50)]
PARAMS=['baseline','amp_raw','onset','t_rise','alpha','t_fall1','t_fall2']
COLORS={0:'#aeb3b9',1:'#267ac4',2:'#df7539'}
GROUPS={0:'Transition / other',1:'Peak core',2:'Low-energy tail'}

def load_fit(path,field='denoised'):
    with h5py.File(path,'r') as f:
        arrays={k[:-len('_'+field)]:v[:].astype(float) for k,v in f.items()
                if isinstance(v,h5py.Dataset) and k.endswith('_'+field)}
        noise={k:v[:].astype(float) for k,v in f['noise_params'].items()}
        true=float(f.attrs['true_amp_V'])
    return arrays,noise,true

def reconstruct(f,ids,t):
    ids=np.asarray(ids,int)
    delta=t[None,:]-f['onset'][ids,None]
    dt=np.maximum(delta,0)
    shape=(-np.exp(-dt/f['t_rise'][ids,None])
           +f['alpha'][ids,None]*np.exp(-dt/f['t_fall1'][ids,None])
           +(1-f['alpha'][ids,None])*np.exp(-dt/f['t_fall2'][ids,None]))
    return f['baseline'][ids,None]+f['amp_raw'][ids,None]*shape*(delta>=0)

def mixture(x,components):
    x=np.asarray(x,float)
    center=float(np.mean(x));scale=float(np.std(x))
    z=(x-center)/scale
    best=None
    starts=([np.array([.5])] if components==1 else
            [np.linspace(.15,.85,components),np.linspace(.05,.75,components),
             np.linspace(.3,.92,components)])
    for quantiles in starts:
        mu=np.quantile(z,quantiles)
        var=np.linspace(.8,.04,components) if components>1 else np.ones(1)
        w=np.ones(components)/components
        prev=-np.inf
        for iteration in range(1200):
            lp=np.log(w)[None,:]-.5*(np.log(2*np.pi*var)[None,:]+(z[:,None]-mu[None,:])**2/var[None,:])
            den=logsumexp(lp,axis=1);ll=float(den.sum())
            post=np.exp(lp-den[:,None])
            nk=post.sum(axis=0)+1e-12
            w=nk/nk.sum();mu=(post*z[:,None]).sum(axis=0)/nk
            var=np.maximum((post*(z[:,None]-mu)**2).sum(axis=0)/nk,1e-6)
            if np.isfinite(prev) and abs(ll-prev)<1e-7*len(x):break
            prev=ll
        lp=np.log(w)[None,:]-.5*(np.log(2*np.pi*var)[None,:]+(z[:,None]-mu)**2/var)
        ll=float(logsumexp(lp,axis=1).sum()-len(x)*np.log(scale))
        idx=np.argsort(mu)
        candidate={'n_components':components,'means':(mu[idx]*scale+center).tolist(),
                   'sigmas':(np.sqrt(var[idx])*scale).tolist(),'weights':w[idx].tolist(),
                   'log_likelihood':ll,'bic':float(-2*ll+(3*components-1)*np.log(len(x))),
                   'iterations':iteration+1}
        if best is None or ll>best['log_likelihood']:best=candidate
    return best

def posterior(x,mix):
    mu=np.array(mix['means']);sd=np.array(mix['sigmas']);w=np.array(mix['weights'])
    lp=np.log(w)[None,:]-.5*((x[:,None]-mu)/sd)**2-np.log(sd*np.sqrt(2*np.pi))
    return np.exp(lp-logsumexp(lp,axis=1)[:,None])

def classify(x,mix):
    mu=mix['means'][-1];sd=mix['sigmas'][-1]
    post=posterior(x,mix)[:,-1]
    groups=np.zeros(len(x),np.uint8)
    groups[(np.abs(x-mu)<=2*sd)&(post>=.9)]=1
    groups[x<mu-3*sd]=2
    return groups,{'mu_peak':mu,'sigma_peak':sd,'tail_below':mu-3*sd,
                   'core_half_width':2*sd,'core_posterior_min':.9}

def scatter(ax,x,y,groups,xlabel,ylabel,log=False):
    for g in (0,1,2):
        idx=groups==g
        ax.scatter(x[idx],y[idx],s=4,c=COLORS[g],alpha=.32,edgecolors='none',
                   rasterized=True,label=f'{GROUPS[g]} (N={idx.sum()})')
    ax.set(xlabel=xlabel,ylabel=ylabel)
    if log:ax.set_yscale('log')
    ax.grid(alpha=.18)

def paths(args,level,e):
    name=f'eval_efficiency_{e}keV.h5'
    return (args.data_root/f'eval_data_holdout/efficiency_{level}/{name}',
            args.data_root/f'model261108_inference/efficiency_{level}/{name}',
            args.notes_root/f'results/fit_results/model261108_nonoise/efficiency_{level}/fit_efficiency_{e}keV.h5',
            args.notes_root/f'results/fit_results/{level}_zeroinit/fit_efficiency_{e}keV.h5')

def features(raw_path,out_path,f,groups,clean):
    n=len(groups);t=np.arange(len(clean))/1000.
    template=clean-np.median(clean[:500])
    peak=float(template.max());template_center=template-template.mean()
    norm=float(template_center@template_center)
    freq=np.fft.rfftfreq(len(clean),.001)
    bands=[(0,1),(1,20),(20,70),(70,501)]
    result={k:np.empty(n) for k in (
        'input_scale_V','input_baseline_V','input_baseline_sigma_V','input_noise_projection_ratio',
        'input_noise_peakregion_V','input_noise_preonset_V',
        'output_baseline_V','output_baseline_sigma_V','output_projection_ratio',
        'output_peak_ratio','output_area_ratio','output_peak_time_s','output_rms_to_clean_V',
        'fit_rms_V','fit_peakregion_rms_V','fit_minus_direct_peak_ratio')}
    for low,high in bands:result[f'input_noise_power_{low}_{high}_Hz_fraction']=np.empty(n)
    sums={g:{k:np.zeros(len(clean)) for k in ('raw','output','fit','noise')} for g in (0,1,2)}
    with h5py.File(raw_path,'r') as raw,h5py.File(out_path,'r') as output:
        assert output.attrs['source_sha256']==digest(raw_path)
        assert np.array_equal(output['source_row'][:],np.arange(n))
        assert output['waveforms_denoised'].shape==(n,10000)
        for start in range(0,n,128):
            end=min(start+128,n);ids=np.arange(start,end);sl=slice(start,end)
            x=raw['waveforms_noisy'][sl].astype(float)
            y=output['waveforms_denoised'][sl].astype(float)
            noise=x-clean
            fit=reconstruct(f,ids,t)
            b=np.median(y[:,:500],axis=1)
            sigma=np.std(y[:,:500],axis=1)
            yy=y-b[:,None]
            direct=yy.max(axis=1)
            result['input_scale_V'][sl]=np.max(np.abs(x),axis=1)
            result['input_baseline_V'][sl]=np.median(x[:,:500],axis=1)
            result['input_baseline_sigma_V'][sl]=np.std(x[:,:500],axis=1)
            result['input_noise_projection_ratio'][sl]=noise@template_center/norm
            result['input_noise_peakregion_V'][sl]=noise[:,1500:2200].mean(axis=1)
            result['input_noise_preonset_V'][sl]=noise[:,1000:1500].mean(axis=1)
            result['output_baseline_V'][sl]=b
            result['output_baseline_sigma_V'][sl]=sigma
            result['output_projection_ratio'][sl]=y@template_center/norm
            result['output_peak_ratio'][sl]=direct/peak
            result['output_area_ratio'][sl]=np.trapezoid(yy[:,1500:6000],axis=1)/np.trapezoid(template[1500:6000])
            result['output_peak_time_s'][sl]=np.argmax(yy,axis=1)/1000.
            result['output_rms_to_clean_V'][sl]=np.sqrt(np.mean((y-clean)**2,axis=1))
            result['fit_rms_V'][sl]=np.sqrt(np.mean((y-fit)**2,axis=1))
            result['fit_peakregion_rms_V'][sl]=np.sqrt(np.mean((y[:,1450:2300]-fit[:,1450:2300])**2,axis=1))
            result['fit_minus_direct_peak_ratio'][sl]=f['amp'][sl]/peak-direct/peak
            fft=np.fft.rfft(noise-noise.mean(axis=1,keepdims=True),axis=1)
            power=np.abs(fft)**2
            total=power[:,1:].sum(axis=1)
            for low,high in bands:
                mask=(freq>low)&(freq<=high)
                result[f'input_noise_power_{low}_{high}_Hz_fraction'][sl]=power[:,mask].sum(axis=1)/total
            for g in sums:
                mask=groups[sl]==g
                sums[g]['raw']+=x[mask].sum(axis=0)
                sums[g]['output']+=y[mask].sum(axis=0)
                sums[g]['fit']+=fit[mask].sum(axis=0)
                sums[g]['noise']+=noise[mask].sum(axis=0)
    # Stored parameters are float32. Reconstruction may differ slightly from the
    # original optimizer curve; this check is numerical, not a scientific gate.
    expected=np.sqrt(f['chi2']*(10000-7)/10000)*result['output_baseline_sigma_V']
    reconstruction_error=float(np.max(np.abs(result['fit_rms_V']-expected)/np.maximum(expected,1e-12)))
    assert reconstruction_error<.005,reconstruction_error
    assert np.allclose(result['output_peak_ratio'],f['amp_maxbase']/peak,rtol=2e-5,atol=2e-5)
    means={g:{key:value/max(1,int((groups==g).sum())) for key,value in sums[g].items()} for g in sums}
    return result,means,reconstruction_error

def plots(args,level,e,f,old,noise,feat,groups,mixes,definition,clean,k,means):
    dest=args.output/f'{level}_{e}keV';dest.mkdir(exist_ok=True)
    x=f['amp']*k
    title=f'{level.replace("p",".")}, {e} keV; N={len(x)}'
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    bins=np.linspace(min(x.min(),k*f['amp_maxbase'].min()),max(x.max(),k*f['amp_maxbase'].max()),120)
    axes[0].hist(x,bins=bins,density=True,histtype='step',color='black',lw=1,label='Observed tri-exp amplitudes')
    grid=np.linspace(bins[0],bins[-1],500)
    mix=mixes[1]
    components=[]
    for i,(mu,sd,w) in enumerate(zip(mix['means'],mix['sigmas'],mix['weights'])):
        component=w*np.exp(-.5*((grid-mu)/sd)**2)/(sd*np.sqrt(2*np.pi))
        components.append(component)
        axes[0].plot(grid,component,ls='--',color=COLORS[2 if i==0 else 1],
                     label=f'2-Gaussian component {i+1} (descriptive)')
    axes[0].plot(grid,np.sum(components,axis=0),color='#7a4d9f',lw=1,label='2-component sum')
    axes[0].axvline(definition['tail_below'],color=COLORS[2],ls=':',label='Diagnostic tail boundary')
    axes[0].set(xlabel='Reconstructed energy (keV)',ylabel='Density (1/keV)',title='Shape description; not two physical populations')
    axes[0].legend(fontsize=6.7)
    for name,values,color in [('Tri-exp peak',x,'black'),('Direct max - baseline',f['amp_maxbase']*k,'#139b73')]:
        axes[1].hist(values,bins=bins,density=True,histtype='step',color=color,label=name)
    axes[1].set(xlabel='Amplitude × same k (keV)',ylabel='Density (1/keV)',yscale='log',
                title='Tail persists without tri-exponential fitting')
    axes[1].legend(fontsize=8)
    scatter(axes[2],x,feat['output_projection_ratio'],groups,'Tri-exp energy (keV)','Output / clean template projection')
    axes[2].set_title('Independent waveform diagnostic (not OF)')
    fig.suptitle(title)
    fig.supxlabel('Groups only label plots: peak core = ±2σ of high-mean component and posterior ≥0.9; tail = below μ−3σ. No production cut.',fontsize=8)
    for ax in axes:ax.grid(alpha=.18)
    save_figure(fig,dest/'spectrum_and_estimators')

    specs=[('t_rise',1000,'Rise time (ms)',False),('t_fall1',1000,'Fast decay (ms)',False),
           ('t_fall2',1000,'Slow decay (ms)',False),('alpha',1,'Fast-decay weight α',False),
           ('onset',1,'Fitted onset (s)',False),('baseline',1e6,'Fitted baseline (µV)',False),
           ('chi2',1,'Waveform χ²/ndf',True)]
    fig,axes=plt.subplots(3,3,figsize=(14,10),layout='constrained')
    for ax,(name,scale,label,log) in zip(axes.flat,specs):
        scatter(ax,x,f[name]*scale,groups,'Tri-exp energy (keV)',label,log)
        rho=float(spearmanr(x,f[name]).statistic)
        ax.set_title(f'Spearman ρ = {rho:.3f}',fontsize=10)
    scatter(axes.flat[7],x,100*feat['fit_rms_V']/float(clean.max()),groups,'Tri-exp energy (keV)','Fit RMS / clean peak (%)',True)
    scatter(axes.flat[8],x,100*feat['output_rms_to_clean_V']/float(clean.max()),groups,'Tri-exp energy (keV)','Output error RMS / clean peak (%)',True)
    axes.flat[0].legend(fontsize=7,loc='best',markerscale=2)
    fig.suptitle(title+' — per-window fitting parameters')
    fig.supxlabel('All 5,000 windows are plotted. Fitted time constants describe output shapes; they are not independent physical measurements.',fontsize=9)
    save_figure(fig,dest/'energy_vs_fit_parameters')

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(name,label,scale,log) in zip(axes.flat,[
        ('t_rise','Rise time (ms)',1000,False),('t_fall1','Fast decay (ms)',1000,False),
        ('t_fall2','Slow decay (ms)',1000,False),('alpha','Fast-decay weight α',1,False),
        ('onset','Onset (s)',1,False),('chi2','Waveform χ²/ndf',1,True)]):
        v=f[name]*scale
        edges=np.geomspace(max(v.min(),1e-6),v.max()*1.001,70) if log else np.linspace(v.min(),v.max()+1e-9,70)
        for g in (1,2):
            if (groups==g).sum()>0:
                ax.hist(v[groups==g],bins=edges,density=True,histtype='step',color=COLORS[g],label=GROUPS[g])
        ax.set(xlabel=label,ylabel='Within-group density')
        if log:ax.set_xscale('log')
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle(title+' — core / tail parameter distributions')
    fig.supxlabel('Each group is normalized separately. Display groups are defined from the observed energy spectrum, not a selection for resolution.',fontsize=9)
    save_figure(fig,dest/'parameter_distributions')

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    pairs=[(f['t_rise']*1000,f['t_fall1']*1000,'Rise time (ms)','Fast decay (ms)'),
           (f['t_fall1']*1000,f['t_fall2']*1000,'Fast decay (ms)','Slow decay (ms)'),
           (f['onset'],f['t_rise']*1000,'Fitted onset (s)','Rise time (ms)'),
           (feat['output_peak_ratio'],f['t_rise']*1000,'Direct peak / clean peak','Rise time (ms)'),
           (feat['input_scale_V']*1000,feat['output_peak_ratio'],'Input normalization scale (mV)','Direct peak / clean peak'),
           (feat['input_noise_projection_ratio'],feat['output_peak_ratio'],'Input noise projection / clean','Direct peak / clean peak')]
    for ax,(xx,yy,xl,yl) in zip(axes.flat,pairs):scatter(ax,xx,yy,groups,xl,yl)
    axes.flat[0].legend(fontsize=7,markerscale=2)
    fig.suptitle(title+' — shape and input relationships')
    save_figure(fig,dest/'parameter_pairs')

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    noise_features=[(noise['pt_frac'],'PT fraction proxy'),
                    (noise['ac_frac'],'AC fraction proxy'),
                    (feat['input_noise_power_0_1_Hz_fraction'],'Noise power fraction: 0–1 Hz'),
                    (feat['input_noise_projection_ratio'],'Noise projection / clean'),
                    (feat['input_noise_peakregion_V']*1000,'Noise mean, 1.5–2.2 s (mV)'),
                    (feat['input_scale_V']*1000,'Normalization scale (mV)')]
    for ax,(v,label) in zip(axes.flat,noise_features):
        scatter(ax,v,x,groups,label,'Tri-exp energy (keV)')
        ax.set_title(f'Spearman ρ = {spearmanr(v,x).statistic:.3f}',fontsize=10)
    axes.flat[0].legend(fontsize=7,markerscale=2)
    fig.suptitle(title+' — association with input noise')
    fig.supxlabel('PT/AC fractions are stored amplitude-squared proxies, not measured spectral power fractions. Associations do not establish causal mechanisms.',fontsize=9)
    save_figure(fig,dest/'input_noise_relationships')

    t=np.arange(len(clean))/1000
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    peak=float(clean.max()-np.median(clean[:500]))
    for g in (1,2):
        if (groups==g).sum()==0:continue
        mean=means[g]
        axes[0,0].plot(t,mean['output']/peak,color=COLORS[g],label=GROUPS[g])
        axes[0,0].plot(t,mean['fit']/peak,color=COLORS[g],ls='--',lw=1)
        axes[0,1].plot(t,mean['output']/np.max(mean['output']),color=COLORS[g],label=GROUPS[g])
        axes[1,0].plot(t,(mean['output']-clean)/peak,color=COLORS[g],label=GROUPS[g])
        axes[1,1].plot(t,mean['noise']/peak,color=COLORS[g],label=GROUPS[g])
    axes[0,0].plot(t,clean/peak,'k:',label='Clean')
    axes[0,1].plot(t,clean/clean.max(),'k:',label='Clean')
    for ax in axes.flat:ax.set_xlim(1.2,3.5);ax.grid(alpha=.2);ax.set_xlabel('Time (s)');ax.legend(fontsize=8)
    axes[0,0].set(title='Mean output (solid) and fit (dashed)',ylabel='Amplitude / clean peak')
    axes[0,1].set(title='Mean shapes normalized to their own peaks',ylabel='Relative amplitude')
    axes[1,0].set(title='Mean output minus clean',ylabel='Residual / clean peak')
    axes[1,1].set(title='Input noise, conditioned on output-energy group',ylabel='Noise / clean peak')
    fig.suptitle(title+' — group-average waveforms')
    fig.supxlabel('Group conditioning can create a nonzero mean input noise. This is an association, not evidence of a nonzero generator mean.',fontsize=9)
    save_figure(fig,dest/'mean_waveforms')

    # Examples chosen by fixed quantiles within diagnostic groups, not visual quality.
    selections=[]
    for g,q,label in [(1,.5,'Core median'),(2,.9,'Tail upper 10%'),(2,.5,'Tail median'),(2,.1,'Tail lower 10%')]:
        ids=np.flatnonzero(groups==g)
        if not len(ids):continue
        target=np.quantile(x[ids],q);idx=int(ids[np.argmin(abs(x[ids]-target))])
        selections.append({'row':idx,'label':label,'energy_keV':float(x[idx]),'group':int(g),'quantile':q})
    waveform_examples(args,level,e,f,clean,selections,dest/'waveform_examples',peak)
    return selections

def waveform_examples(args,level,e,f,clean,selections,destination,peak):
    raw_path,out_path,_,_=paths(args,level,e)
    t=np.arange(len(clean))/1000
    fig,axes=plt.subplots(4,len(selections),figsize=(4*len(selections),11),layout='constrained',squeeze=False)
    with h5py.File(raw_path) as raw,h5py.File(out_path) as output:
        for c,s in enumerate(selections):
            i=s['row'];y=output['waveforms_denoised'][i].astype(float)
            x=raw['waveforms_noisy'][i].astype(float);fit=reconstruct(f,[i],t)[0]
            for r in (0,1):
                axes[r,c].plot(t,y/peak,color='#e17736',lw=1,label='New model output')
                axes[r,c].plot(t,fit/peak,'k--',lw=1,label='Saved tri-exp fit')
                axes[r,c].plot(t,clean/peak,color='#139b73',ls=':',lw=1.5,label='Clean truth')
            axes[0,c].set_xlim(0,6);axes[1,c].set_xlim(1.3,2.4)
            axes[2,c].plot(t,(y-fit)/peak,color='#8e4da0',label='Output − fit')
            axes[2,c].plot(t,(y-clean)/peak,color='#777777',alpha=.65,label='Output − clean')
            axes[2,c].set_xlim(1.2,3.5)
            axes[3,c].plot(t,x/peak,color='#999999',lw=.5,label='Input noisy')
            axes[3,c].plot(t,clean/peak,color='#139b73',lw=1.5,label='Clean truth')
            axes[3,c].set_xlim(1.2,2.7)
            axes[0,c].set_title(f"{s['label']}; row {i}\nE_fit={s['energy_keV']:.3f} keV",fontsize=10)
            for r in range(4):
                axes[r,c].grid(alpha=.2);axes[r,c].set_xlabel('Time (s)')
                axes[r,c].set_ylabel('Amplitude / clean peak' if r!=2 else 'Residual / clean peak')
                axes[r,c].legend(fontsize=6.5,loc='best')
    fig.suptitle(f'{level}, {e} keV — same-window waveform, fit and truth')
    fig.supxlabel('Source rows are zero-based. Saved fit parameters reconstructed without refitting; examples selected by energy quantiles within groups.',fontsize=9)
    save_figure(fig,destination)

def high_examples(args):
    fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    metrics=[]
    for col,level in enumerate(('1p382mV','7p5mV')):
        raw_path,out_path,fit_path,_=paths(args,level,2615)
        f,_,_=load_fit(fit_path)
        with h5py.File(raw_path) as raw:clean=raw['clean_template'][:].astype(float)
        cf=fit_pulse(clean,n_pulses=1,fs=1000.)
        assert cf.success
        t=np.arange(10000)/1000.;curve=cf.model_fn(t,*cf.params)
        with h5py.File(out_path) as output:
            for j,q in enumerate((.5,.95)):
                c=2*col+j;idx=int(np.argmin(abs(f['chi2']-np.quantile(f['chi2'],q))))
                y=output['waveforms_denoised'][idx].astype(float);fit=reconstruct(f,[idx],t)[0]
                sigma=np.std(y[:500]);rms=np.sqrt(np.mean((y-fit)**2))
                for r in (0,1):
                    axes[r,c].plot(t,y,color='#e17736',label='New model output',lw=1)
                    axes[r,c].plot(t,fit,'k--',label='Saved tri-exp fit',lw=1)
                    axes[r,c].plot(t,clean,color='#139b73',ls=':',label='Clean truth',lw=1.4)
                axes[0,c].set_xlim(0,8);axes[1,c].set_xlim(1.4,2.05)
                axes[2,c].plot(t,(y-fit)*1000,color='#8e4da0',label='Output − fit')
                axes[2,c].plot(t,(clean-curve)*1000,color='#139b73',ls='--',label='Clean − its tri-exp fit')
                axes[2,c].axhspan(-sigma*1000,sigma*1000,color='#aaaaaa',alpha=.25,label='± baseline σ')
                axes[2,c].set_xlim(1.3,3.5)
                axes[0,c].set_title(f'{level}; row {idx}; χ² P{100*q:.0f}\nχ²/ndf={f["chi2"][idx]:.1f}',fontsize=10)
                for r in range(3):
                    axes[r,c].set_xlabel('Time (s)');axes[r,c].set_ylabel('Voltage (V)' if r<2 else 'Residual (mV)')
                    axes[r,c].grid(alpha=.2);axes[r,c].legend(fontsize=6.5)
                metrics.append({'noise':level,'energy_keV':2615,'row':idx,'chi2_quantile':q,
                    'chi2_ndf':f['chi2'][idx],'baseline_sigma_mV':sigma*1000,'fit_rms_mV':rms*1000,
                    'clean_fit_rms_mV':float(np.sqrt(np.mean((clean-curve)**2))*1000),
                    'sources':[{'path':str(p),'sha256':digest(p)} for p in (raw_path,out_path,fit_path)]})
    fig.suptitle('2615 keV: waveform fits and residual structure at two noise levels')
    fig.supxlabel('Only examples are displayed; the existing quality gate is unchanged. Clean-template fitting is a control, not a replacement model.',fontsize=9)
    save_figure(fig,args.output/'high_energy_fit_examples')
    return metrics

def summarize_features(f,noise,feat,groups,energy):
    data={**{'fit_'+k:v for k,v in f.items() if len(v)==len(groups)},
          **{'noise_'+k:v for k,v in noise.items()},**feat}
    rows=[]
    for name,x in data.items():
        if np.all(np.isfinite(x)) and np.std(x)>1e-20:
            rho=float(spearmanr(energy,x).statistic)
        else:rho=None
        rec={'feature':name,'spearman_rho_with_energy':rho}
        for g,label in [(1,'core'),(2,'tail')]:
            mask=groups==g
            rec[label+'_median']=float(np.nanmedian(x[mask])) if mask.any() else None
            rec[label+'_q10_q90']=np.nanpercentile(x[mask],[10,90]).tolist() if mask.any() else None
        rows.append(rec)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--notes-root',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    comparison=json.loads(args.comparison.read_text())
    report={'code_sha256':digest(__file__),'comparison_sha256':digest(args.comparison),
            'group_rule':'Energy-defined diagnostics only: 2-component Gaussian mixture high-mean peak; core ±2sigma and responsibility>=.9; tail below mu-3sigma.',
            'limitations':['Mixtures are descriptive; no physical two-population claim.',
                           'Correlations of parameters from the same fit are not independent validation.',
                           'Direct peak and template projection use the stored output waveform; projection is unweighted and is not OF.',
                           'Stored PT/AC fractions are amplitude-squared proxies.',
                           'No inference rerun, no new OF, no change of waveform-quality gate.'],
            'cases':{},'high_energy_examples':high_examples(args)}
    for level,e in CASES:
        key=f'{level}_{e}keV';print('START',key,flush=True)
        raw_path,out_path,fit_path,old_path=paths(args,level,e)
        f,noise,true=load_fit(fit_path);old,_,_=load_fit(old_path)
        with h5py.File(raw_path) as h:clean=h['clean_template'][:].astype(float)
        assert np.all(f['success']) and len(f['amp'])==5000 and np.all(np.isfinite(f['amp']))
        k=comparison['groups'][level]['calibrations']['new_nonoise']['k_keV_per_amplitude']
        energy=k*f['amp']
        mixes=[mixture(energy,n) for n in (1,2,3)]
        groups,definition=classify(energy,mixes[1])
        feat,means,error=features(raw_path,out_path,f,groups,clean)
        selections=plots(args,level,e,f,old,noise,feat,groups,mixes,definition,clean,k,means)
        stats=summarize_features(f,noise,feat,groups,energy)
        counts={GROUPS[g]:int((groups==g).sum()) for g in GROUPS}
        sources=[{'path':str(pth),'sha256':digest(pth)} for pth in (raw_path,out_path,fit_path,old_path)]
        rec={'sources':sources,'calibration_k':k,'counts':counts,'mixtures':mixes,'group_definition':definition,
             'features':stats,'selected_examples':selections,'reconstructed_fit_rms_max_relative_difference':error,
             'fit_vs_direct_peak_spearman':float(spearmanr(f['amp'],f['amp_maxbase']).statistic),
             'fit_vs_projection_spearman':float(spearmanr(f['amp'],feat['output_projection_ratio']).statistic),
             'old_vs_new_fit_amplitude_spearman':float(spearmanr(old['amp'],f['amp']).statistic),
             'rise_upper_bound_count':int((f['t_rise']>=.2*(1-1e-5)).sum())}
        report['cases'][key]=rec
        dest=args.output/key
        np.savez_compressed(dest/'per_event_features.npz',source_row=np.arange(len(groups)),group=groups,
                            energy_keV=energy,old_amp=old['amp'],**{'fit_'+k:v for k,v in f.items()},
                            **{'noise_'+k:v for k,v in noise.items()},**feat)
        with (dest/'feature_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
        (args.output/'diagnostics.json').write_text(json.dumps(json_safe(report),indent=2),encoding='utf-8')
        print(key,'groups',counts,'rho direct',rec['fit_vs_direct_peak_spearman'],
              'BIC',[round(m['bic'],1) for m in mixes],flush=True)
    report['complete']=True
    (args.output/'diagnostics.json').write_text(json.dumps(json_safe(report),indent=2),encoding='utf-8')
    print('TAIL_DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':main()

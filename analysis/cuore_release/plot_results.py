"""Render CUORE fit tables without rerunning fits. Full ranges, no subsampling."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fits',required=True); p.add_argument('--plots',required=True)
    a=p.parse_args(); out=Path(a.plots)
    rows=[json.loads(s) for s in (Path(a.fits)/'fits.jsonl').read_text().splitlines()]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    groups=[(1,False,'Pileup, flagged','#c77c39','x'),(1,True,'Pileup, screening pass','#c77c39','.'),
            (0,False,'Single, flagged','#687785','x'),(0,True,'Single, screening pass','#0066bb','o')]
    for tag,sel in [('all_windows',rows),('single_pulses',[r for r in rows if not r['label']])]:
        fig,axs=plt.subplots(1,3,figsize=(15,5.6),layout='constrained')
        for ax,key,title in zip(axs,['tau_r','tau_fast','tau_slow'],
                               [r'Rise: $\tau_r$',r'Fast decay: $\tau_{d1}$',r'Slow decay: $\tau_{d2}$']):
            for cls,good,label,c,marker in groups:
                rr=[r for r in sel if r['success'] and r['label']==cls and r['diagnostic_pass']==good]
                if rr:
                    ax.scatter([r['amplitude_adc']/1000 for r in rr],[r[key][r['dominant']] for r in rr],
                               s=10 if not good else 18,alpha=.55 if not good else .9,c=c,
                               marker=marker,linewidths=.55,label=f'{label} ({len(rr):,})',rasterized=True)
            ax.set(xlabel='Window amplitude ($10^3$ ADC)',ylabel='Time constant (s)',title=title)
            if tag=='all_windows' and key!='tau_r': ax.set_yscale('log')
            else:
                fmt=ScalarFormatter(useOffset=False); fmt.set_scientific(False); ax.yaxis.set_major_formatter(fmt)
            ax.grid(alpha=.18)
        leg=axs[0].legend(fontsize=9,loc='lower right',framealpha=.95)
        for handle in leg.legend_handles: handle.set_alpha(1)
        ok=sum(r['success'] for r in sel)
        caption='Pileup: largest fitted component per window; decay axes are logarithmic.' if tag=='all_windows' else 'All 5,000 single-pulse windows; linear axes.'
        fig.suptitle(f'CUORE | {tag.replace("_"," ")} | numerical fits: {ok:,}/{len(sel):,}\n{caption}\nExploratory fits: flagged parameters are not validated physical constants.',fontsize=12)
        for ext in ('png','pdf'): fig.savefig(out/f'tau_vs_amplitude_{tag}.{ext}',dpi=200)
        plt.close(fig)
    print('Re-rendered two scatter figures.')
if __name__=='__main__': main()

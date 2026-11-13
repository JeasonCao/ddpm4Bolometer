"""Inspect the non-Gaussian 7.5 mV spectra around the 15--30 keV transition."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.model261108_inference.full_comparison import calibration, digest, json_safe, fit_gaussian
import json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    energies=(15,20,25,30,40,50)
    sources={'old_zeroinit':('7p5mV_zeroinit','#2274c9','Old zero-init'),
             'new_nonoise':('model261108_nonoise/efficiency_7p5mV','#e46c35','New no_noise')}
    rows={}
    arrays={}
    provenance=[]
    for e in sorted(set(energies+(239,511,727,1461,2615))):
        rows[e]={}
        arrays[e]={}
        for method,(folder,color,label) in sources.items():
            path=args.notes_root/f'results/fit_results/{folder}/fit_efficiency_{e}keV.h5'
            with h5py.File(path,'r') as f:
                a=f['amp_denoised'][:].astype(float)
                good=f['success_denoised'][:].astype(bool)&np.isfinite(a)
                arrays[e][method]=a[good]
                rows[e][method]={'gaussian':fit_gaussian(a[good])}
            provenance.append({'path':str(path),'sha256':digest(path)})
    cal={method:calibration(rows,method) for method in sources}
    fig,axes=plt.subplots(4,3,figsize=(15,11.5),layout='constrained',gridspec_kw={'height_ratios':[2.2,1,2.2,1]})
    diagnostics={}
    for index,e in enumerate(energies):
        ax=axes[(index//3)*2,index%3]
        inset=axes[(index//3)*2+1,index%3]
        residuals={m:a*cal[m]['k_keV_per_amplitude']-e for m,a in arrays[e].items()}
        full=np.concatenate(list(residuals.values()))
        fullbins=np.linspace(float(full.min())-.02,float(full.max())+.02,120)
        bins=np.linspace(-1.6,1.6,100)
        diagnostics[e]={}
        for method,(_,color,label) in sources.items():
            x=residuals[method]
            # Normalize by all events, so the main-window truncation does not
            # renormalize away tails. The inset displays the complete range.
            h,edges=np.histogram(x,bins=bins)
            ax.stairs(h/(len(x)*np.diff(edges)),edges,color=color,lw=1.3,label=label)
            inset.hist(x,bins=fullbins,histtype='step',color=color,lw=.8)
            g=rows[e][method]['gaussian']
            fwhm=2.355*cal[method]['k_keV_per_amplitude']*g['sigma']
            diagnostics[e][method]={'fwhm_keV':fwhm,'gaussian_chi2_ndf':g['chi2ndf'],
                                    'outside_display_window_fraction':float((np.abs(x)>1.6).mean()),
                                    'std_keV':float(x.std())}
        inset.set_yscale('log');inset.tick_params(labelsize=8)
        inset.set_ylabel('Counts (log)');inset.set_xlabel('Full residual range (keV)',fontsize=9)
        inset.grid(alpha=.15)
        ax.axvline(0,color='black',ls=':',lw=.7)
        ax.set_xlim(-1.6,1.6)
        old=diagnostics[e]['old_zeroinit'];new=diagnostics[e]['new_nonoise']
        ax.set_title(f'{e} keV: Gaussian FWHM {old["fwhm_keV"]:.3f} → {new["fwhm_keV"]:.3f} keV',fontsize=10)
        ax.text(.98,.95,f'Gaussian χ²/ndf\nold {old["gaussian_chi2_ndf"]:.1f}, new {new["gaussian_chi2_ndf"]:.1f}',
                transform=ax.transAxes,ha='right',va='top',fontsize=8)
        ax.set_xlabel('Reconstructed energy − true energy (keV)')
        ax.set_ylabel('Probability density (1/keV)')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('7.5 mV: low-energy spectral shapes and tails',fontsize=15)
    fig.supxlabel('Successful finite tri-exponential amplitudes; no chi-square gate.\n'
                  'Five-line calibration; peak widths alone do not describe these asymmetric distributions.',fontsize=9)
    args.output.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf'):
        fig.savefig(args.output/f'low_energy_spectra_7p5mV.{extension}',dpi=180)
    plt.close(fig)
    (args.output/'low_energy_spectra_7p5mV.json').write_text(json.dumps(json_safe(
        {'calibrations':cal,'spectra':diagnostics,'inputs':provenance,'code_sha256':digest(__file__)}),indent=2),encoding='utf-8')
    print(json.dumps(json_safe(diagnostics),indent=2))

if __name__=='__main__':
    main()

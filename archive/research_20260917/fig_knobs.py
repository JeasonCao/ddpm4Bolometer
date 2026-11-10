import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['DejaVu Sans']
OUT='/tmp/claude-1000/-home-wsl-0vbb/946b7f43-5893-49fc-8189-f0872f2eda18/scratchpad'
NOISE='#5b7fa6'; FLOOR='#c0562c'; GOOD='#2f6b57'; GREY='#8b9490'

s    = np.array([1.0,0.7,0.5,0.3,0.2,0.1,0.05,0.02,0.0])
std26= np.array([0.4063,0.1167,0.0663,0.0328,0.0245,0.0219,0.0216,0.0213,0.0214])
std15= np.array([1.6479,1.0251,0.9396,0.9313,0.9475,0.9697,0.9827,0.9916,0.9997])
eta  = np.array([1.0,0.7,0.5,0.3,0.1,0.0])
e26  = np.array([0.3181,0.3351,0.3344,0.3541,0.3876,0.4063])
e15  = np.array([1.2323,1.2854,1.2801,1.3533,1.5392,1.6479])

fig,ax=plt.subplots(1,2,figsize=(12.4,4.6)); fig.patch.set_facecolor('white')

a=ax[0]; a.set_facecolor('white')
a.plot(s,std26,'o-',color=FLOOR,lw=2.4,ms=6,label='2615 keV')
a.axhline(0.0394,color=GREY,ls='--',lw=1.4)
a.text(0.62,0.047,'no denoising (2615 keV)',fontsize=8.5,color=GREY)
a.axvspan(-0.02,0.105,color=GOOD,alpha=.10)
a.text(0.045,0.30,'plateau',ha='center',fontsize=10,color=GOOD,fontweight='bold')
a.set_xlabel(r'$x_T$ scale  $s$   ($x_T = s\cdot\mathrm{randn}$,  $\eta=0$)',fontsize=11)
a.set_ylabel('std(a)  (keV)   — the part that sets resolution',fontsize=10.5)
a.set_title('Shrinking the invented field helps monotonically\nand saturates below s $\\approx$ 0.1',fontsize=11.5,fontweight='bold')
a.grid(alpha=.3); a.legend(fontsize=9.5); a.set_xlim(-0.03,1.05); a.set_ylim(0,0.45)
a2=a.twinx(); a2.plot(s,std15,'s--',color=NOISE,lw=1.6,ms=4.5,label='15 keV (right axis)')
a2.set_ylabel('std(a) (keV), 15 keV',fontsize=9.5,color=NOISE); a2.tick_params(axis='y',colors=NOISE)
a2.set_ylim(0.8,1.8); a2.legend(fontsize=8.5,loc='center right')

b=ax[1]; b.set_facecolor('white')
b.plot(eta,e26,'o-',color=FLOOR,lw=2.4,ms=6,label='2615 keV')
b.set_xlabel(r'$\eta$   (per-step injection,  $x_T=\mathrm{randn}$ fixed)',fontsize=11)
b.set_ylabel('std(a)  (keV)',fontsize=10.5)
b.set_title('More per-step injection also helps — it washes $x_T$ out\nbut far less effectively (0.32 vs 0.021)',fontsize=11.5,fontweight='bold')
b.grid(alpha=.3); b.set_ylim(0,0.45)
b.annotate('$\\eta$=0 : DDIM',xy=(0,0.4063),xytext=(0.12,0.43),fontsize=9,
           arrowprops=dict(arrowstyle='-',color=GREY,lw=1))
b.annotate('$\\eta$=1 : DDPM',xy=(1,0.3181),xytext=(0.62,0.26),fontsize=9,
           arrowprops=dict(arrowstyle='-',color=GREY,lw=1))
b.axhline(0.0214,color=GOOD,ls='--',lw=1.8)
b.text(0.30,0.045,'what $x_T\\rightarrow 0$ achieves (0.021)',fontsize=9.5,color=GOOD,fontweight='bold')
b2=b.twinx(); b2.plot(eta,e15,'s--',color=NOISE,lw=1.6,ms=4.5,label='15 keV (right axis)')
b2.set_ylabel('std(a) (keV), 15 keV',fontsize=9.5,color=NOISE); b2.tick_params(axis='y',colors=NOISE)
b2.set_ylim(0.8,1.8); b2.legend(fontsize=8.5,loc='center right')

fig.suptitle('Two ways to reduce the sampler-injected field — and how far each one gets you',fontsize=12.5,y=1.02)
fig.tight_layout()
fig.savefig(OUT+'/fig_knobs.png',dpi=170,facecolor='white',bbox_inches='tight')
print('ok')

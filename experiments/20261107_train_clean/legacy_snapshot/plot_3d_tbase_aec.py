"""3-D scatter of the existing blue-only single-event selection, all points."""
from plot_after_fit import ROOT, DEST, TAUS, read_all, quality, save, atomic_json, now, np, plt
from pathlib import Path
import hashlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def panel(ax,x,y,z,label,title):
    points=ax.scatter(x,y,z,c=z,cmap='viridis',s=1.8,alpha=.36,
                      linewidths=0,depthshade=False,rasterized=True)
    ax.set_xlabel(r'$T_{base}$ (mK)',labelpad=8)
    ax.set_ylabel(r'$a_{ec}$',labelpad=8)
    ax.set_zlabel(label,labelpad=8)
    ax.set_title(title,pad=12)
    ax.view_init(elev=24,azim=-58)
    ax.set_box_aspect((1,1,.85),zoom=.85)
    ax.tick_params(labelsize=8,pad=1)
    return points


def main():
    files,a,fingerprint=read_all(False)
    good,flags,_,_=quality(a)
    mask=(a['n_pulses_fit']==1)&good&~flags
    with np.load(DEST/'blue_only/selected_windows.npz') as previous:
        np.testing.assert_array_equal(previous['source_shard'],a['shard'][mask])
        np.testing.assert_array_equal(previous['window_idx'],a['window_idx'][mask])
    dest=DEST/'blue_only/three_dimensional'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        raise FileExistsError('3-D outputs already complete')
    x=a['params']['T_base'][mask]*1e3
    y=a['params']['a_ec'][mask]
    count=len(x)
    titles=['Rise time','Fast decay time','Slow decay time']
    print(f'Rendering all {count} selected single windows',flush=True)
    fig=plt.figure(figsize=(21,7.3))
    fig.subplots_adjust(left=.01,right=.98,bottom=.08,top=.83,wspace=.08)
    for i,((key,scale,label),title) in enumerate(zip(TAUS,titles)):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        panel(ax,x,y,a[key][mask,0]*scale,label,title)
    fig.suptitle(f'Selected single events: N={count:,} | X: T_base, Y: a_ec, Z: fitted time constant\n'
                 'Same blue-only selection; all windows plotted. Color encodes Z within each panel.',fontsize=14,y=.97)
    save(fig,dest/'Tbase_aec_time_constants_3d')
    for (key,scale,label),title in zip(TAUS,titles):
        fig=plt.figure(figsize=(8,7.4))
        ax=fig.add_subplot(111,projection='3d')
        fig.subplots_adjust(left=.02,right=.86,bottom=.08,top=.86)
        points=panel(ax,x,y,a[key][mask,0]*scale,label,title)
        cax=fig.add_axes([.88,.25,.025,.45])
        colorbar=fig.colorbar(points,cax=cax)
        colorbar.set_label(label)
        fig.suptitle(f'Selected single events | N={count:,} | color = Z',fontsize=13)
        save(fig,dest/('Tbase_aec_'+key+'_3d'))
    # HTML embeds plotly.js, so rotation and hover work without network access.
    interactive=make_subplots(rows=1,cols=3,specs=[[{'type':'scene'}]*3],subplot_titles=titles,
                              horizontal_spacing=.025)
    custom=np.column_stack([a['shard'][mask],a['window_idx'][mask]])
    scenes={}
    for i,(key,scale,label) in enumerate(TAUS):
        z=a[key][mask,0]*scale
        unit='ms' if i==0 else 's'
        interactive.add_trace(go.Scatter3d(x=x,y=y,z=z,mode='markers',customdata=custom,
            marker=dict(size=1.6,opacity=.45,color=z,colorscale='Viridis',showscale=False),
            name=titles[i],showlegend=False,
            hovertemplate='T_base=%{x:.5f} mK<br>a_ec=%{y:.6f}<br>'+key+'=%{z:.6g} '+unit+
                          '<br>source shard=%{customdata[0]}<br>window_idx=%{customdata[1]}<extra></extra>'),row=1,col=i+1)
        scenes['scene' if i==0 else f'scene{i+1}']=dict(
            xaxis_title='T_base (mK)',yaxis_title='a_ec',zaxis_title=f'{key} ({unit})',
            aspectmode='cube',camera=dict(eye=dict(x=1.6,y=-1.8,z=1.0)))
    interactive.update_layout(**scenes,height=760,margin=dict(l=0,r=0,b=0,t=105),
        title=dict(text=f'T_base × a_ec × time constant — {count:,} selected single events<br>'
                         '<sup>All points; color encodes Z per panel. Drag to rotate; scroll to zoom; hover for source row.</sup>',x=.5))
    interactive.write_html(dest/'Tbase_aec_time_constants_3d.html',include_plotlyjs=True,
                           full_html=True,config=dict(displaylogo=False,scrollZoom=True))
    report=dict(created_at=now(),count=count,source_fingerprint=fingerprint,
                x='params/T_base (K converted to mK)',y='params/a_ec (dimensionless)',
                z=['t_rise (ms)','t_fall1 (s)','t_fall2 (s)'],no_downsampling=True,
                selection='Exact blue_only/selected_windows.npz; single-event column 0',
                color='Z value, separately scaled per panel; viridis',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(dest/'COMPLETE.json',report)
    (dest/'README.md').write_text(
        '# T_base、a_ec与三个波形时间常数：三维散点\n\n'
        f'沿用蓝色筛选的{count:,}个单事件，与上次保存的源分片/行号逐项相同，不抽样。'
        'X=T_base（mK），Y=a_ec（无量纲），三个Z分别是上升时间（ms）、快下降时间（s）、慢下降时间（s）。'
        '这里a_ec是ODE热导指数，不是三指数拟合中的混合权重alpha。\n\n'
        '颜色仅表示该面板的Z值，各面板独立色标；所有点都属于原来的蓝色质量筛选集合。'
        '总览及三个单图均有PNG/PDF；HTML包含离线绘图库，可拖动旋转、滚轮缩放、悬停查看源分片/窗口行号。'
        '这是散点关系，不是拟合曲面或因果关系。未修改原图、clean或fit文件。\n',encoding='utf-8')
    print(f'Completed: {dest}',flush=True)


if __name__=='__main__':
    main()

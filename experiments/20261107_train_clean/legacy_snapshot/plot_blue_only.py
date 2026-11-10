"""Render the exact blue subset from the existing full-data scatter figures."""
from plot_after_fit import (
    ROOT, DEST, PARAMS, TAUS, read_all, quality, draw_panel, save,
    atomic_json, now, np, plt,
)
import hashlib
from pathlib import Path


def main():
    files,a,fingerprint=read_all(False)
    good,flags,_,_=quality(a)
    single=(a['n_pulses_fit']==1)&good
    mask=single & ~flags
    # Verify the exact same input row mapping and color selection as the original.
    with np.load(DEST/'plot_index.npz') as old:
        np.testing.assert_array_equal(a['shard'],old['source_shard'])
        np.testing.assert_array_equal(a['window_idx'],old['window_idx'])
        np.testing.assert_array_equal(mask,old['single_plotted'] & ~old['diagnostic_flag'])
    count=int(mask.sum())
    excluded=int(single.sum()-count)
    dest=DEST/'blue_only'
    dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        raise FileExistsError('Completed blue-only figures already exist')
    print(f'Plotting {count} blue windows; excluded {excluded} flagged singles',flush=True)
    fig,axes=plt.subplots(3,8,figsize=(25,10),layout='constrained')
    for col,(key,scale,xlabel) in enumerate(PARAMS):
        for row,(tau,yscale,ylabel) in enumerate(TAUS):
            draw_panel(axes[row,col],a['params'][key][mask]*scale,
                       a[tau][mask,0]*yscale,flags[mask],xlabel,ylabel,True)
    fig.suptitle(f'Single events: blue subset only | N={count:,} | {excluded:,} flagged windows excluded\n'
                 'One point per window per panel; no downsampling',fontsize=14)
    save(fig,dest/'single_blue_overview')
    for key,scale,xlabel in PARAMS:
        fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
        for ax,(tau,yscale,ylabel) in zip(axes,TAUS):
            draw_panel(ax,a['params'][key][mask]*scale,a[tau][mask,0]*yscale,
                       flags[mask],xlabel,ylabel)
        fig.suptitle(f'Single events: blue subset only | N={count:,} | no diagnostic flags',fontsize=12)
        save(fig,dest/('single_blue_vs_'+key))
    np.savez_compressed(dest/'selected_windows.npz',source_shard=a['shard'][mask],
                        window_idx=a['window_idx'][mask])
    report=dict(created_at=now(),source_fingerprint=fingerprint,selected=count,
                excluded_flagged_singles=excluded,original_single_count=int(single.sum()),
                selection='Exact original single_plotted AND NOT diagnostic_flag',
                no_downsampling=True,axes='Autoscaled after filtering',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_files=[str(p) for p in files])
    atomic_json(dest/'selection.json',report)
    (dest/'README.md').write_text(
        '# 仅蓝色单事件散点图\n\n'
        f'保留原图蓝色单事件 {count:,} 窗，排除橙色诊断标记单事件 {excluded:,} 窗；'
        f'原单事件总数 {int(single.sum()):,}。已逐行核对与原图保存的颜色选择一致。\n\n'
        '蓝色条件：拟合成功且参数有限、无tau贴边/近等衰减标记、全窗相对残差RMS≤1%、'
        '起点绝对误差≤50ms。筛选不代表每个拟合参数均已准确辨识。\n\n'
        '8个ODE参数×3个时间常数，每面板每个入选窗口一个点，不抽样。'
        '上升时间用ms、下降时间用s。坐标范围在筛选后重新自动缩放。'
        '总览和8张分参数图均有PNG/PDF；selected_windows.npz保留源分片编号及原窗口行号。'
        '原图、原fit HDF5和clean未修改。\n',encoding='utf-8')
    atomic_json(dest/'COMPLETE.json',report)
    print(f'Completed: {dest}',flush=True)


if __name__=='__main__':
    main()

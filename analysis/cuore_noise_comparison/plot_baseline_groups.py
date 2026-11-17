"""Plot empirical CUORE baseline groups; verify offsets against raw windows."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plots', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    a.plots.mkdir(parents=True, exist_ok=True)
    with a.features.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5000
    assert len({(r['source_file'], r['source_row']) for r in rows}) == 5000
    offsets = np.array([float(r['offset_adc']) for r in rows])
    groups = np.array([r['baseline_group'] for r in rows])
    np.testing.assert_array_equal(groups, np.where(offsets > -70000, 'A', 'B'))
    schema = {}
    for name in sorted({r['source_file'] for r in rows}):
        selected = [i for i, r in enumerate(rows) if r['source_file'] == name]
        selected.sort(key=lambda i: int(rows[i]['source_row']))
        with h5py.File(a.data / name, 'r') as h:
            schema[name] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in h.items() if isinstance(v, h5py.Dataset)}
            assert h['Waveform'].shape[1] == 10000
            for j in range(0, len(selected), 128):
                ii = np.array(selected[j:j+128])
                rr = np.array([int(rows[i]['source_row']) for i in ii])
                raw = h['Waveform'][rr, :2500].astype(float)
                np.testing.assert_allclose(raw.mean(axis=1), offsets[ii], atol=1e-6, rtol=0)
                np.testing.assert_allclose(h['normalizationOffset'][rr].ravel(), offsets[ii], atol=.01, rtol=0)
                np.testing.assert_array_equal(h['eventId'][rr].ravel(), [int(rows[i]['event_id']) for i in ii])
                if 'Labels' in h:
                    assert np.all(h['Labels'][rr] == 0)
    summaries = {}
    for g in ['A', 'B']:
        v = offsets[groups == g]
        qs = np.quantile(v, [.025, .16, .5, .84, .975])
        summaries[g] = dict(n=len(v), mean=float(v.mean()), std=float(v.std(ddof=1)), minimum=float(v.min()), maximum=float(v.max()), **dict(zip(['q025','q16','median','q84','q975'], map(float,qs))))
    plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=.5, wspace=.24)
    ax = fig.add_subplot(gs[0, :])
    colors = {'A': '#277db3', 'B': '#d78437'}
    bins = np.linspace(offsets.min()-100, offsets.max()+100, 95)
    for g in ['A', 'B']:
        ax.hist(offsets[groups == g], bins=bins, color=colors[g], alpha=.85, label=f'基线分组 {g}（n={summaries[g]["n"]}）')
    ax.axvline(-70000, color='#555555', ls='--', lw=1.5, label='划分阈值 −70,000 ADC')
    ax.set(title='全部 5,000 个单事件：前 2.5 秒的基线均值', xlabel='基线均值（ADC）', ylabel='窗口数')
    ax.legend(loc='upper center', fontsize=10)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    for j, g in enumerate(['A','B']):
        ax = fig.add_subplot(gs[1, j])
        v = offsets[groups == g]; s = summaries[g]
        ax.hist(v, bins=45, color=colors[g], alpha=.85)
        ax.axvspan(s['q16'], s['q84'], color=colors[g], alpha=.15, label='中央 68% 区间')
        ax.axvline(s['median'], color='#333333', ls='--', label=f'中位数 {s["median"]:,.1f}')
        ax.set(title=f'{g} 组放大（独立横轴范围）', xlabel='基线均值（ADC）', ylabel='窗口数')
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    fig.suptitle('基线位置的分布：每个窗口贡献一个均值', fontsize=16, y=.98)
    fig.text(.5, .01, '基线均值描述直流偏置；噪声 RMS 描述窗口内部围绕均值的起伏。A/B 为分析分组，非官方类别。', ha='center', fontsize=10)
    fig.subplots_adjust(top=.88, bottom=.10)
    for ext in ['png','pdf']:
        fig.savefig(a.plots / f'05_baseline_offset_groups.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    with (a.output / 'baseline_offset_per_event.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['source_file','source_row','event_id','baseline_group','baseline_mean_adc'])
        writer.writeheader()
        for r, v in zip(rows,offsets):
            writer.writerow(dict(source_file=r['source_file'], source_row=r['source_row'], event_id=r['event_id'], baseline_group=r['baseline_group'], baseline_mean_adc=float(v)))
    result = dict(selection='Existing 5000 single-event identities; test Labels are absent, classification inherited from prior analysis', threshold_adc=-70000, baseline_samples=[0,2500], groups=summaries, hdf5_schema=schema, checks='All 5000 means, stored offsets, event IDs and group rules verified against raw HDF5; available training labels checked', feature_sha256=hashlib.sha256(a.features.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output / 'baseline_offset_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2),encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

"""run_summary.json for the Octopus delivery package: file facts only, no truth (energies, times)."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import generate_of_stream as GEN
from verify_of_stream import read_bin

ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, required=True); a = ap.parse_args()
info = json.load(open(os.path.join(GEN.TRUTH, f'run{a.run}', 'truth', 'run_info.json')))
raw = os.path.join(GEN.STAGE, f'run{a.run}', 'raw')
parts = []
for fn in info['partials']:
    hdr, fs, R, d = read_bin(os.path.join(raw, fn))
    parts.append(dict(file=fn, n_samples=int(len(d)), duration_s=len(d) / fs, size_bytes=os.path.getsize(os.path.join(raw, fn))))
out = dict(run=f'{a.run:06d}', channel='001', filename_prefix=info['prefix'], prefix_timezone='Asia/Shanghai (UTC+8)',
           sampling_rate_hz=1000.0, adc_range_v=10.0, header='uint32 0x00006C20, float32 fs, float32 ADC range (little endian)',
           sample_encoding='uint32 LE, value in the upper 24 bits: V = (s >> 8) / 2^23 * 10 - 10',
           noise_rms_v=info['rms'], noise='stationary (one fixed noise state)' if info['noise'] == 'stationary'
           else 'one noise state per partial (~24 h), 1 h power-preserving crossfade at each partial boundary',
           n_pulses=info['n_events'], pile_up=False, baseline_v=0.0, pulse_polarity='positive',
           suggested_window_s=10.0, suggested_pretrigger_s=1.5,
           total_samples=info['n_samples'], total_duration_h=info['duration_h'], partials=parts)
json.dump(out, open(os.path.join(GEN.STAGE, f'run{a.run}', 'run_summary.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ('run', 'total_duration_h', 'n_pulses')}), [p['size_bytes'] // 2**20 for p in parts], 'MiB')

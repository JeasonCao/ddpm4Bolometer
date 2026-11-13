"""Complete Model261108 fits using the canonical fitter and cached raw fits.

Inputs are read-only. Paths are CLI arguments; fitting thresholds are unchanged.
Historical raw fits are copied with provenance, never silently recomputed.
"""
import os
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from scripts.run_fitting import _fit_one
from scripts.run_gauss_fit import process_file
from src.basics.amplitude import max_baseline_amplitude


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def initialize(template):
    global CLEAN
    CLEAN = template


def fit_row(item):
    index, waveform = item
    return index, _fit_one((waveform, 'denoised', 10., 1e-5, 9.9, CLEAN))


def fit_file(record, args):
    relative = Path(record['path'])
    group = relative.parts[0]
    level = group.removeprefix('efficiency_')
    energy = int(record['energy_kev'])
    raw_path = args.data_root / 'eval_data_holdout' / relative
    denoised_path = args.data_root / 'model261108_inference' / relative
    cached_path = args.notes_root / f'results/fit_results/{level}_zeroinit/fit_efficiency_{energy}keV.h5'
    target = args.notes_root / f'results/fit_results/model261108_nonoise/{group}/fit_efficiency_{energy}keV.h5'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with h5py.File(target, 'r') as existing:
            assert existing.attrs['denoised_sha256'] == record['sha256']
            assert existing.attrs['source_sha256'] == record['source_sha256']
            assert len(existing['amp_denoised']) == 5000
        return {'path': str(target), 'reused': True, 'sha256': digest(target)}
    assert digest(denoised_path) == record['sha256'], str(denoised_path)
    with h5py.File(raw_path, 'r') as raw, h5py.File(denoised_path, 'r') as fresh, h5py.File(cached_path, 'r') as cached:
        assert fresh.attrs['source_sha256'] == record['source_sha256']
        assert fresh.attrs['model_sha256'] == record['model_sha256']
        assert fresh['waveforms_denoised'].shape == raw['waveforms_noisy'].shape == (5000, 10000)
        assert np.array_equal(fresh['source_row'][:], np.arange(5000))
        assert int(cached.attrs['energy_kev']) == energy
        assert len(cached['amp_noisy']) == 5000
        for key in ('snr_db', 'noise_rms'):
            assert np.allclose(cached[key][:], raw[key][:], rtol=1e-6, atol=1e-8)
        for key in raw['noise_params']:
            assert np.array_equal(cached['noise_params'][key][:], raw['noise_params'][key][:])
        template = raw['clean_template'][:]
        true_amp = max_baseline_amplitude(template)
        assert np.isclose(true_amp, cached.attrs['true_amp_V'], rtol=1e-8)
        waveforms = fresh['waveforms_denoised'][:]
    started = time.monotonic()
    results = [None] * len(waveforms)
    with mp.Pool(args.workers, initializer=initialize, initargs=(template,)) as pool:
        for count, (index, result) in enumerate(pool.imap_unordered(fit_row, enumerate(waveforms), chunksize=16), 1):
            results[index] = result
            if count % 1000 == 0:
                print(f'{group} {energy} keV: {count}/5000', flush=True)
                save_json(args.work_dir / 'fit_status.json', {'stage': 'fitting', 'group': group,
                          'energy_keV': energy, 'events': count, 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
    temporary = target.with_suffix('.h5.partial')
    with h5py.File(cached_path, 'r') as cached, h5py.File(temporary, 'w') as output:
        for key, value in cached.attrs.items():
            if not key.startswith('gauss_'):
                output.attrs[key] = value
        for key in cached:
            if not key.endswith('_denoised'):
                cached.copy(key, output)
        for key in results[0]:
            dtype = bool if key.startswith(('success_', 'converged_')) else np.float32
            output.create_dataset(key, data=np.array([row[key] for row in results], dtype=dtype))
        deviation = np.abs(output['amp_denoised'][:] - true_amp) / true_amp
        output.create_dataset('amp_dev_denoised', data=deviation.astype(np.float32))
        output.create_dataset('efficient_denoised', data=output['converged_denoised'][:] & (deviation <= .20))
        output.create_dataset('source_row', data=np.arange(5000, dtype=np.int64))
        output.attrs.update(source_file=str(raw_path), source_sha256=record['source_sha256'],
                            denoised_file=str(denoised_path), denoised_sha256=record['sha256'],
                            model_sha256=record['model_sha256'], noisy_fit_source=str(cached_path),
                            noisy_fit_source_sha256=digest(cached_path), fitting_elapsed_seconds=time.monotonic()-started,
                            fit_function='scripts.run_fitting._fit_one', source_row_convention='zero-based',
                            full_fit_driver_sha256=digest(__file__))
    # Retain conventional gated Gaussian attributes. Ungated plotting statistics
    # are saved separately, never written over these conventional attributes.
    process_file(str(temporary))
    with h5py.File(temporary, 'r') as output:
        assert not any(key.startswith('waveform') for key in output)
        assert np.all(np.isfinite(output['amp_denoised'][:][output['success_denoised'][:]]))
        success = int(output['success_denoised'][:].sum())
        gated = int(output['converged_denoised'][:].sum())
    temporary.replace(target)
    return {'path': str(target), 'reused': False, 'sha256': digest(target),
            'fit_success': success, 'quality_gate_pass': gated, 'seconds': time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('data-root', 'notes-root', 'work-dir'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    manifest = json.loads((args.data_root / 'model261108_inference/inference_download_manifest.json').read_text())
    assert manifest['complete'] and len(manifest['files']) == 34
    records = sorted(manifest['files'], key=lambda r: (-r['energy_kev'], r['path']))
    completed = []
    try:
        for record in records:
            print(f'Start {len(completed)+1}/34: {record["path"]}', flush=True)
            completed.append(fit_file(record, args))
            save_json(args.work_dir / 'fit_manifest.json', {'complete': len(completed)==34, 'files': completed})
        save_json(args.work_dir / 'fit_status.json', {'stage': 'complete', 'files': 34,
                  'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
        print('ALL_34_FITS_COMPLETE', flush=True)
    except Exception:
        import traceback
        save_json(args.work_dir / 'fit_status.json', {'stage': 'failed', 'error': traceback.format_exc()})
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    mp.freeze_support()
    main()

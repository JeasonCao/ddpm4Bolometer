"""Exercise file integrity and the linked-HDF5 fitting entry point on two old events."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'code'), str(HERE / 'plot_deps'),
               r'D:\storage\research\DDPM4bolometer\_development\train_clean_triexp\deps']
import h5py
import numpy as np
from common import record_file, verify_record, save_json
from scripts.run_fitting import fit_dataset


def main():
    validation = HERE / 'validation'
    validation.mkdir(exist_ok=True)
    data = Path(r'D:\storage\research\DDPM4bolometer')
    name = 'efficiency_1p382mV/eval_efficiency_2615keV.h5'
    with h5py.File(data / 'eval_data_holdout' / name) as src, h5py.File(validation / 'source.h5', 'w') as out:
        out.attrs.update(dict(src.attrs))
        for key in ('waveforms_noisy', 'snr_db', 'noise_rms'):
            out[key] = src[key][:2]
        out['clean_template'] = src['clean_template'][:]
        group = out.create_group('noise_params')
        for key in src['noise_params']:
            group[key] = src['noise_params'][key][:2]
    with h5py.File(data / 'nonoise_out' / name) as old, h5py.File(validation / 'denoised.h5', 'w') as out:
        out['waveforms_denoised'] = old['waveforms_denoised_nonoise'][:2]
    with h5py.File(validation / 'source.h5') as src, h5py.File(validation / 'linked.h5', 'w') as out:
        out.attrs.update(dict(src.attrs))
        for key in src:
            out[key] = h5py.ExternalLink(str(validation / 'source.h5'), '/' + key)
        out['waveforms_denoised'] = h5py.ExternalLink(str(validation / 'denoised.h5'), '/waveforms_denoised')
    fit_dataset(str(validation / 'linked.h5'), str(validation / 'fit_smoke.h5'), 1, 10., 1e-5, 9.9, .2)
    with h5py.File(validation / 'fit_smoke.h5') as fit:
        assert len(fit['amp_denoised']) == 2
        assert np.isfinite(fit['amp_denoised'][:]).all()
    path = validation / 'integrity_probe.txt'
    path.write_text('original')
    record = record_file(path, validation)
    assert verify_record(validation, record)
    path.write_text('corrupt!')
    assert not verify_record(validation, record)
    save_json(validation / 'checks.json', {'linked_input_fit_passed': True, 'corruption_detected': True})


if __name__ == '__main__':
    main()

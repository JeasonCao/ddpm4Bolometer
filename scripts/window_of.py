"""Independent, batched HDF5 window optimum filtering. NumPy + h5py only.

See window_of_README.md for conventions. Does not run a denoising model.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import h5py


def subtract_baseline(x, samples):
    x = np.asarray(x, dtype=np.float64)
    baseline = x[..., :samples].mean(axis=-1)
    return x - baseline[..., None], baseline


def estimate(X, template, spectrum, shifts, mode='standard', dc='exclude', floor=1e-12):
    """FFT events -> signed template coefficient, delay, boundary and validity.

    Positive delay means an event later than the template. DFT shifts are cyclic.
    Standard uses real-signal one-sided multiplicities; legacy matches the old
    Octopus Python formula (equal rFFT-bin weights). Search minimizes residual
    quadratic form with an unconstrained signed amplitude, not positive-only.
    """
    n = len(template)
    S = np.fft.rfft(template)
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.shape != S.shape or not np.isfinite(spectrum).all() or np.any(spectrum < 0):
        raise ValueError('Noise spectrum must be finite, nonnegative and match the window.')
    active = np.ones(len(S), dtype=bool)
    if dc == 'exclude':
        active[0] = False
    positive = spectrum[active & (spectrum > 0)]
    if not len(positive):
        raise ValueError('Noise spectrum has no positive active bins.')
    regularized = np.maximum(spectrum, np.median(positive) * floor)
    if np.any(regularized[active] <= 0):
        raise ValueError('Zero spectrum requires a positive spectral floor.')
    weights = np.ones(len(S))
    if mode == 'standard':
        weights[1:] = 2
        if n % 2 == 0:
            weights[-1] = 1
    weights[~active] = 0
    inv = np.divide(weights, regularized, out=np.zeros_like(weights), where=active)
    phase = np.exp(-2j * np.pi * np.outer(shifts, np.arange(len(S))) / n)
    shifted = S[None, :] * phase
    if mode == 'standard' and n % 2 == 0:
        # A real fractional cyclic shift has a real Nyquist endpoint.
        shifted[:, -1] = shifted[:, -1].real
    norms = (np.abs(shifted) ** 2 * inv).sum(axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Template has zero/invalid norm in the selected frequency band.')
    numerators = np.real(X @ (np.conj(shifted) * inv).T)
    scores = numerators ** 2 / norms
    finite = np.isfinite(scores).all(axis=1)
    best = np.argmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    amp = numerators[np.arange(len(X)), best] / norms[best]
    delay = np.asarray(shifts)[best].copy()
    amp[~finite] = np.nan
    delay[~finite] = np.nan
    boundary = finite & (len(shifts) > 1) & ((best == 0) | (best == len(shifts)-1))
    return amp, delay, boundary, finite


def shape_check(ds, n=None, vector=False):
    if ds.ndim != (1 if vector else 2) or not np.issubdtype(ds.dtype, np.number) or np.issubdtype(ds.dtype, np.complexfloating):
        raise ValueError(f'{ds.name}: expected a real numeric {1 if vector else 2}D array.')
    if ds.shape[-1] < 3 or (n is not None and ds.shape[-1] != n):
        raise ValueError(f'{ds.name}: incompatible window length {ds.shape[-1]}.')
    if not vector and ds.shape[0] == 0:
        raise ValueError(f'{ds.name}: empty window collection.')


def rate_check(f, ds, rate):
    # Explicit --sample-rate is required; recognized metadata must agree with it.
    for attrs in (f.attrs, ds.attrs):
        for key in ('f_sample', 'sample_rate', 'sampling_rate', 'sample_rate_hz', 'sampling_rate_hz', 'fs', 'fs_Hz'):
            if key in attrs and not np.isclose(float(attrs[key]), rate, rtol=1e-9):
                raise ValueError(f'{f.filename}: {key} disagrees with --sample-rate.')


def batches(ds, size, limit=None):
    stop = len(ds) if limit is None else min(limit, len(ds))
    for start in range(0, stop, size):
        yield start, np.asarray(ds[start:min(start+size, stop)], dtype=np.float64)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Event HDF5 file (read only).')
    p.add_argument('--input-key', required=True)
    ap = p.add_mutually_exclusive_group(required=True)
    ap.add_argument('--template-file', help='HDF5 file containing a 1D template.')
    ap.add_argument('--ap-pulses-file', help='HDF5 file containing aligned pulses to average.')
    p.add_argument('--ap-key', required=True)
    p.add_argument('--noise-files', nargs='+', required=True)
    p.add_argument('--noise-key', required=True)
    p.add_argument('--sample-rate', type=float, required=True, help='Hz; assertion for metadata-free inputs.')
    p.add_argument('--output', required=True, help='New HDF5 file; existing paths are never overwritten.')
    p.add_argument('--weighting', choices=['standard', 'legacy', 'both'], default='both')
    p.add_argument('--baseline-samples', type=int, default=1400)
    p.add_argument('--delay-range', type=float, default=0, help='Symmetric half-range in samples; 0 = fixed.')
    p.add_argument('--delay-step', type=float, default=0.5)
    p.add_argument('--dc', choices=['exclude', 'include'], default='exclude')
    p.add_argument('--spectral-floor', type=float, default=1e-12, help='Fraction of median positive spectrum.')
    p.add_argument('--ap-average', choices=['raw', 'peak-normalized'], default='raw', help='For aligned pulse collections only.')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-events', type=int, help='Optional bounded evaluation; default all events.')
    args = p.parse_args(argv)
    if not np.isfinite(args.sample_rate) or args.sample_rate <= 0:
        p.error('--sample-rate must be positive and finite.')
    if args.batch_size < 1 or args.baseline_samples < 1 or (args.max_events is not None and args.max_events < 1):
        p.error('Batch size, baseline length and max-events must be positive.')
    if not np.isfinite([args.delay_range, args.delay_step, args.spectral_floor]).all() or args.delay_range < 0 or args.delay_step <= 0 or args.spectral_floor < 0:
        p.error('Invalid delay range/step or spectral floor.')
    if not np.isclose(args.delay_range / args.delay_step, round(args.delay_range / args.delay_step)):
        p.error('--delay-range must be an integer multiple of --delay-step.')
    return args


def run(args):
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f'Output already exists: {target}')
    with h5py.File(args.input, 'r') as f:
        ds = f[args.input_key]
        shape_check(ds)
        n = ds.shape[1]
        rate_check(f, ds, args.sample_rate)
        total = len(ds) if args.max_events is None else min(len(ds), args.max_events)
    if args.baseline_samples >= n or args.delay_range >= n/2:
        raise ValueError('Baseline must be shorter than a window; delay range must be < N/2.')
    with h5py.File(args.template_file or args.ap_pulses_file, 'r') as f:
        ds = f[args.ap_key]
        shape_check(ds, n, vector=bool(args.template_file))
        rate_check(f, ds, args.sample_rate)
        if args.template_file:
            ap, _ = subtract_baseline(ds[:], args.baseline_samples)
            ap_count = 1
        else:
            ap = np.zeros(n)
            ap_count = 0
            for _, x in batches(ds, args.batch_size):
                x, _ = subtract_baseline(x, args.baseline_samples)
                if not np.isfinite(x).all():
                    raise ValueError('AP pulses contain nonfinite values.')
                if args.ap_average == 'peak-normalized':
                    peaks = x.max(axis=1)
                    if np.any(peaks <= 0):
                        raise ValueError('AP pulse peak must be positive.')
                    x = x / peaks[:, None]
                ap += x.sum(axis=0)
                ap_count += len(x)
            ap /= ap_count
    peak = ap.max()
    if not np.isfinite(ap).all() or peak <= 0:
        raise ValueError('AP must be finite with a positive peak after baseline subtraction.')
    ap /= peak
    magnitude = np.zeros(n//2+1)
    power = np.zeros_like(magnitude)
    noise_count = 0
    for path in args.noise_files:
        with h5py.File(path, 'r') as f:
            ds = f[args.noise_key]
            shape_check(ds, n)
            rate_check(f, ds, args.sample_rate)
            for _, x in batches(ds, args.batch_size):
                x, _ = subtract_baseline(x, args.baseline_samples)
                if not np.isfinite(x).all():
                    raise ValueError(f'Noise contains nonfinite values: {path}')
                absolute = np.abs(np.fft.rfft(x))
                magnitude += absolute.sum(axis=0)
                power += (absolute**2).sum(axis=0)
                noise_count += len(x)
    spectra = {'standard': power/noise_count, 'legacy': magnitude/noise_count}
    modes = ['standard', 'legacy'] if args.weighting == 'both' else [args.weighting]
    steps = int(round(args.delay_range/args.delay_step))
    shifts = np.arange(-steps, steps+1, dtype=float)*args.delay_step
    # Validate all filters before opening output.
    for mode in modes:
        estimate(np.fft.rfft(ap[None]), ap, spectra[mode], shifts, mode, args.dc, args.spectral_floor)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, 'x') as out, h5py.File(args.input, 'r') as source:
        out.attrs['complete'] = False
        out.attrs['config_json'] = json.dumps(vars(args), ensure_ascii=False, sort_keys=True)
        out.attrs['sample_rate_hz'] = args.sample_rate
        out.attrs['window_samples'] = n
        out.attrs['ap_count'] = ap_count
        out.attrs['noise_count'] = noise_count
        out.attrs['ap_original_peak'] = peak
        out.attrs['amplitude_units'] = 'same as input waveform; template normalized to positive peak 1'
        out.attrs['delay_convention'] = 'positive means event later than AP; cyclic DFT shift'
        out.attrs['spectrum_convention'] = 'raw rFFT magnitude/power means, not V^2/Hz'
        out.create_dataset('template', data=ap)
        out.create_dataset('frequency_hz', data=np.fft.rfftfreq(n, 1/args.sample_rate))
        out.create_dataset('noise_mean_magnitude', data=spectra['legacy'])
        out.create_dataset('noise_mean_power', data=spectra['standard'])
        out.create_dataset('event_index', data=np.arange(total, dtype=np.int64))
        out.create_dataset('baseline', shape=(total,), dtype='f8')
        for mode in modes:
            group = out.create_group(mode)
            for key in ('amplitude', 'delay_samples', 'delay_seconds'):
                group.create_dataset(key, shape=(total,), dtype='f8', fillvalue=np.nan)
            for key in ('valid', 'at_delay_boundary'):
                group.create_dataset(key, shape=(total,), dtype='bool')
        for start, x in batches(source[args.input_key], args.batch_size, total):
            x, baseline = subtract_baseline(x, args.baseline_samples)
            sl = slice(start, start+len(x))
            out['baseline'][sl] = baseline
            finite = np.isfinite(x).all(axis=1)
            X = np.fft.rfft(np.where(finite[:, None], x, 0))
            for mode in modes:
                amp, delay, edge, valid = estimate(X, ap, spectra[mode], shifts, mode, args.dc, args.spectral_floor)
                valid &= finite
                amp[~valid] = np.nan
                delay[~valid] = np.nan
                group = out[mode]
                group['amplitude'][sl] = amp
                group['delay_samples'][sl] = delay
                group['delay_seconds'][sl] = delay/args.sample_rate
                group['at_delay_boundary'][sl] = edge & valid
                group['valid'][sl] = valid
        out.attrs['complete'] = True
    print(f'Completed {total} events, {noise_count} noise windows, modes={modes}: {target}')


if __name__ == '__main__':
    run(parse_args())

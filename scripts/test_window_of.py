"""Short CPU verification; temporary synthetic files only."""
import tempfile
from pathlib import Path
import unittest
import h5py
import numpy as np
from window_of import estimate, parse_args, run


class WindowOFTests(unittest.TestCase):
    def test_standard_against_time_domain_least_squares(self):
        rng = np.random.default_rng(14)
        for n in (63, 64):
            t = rng.normal(size=n)
            x = rng.normal(size=(9, n)) + 2*t
            a, delay, _, valid = estimate(np.fft.rfft(x), t, np.ones(n//2+1), [0.], dc='include')
            np.testing.assert_allclose(a, x@t/(t@t), atol=1e-12)
            self.assertTrue(valid.all())

    def test_integer_delays_and_signed_amplitude(self):
        n = 128
        t = np.exp(-((np.arange(n)-50)/6)**2)
        x = np.stack([3*np.roll(t, 2), -2*np.roll(t, -3)])
        for mode in ('standard', 'legacy'):
            a, d, edge, valid = estimate(np.fft.rfft(x), t, np.ones(65), np.arange(-4., 5.), mode)
            np.testing.assert_allclose(a, [3, -2], atol=1e-12)
            np.testing.assert_equal(d, [2, -3])
            self.assertTrue(valid.all())
            self.assertFalse(edge.any())

    def test_legacy_formula_regression(self):
        rng = np.random.default_rng(19)
        n = 100
        t = rng.normal(size=n)
        X = np.fft.rfft(rng.normal(size=(20, n)))
        S = np.fft.rfft(t)
        D = rng.uniform(0.5, 2, len(S))
        shifts = np.arange(-3, 3.01, .5)
        H = S.conj()/D
        old = (X@(H[None]*np.exp(-2j*np.pi*np.outer(shifts, np.arange(len(S)))/n)).T).real/(H*S).sum().real
        best = np.argmax(old**2, axis=1)
        a, d, _, _ = estimate(X, t, D, shifts, 'legacy', 'include', 0)
        np.testing.assert_allclose(a, old[np.arange(len(X)), best], atol=1e-12)
        np.testing.assert_allclose(d, -shifts[best], atol=1e-12)

    def test_zero_noise_rejected(self):
        with self.assertRaises(ValueError):
            estimate(np.ones((1, 6)), np.arange(10.), np.zeros(6), [0.])

    def test_fractional_delay_with_nyquist_component(self):
        n = 64
        t = np.random.default_rng(33).normal(size=n) + (-1.)**np.arange(n)
        S = np.fft.rfft(t)
        shifted = S*np.exp(-2j*np.pi*np.arange(len(S))*0.5/n)
        shifted[-1] = shifted[-1].real
        x = 2.5*np.fft.irfft(shifted, n=n)
        a, d, _, _ = estimate(np.fft.rfft(x[None]), t, np.ones(len(S)), np.arange(-1, 1.01, .5), dc='include')
        np.testing.assert_allclose(a, [2.5], atol=1e-12)
        np.testing.assert_equal(d, [.5])

    def test_cli_io_template_mean_ap_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'input.h5'
            t = np.zeros(128)
            t[30:80] = np.sin(np.linspace(0, np.pi, 50))
            t /= t.max()
            x = np.stack([2*t+7, 3*t-2, t])
            x[-1, 80] = np.nan
            with h5py.File(source, 'w') as f:
                f.attrs['sample_rate_hz'] = 1000.
                f['template'] = t
                f['pulses'] = np.stack([t, 2*t])
                f['noise'] = np.random.default_rng(8).normal(size=(24, 128))
                f['events'] = x
            for use_mean in (False, True):
                output = root/f'result_{use_mean}.h5'
                args = parse_args(['--input', str(source), '--input-key', 'events',
                    '--ap-pulses-file' if use_mean else '--template-file', str(source),
                    '--ap-key', 'pulses' if use_mean else 'template',
                    '--noise-files', str(source), '--noise-key', 'noise',
                    '--sample-rate', '1000', '--baseline-samples', '20',
                    '--batch-size', '2', '--output', str(output)])
                run(args)
                with h5py.File(output, 'r') as f:
                    self.assertTrue(f.attrs['complete'])
                    for mode in ('standard', 'legacy'):
                        np.testing.assert_allclose(f[mode+'/amplitude'][:2], [2, 3], atol=1e-12)
                        self.assertFalse(f[mode+'/valid'][2])
                        self.assertTrue(np.isnan(f[mode+'/amplitude'][2]))
                with self.assertRaises(FileExistsError):
                    run(args)
            args.sample_rate = 500
            args.output = str(root/'mismatch.h5')
            with self.assertRaises(ValueError):
                run(args)
            self.assertFalse(Path(args.output).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""
data.py — mirrors parseIetiData.jl

Reads the raw UInt32 ADC samples after the header.
Uses numpy.fromfile for efficient bulk reading instead of sample-by-sample.
"""

import numpy as np


def read_data(f, endianness: str, nbits: int, max_samples: int = None) -> np.ndarray:
    """Read ADC samples from an open binary file (positioned after the header).

    Parameters
    ----------
    f           : open binary file object
    endianness  : 'little' or 'big'
    nbits       : ADC bit depth — only 32 is supported (mirrors Julia code)
    max_samples : if set, read at most this many samples

    Returns
    -------
    adc_values : np.ndarray of uint32
    """
    if nbits != 32:
        raise ValueError(f"Unsupported nbits value: {nbits}")

    dtype = np.dtype("uint32").newbyteorder("<" if endianness == "little" else ">")
    adc_values = np.fromfile(f, dtype=dtype)

    # Ensure native byte order for downstream arithmetic
    adc_values = adc_values.astype(np.uint32)

    if max_samples is not None:
        adc_values = adc_values[:max_samples]

    return adc_values

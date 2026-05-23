"""
convert.py — mirrors convertIetiData.jl

Binary ↔ float codec for IETI .bin files.
"""

import struct
import numpy as np
from .header import read_header
from .data import read_data


# ── DECODER: binary → float ──────────────────────────────────────────────────

def adc_to_voltage(adc_values: np.ndarray, full_range: float, nbits: int) -> np.ndarray:
    """Convert raw ADC counts to voltage.

    Voltage = value / (2^nbits - 1) * full_range
    """
    max_code = 2**nbits - 1
    # full_range from header is in mV; convert to V
    return adc_values.astype(np.float64) * (full_range / max_code / 1000.0)


def process_adc_file(
    filename: str, max_samples: int = None,
    start_sec: float = 0.0, duration_sec: float = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read an IETI .bin file and return time axis, voltages, and header info.

    Parameters
    ----------
    filename    : path to .bin file
    max_samples : if set, read at most this many samples (applied before slicing)
    start_sec   : start time in seconds (default 0)
    duration_sec: duration in seconds (default None = read to end)

    Returns
    -------
    times    : np.ndarray, time in seconds
    voltages : np.ndarray, voltage in volts
    header   : dict with keys endianness, nbits, fs, full_range
    """
    with open(filename, "rb") as f:
        endianness, nbits, fs, full_range = read_header(f)
        adc_values = read_data(f, endianness, nbits, max_samples=max_samples)

    voltages = adc_to_voltage(adc_values, full_range, nbits)

    # Slice if start/duration requested
    if start_sec > 0 or duration_sec is not None:
        start_sample = int(start_sec * fs)
        if duration_sec is not None:
            end_sample = start_sample + int(duration_sec * fs)
            voltages = voltages[start_sample:end_sample]
        else:
            voltages = voltages[start_sample:]

    times = start_sec + np.arange(len(voltages)) / fs

    header = {"endianness": endianness, "nbits": nbits, "fs": fs, "full_range": full_range}
    return times, voltages, header


# ── ENCODER: float → binary ──────────────────────────────────────────────────

def voltage_to_adc(voltages: np.ndarray, full_range: float, nbits: int = 32) -> np.ndarray:
    """Convert voltages back to raw ADC counts (inverse of adc_to_voltage).

    Uses rounding before cast to avoid systematic truncation bias.
    Clips to valid range [0, 2^nbits - 1].
    """
    max_code = 2**nbits - 1
    raw = np.round(voltages * (max_code / full_range))
    return np.clip(raw, 0, max_code).astype(np.uint32)


def write_header(f, endianness: str, nbits: int, fs: float, full_range: float) -> None:
    """Write the 12-byte IETI header to an open binary file.

    Word 1 is always written little-endian (matches how read_header reads it).
    Words 2-3 follow the file's endianness.
    """
    # Word 1: always little-endian
    config = (ord("l") << 8) | nbits
    f.write(struct.pack("<I", config))

    # Words 2-3: sampling frequency and ADC full range as float32
    fmt = "<f" if endianness == "little" else ">f"
    f.write(struct.pack(fmt, fs))
    f.write(struct.pack(fmt, full_range))


def write_bin(
    filename: str,
    voltages: np.ndarray,
    endianness: str,
    nbits: int,
    fs: float,
    full_range: float,
) -> None:
    """Write a complete IETI .bin file from a voltage array.

    Mirror of process_adc_file — accepts the same header dict values.
    """
    adc_values = voltage_to_adc(voltages, full_range, nbits)
    dtype = np.dtype("uint32").newbyteorder("<" if endianness == "little" else ">")

    with open(filename, "wb") as f:
        write_header(f, endianness, nbits, fs, full_range)
        adc_values.astype(dtype).tofile(f)

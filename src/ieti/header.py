"""
header.py — mirrors parseIetiHeader.jl

Reads the 12-byte IETI file header:
  Word 1 (4 bytes): (endianness_char << 8) + nbits
  Word 2 (4 bytes): sampling frequency as Float32
  Word 3 (4 bytes): ADC full range as Float32
"""

import struct
from typing import Literal

Endianness = Literal["little", "big"]


def read_header(f) -> tuple[Endianness, int, float, float]:
    """Read the 12-byte header from an open binary file.

    Returns
    -------
    endianness : 'little' or 'big'
    nbits      : ADC bit depth (typically 32)
    fs         : sampling frequency in Hz
    full_range : ADC full range in volts
    """
    # Word 1: always read as little-endian (the endianness field itself is not byte-swapped)
    config = struct.unpack("<I", f.read(4))[0]
    endianness_char = chr((config >> 8) & 0xFF)
    nbits = config & 0xFF

    endianness: Endianness = "little" if endianness_char == "l" else "big"
    fmt = "<f" if endianness == "little" else ">f"

    # Word 2: sampling frequency
    fs = struct.unpack(fmt, f.read(4))[0]

    # Word 3: ADC full range
    full_range = struct.unpack(fmt, f.read(4))[0]

    return endianness, nbits, fs, full_range

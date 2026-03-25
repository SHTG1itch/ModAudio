"""
Cinema EQ - reproduces the SMPTE/ISO X-curve used in professional movie theaters
plus sub-bass and presence enhancements for the theater "feel."

X-Curve summary (ISO 2969):
  * Flat from ~2 Hz to 2 kHz
  * High-frequency shelf starting at 2 kHz, rolling off toward Nyquist
  * (Optional) gentle low-frequency roll-off below 63 Hz for large rooms -
    we skip this since home listening environments don't have the same modal
    buildup as a 500-seat cinema.

Additional theater character enhancements (not part of the X-curve spec):
  * Sub-bass shelf    (+4-5 dB below 30-80 Hz) -> visceral LFE impact
  * Presence peak     (+1-2 dB at ~3.5 kHz)    -> dialog clarity / intelligibility
"""

from __future__ import annotations
import numpy as np
from .filters import FilterChain, make_highshelf, make_lowshelf, make_peaking


class CinemaEqualizer:
    """
    Three-stage EQ chain:
      1. Sub-bass extension  (low shelf)
      2. Presence boost      (peaking band)
      3. X-curve HF rolloff  (high shelf)
    """

    def __init__(self, fs: int = 48000, num_channels: int = 2, preset: dict | None = None):
        if preset is None:
            from config import THEATER_PRESET
            preset = THEATER_PRESET

        self._chain = FilterChain([
            # (1) Sub-bass extension - adds cinema LFE thump
            make_lowshelf(
                fc=preset["sub_bass_hz"],
                gain_db=preset["sub_bass_db"],
                q=0.70,
                fs=fs,
                ch=num_channels,
            ),
            # (2) Bass boost (80 Hz region) - body and warmth
            make_lowshelf(
                fc=preset["bass_boost_hz"],
                gain_db=preset["bass_boost_db"],
                q=0.60,
                fs=fs,
                ch=num_channels,
            ),
            # (3) Presence - dialog articulation
            make_peaking(
                fc=preset["presence_hz"],
                gain_db=preset["presence_db"],
                q=1.2,
                fs=fs,
                ch=num_channels,
            ),
            # (4) X-curve HF rolloff starting at 2 kHz
            make_highshelf(
                fc=preset["xcurve_hz"],
                gain_db=preset["xcurve_db"],
                q=0.55,
                fs=fs,
                ch=num_channels,
            ),
        ])

    def process(self, block: np.ndarray) -> np.ndarray:
        """Apply all EQ stages to block (N, C) -> (N, C)."""
        return self._chain.process(block)

    def reset(self):
        self._chain.reset()

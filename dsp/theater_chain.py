"""
TheaterChain - master DSP pipeline.

Signal flow:

  Stereo in (float32, (N, 2))
      |
      v
  +- CinemaEqualizer --------------------------------------------------+
  |  X-curve + sub-bass boost + presence                               |
  +--------------------------------------------------------------------+
      |
      v
  +- VirtualSurroundProcessor -----------------------------------------+
  |  Band-split -> 5-channel upmix -> HRTF binaural render             |
  |  (Sub-bass omnidirectional, mid surround, HF widened)              |
  +--------------------------------------------------------------------+
      |
      v
  +- TheaterReverb ----------------------------------------------------+
  |  Early reflections + FDN reverb tail (RT60 ~ 1.2 s)               |
  +--------------------------------------------------------------------+
      |
      v
  +- Output limiter ---------------------------------------------------+
  |  Peak limiter + master gain trim                                   |
  +--------------------------------------------------------------------+
      |
      v
  Stereo out (float32, (N, 2))
"""

from __future__ import annotations
import numpy as np

from .equalizer  import CinemaEqualizer
from .spatializer import VirtualSurroundProcessor
from .reverb      import TheaterReverb


class _PeakLimiter:
    """
    Simple look-ahead peak limiter using a one-pole envelope follower.
    Transparent on transients, prevents digital clipping.
    """

    def __init__(self, threshold: float = 0.93, release_ms: float = 80.0,
                 fs: int = 48000):
        self._threshold = threshold
        self._release   = np.exp(-1.0 / (release_ms * 1e-3 * fs))
        self._env       = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (N, C) float32 -> (N, C) float32"""
        x64 = x.astype(np.float64)
        peak = np.abs(x64).max(axis=1)   # per-sample peak across channels
        out  = np.empty_like(x64)
        thr  = self._threshold
        rel  = self._release
        env  = self._env

        for i in range(len(peak)):
            # Attack: instant
            env = max(peak[i], env * rel)
            gain = thr / max(env, thr)
            out[i] = x64[i] * gain

        self._env = env
        return out.astype(x.dtype)

    def reset(self):
        self._env = 0.0


class TheaterChain:
    """
    Thread-safe, block-based theater audio processor.

    Usage
    -----
    chain = TheaterChain(fs=48000)
    output_block = chain.process(input_block)   # input_block shape: (N, 2)
    """

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        if preset is None:
            from config import THEATER_PRESET
            preset = THEATER_PRESET

        self._fs     = fs
        self._preset = preset

        # Output master gain (linear)
        self._master_gain = 10 ** (preset["output_gain_db"] / 20.0)

        # DSP stages
        self._eq        = CinemaEqualizer(fs=fs, num_channels=2, preset=preset)
        self._surround  = VirtualSurroundProcessor(fs=fs, preset=preset)
        self._reverb    = TheaterReverb(fs=fs, preset=preset)
        self._limiter   = _PeakLimiter(
            threshold=preset["limiter_threshold"],
            release_ms=preset["limiter_release_ms"],
            fs=fs,
        )

    # -- Public API ------------------------------------------------------------

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        Process one audio block through the full theater DSP chain.

        Parameters
        ----------
        block : ndarray, shape (N, 2), dtype float32
            Stereo PCM samples in the range [-1, +1].

        Returns
        -------
        ndarray, shape (N, 2), dtype float32
        """
        if block.ndim == 1:
            block = np.stack([block, block], axis=1)
        if block.shape[1] != 2:
            raise ValueError(f"Expected stereo (2-channel) block, got shape {block.shape}")

        x = block.astype(np.float32, copy=False)

        x = self._eq.process(x)          # Cinema X-curve EQ
        x = self._surround.process(x)    # Virtual surround + binaural
        x = self._reverb.process(x)      # Theater room acoustics
        x = x * self._master_gain        # Master gain trim
        x = self._limiter.process(x)     # Peak limiter

        return x

    def reset(self):
        """Reset all filter/delay states (call on stream restart)."""
        self._eq.reset()
        self._surround.reset()
        self._reverb.reset()
        self._limiter.reset()

    @property
    def fs(self) -> int:
        return self._fs

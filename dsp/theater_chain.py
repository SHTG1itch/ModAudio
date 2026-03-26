"""
TheaterChain - master DSP pipeline.

Signal flow (headphones mode):

  Stereo in (float32, (N, 2))
      |
      v
  +- HarmonicBassEnhancer + AirBandExciter ----+
  |  Psychoacoustic bass synthesis              |
  |  Harmonic sparkle on high frequencies      |
  +--------------------------------------------+
      |
      v
  +- CinemaEqualizer ---------------------------+
  |  X-curve + sub-bass boost + presence       |
  +--------------------------------------------+
      |
      v
  +- BinauralSurroundProcessor -----------------+
  |  Band-split -> 5ch upmix -> HRTF render    |
  |  Brown-Duda head shadow + pinna filters    |
  +--------------------------------------------+
      |
      v
  +- TheaterReverb -----------------------------+
  |  Pre-delay (22ms) + Early reflections      |
  |  + FDN reverb tail (RT60 ~1.3s)            |
  +--------------------------------------------+
      |
      v
  +- MultibandCompressor -----------------------+
  |  4-band theater dynamics                   |
  +--------------------------------------------+
      |
      v
  +- TransientEnhancer -------------------------+
  |  Attack/punch enhancement                  |
  +--------------------------------------------+
      |
      v
  +- Master gain + Peak limiter ----------------+
  +--------------------------------------------+
      |
      v
  Stereo out (float32, (N, 2))

Speaker mode replaces BinauralSurroundProcessor with StereoWidenerProcessor.
"""

from __future__ import annotations
import numpy as np

from .equalizer  import CinemaEqualizer
from .spatializer import make_spatializer
from .reverb      import TheaterReverb
from .dynamics    import MultibandCompressor, TransientEnhancer, PeakLimiter
from .enhancer    import HarmonicBassEnhancer, AirBandExciter


class TheaterChain:
    """
    Complete theater audio processor.

    Parameters
    ----------
    fs     : int  - sample rate (default 48000)
    preset : dict - theater preset (default HEADPHONES_PRESET)
    """

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        if preset is None:
            from config import HEADPHONES_PRESET
            preset = HEADPHONES_PRESET

        self._fs     = fs
        self._preset = preset

        self._master_gain = 10 ** (preset["output_gain_db"] / 20.0)

        # -- Harmonic enhancement (pre-EQ) ------------------------------------
        self._bass_enh = HarmonicBassEnhancer(
            cutoff=120.0,
            drive=preset["bass_harm_drive"],
            level=preset["bass_harm_level"],
            fs=fs,
        )
        self._air_exc = AirBandExciter(
            cutoff=8000.0,
            level=preset["air_exciter_level"],
            fs=fs,
        )

        # -- Cinema EQ --------------------------------------------------------
        self._eq = CinemaEqualizer(fs=fs, num_channels=2, preset=preset)

        # -- Spatialization ---------------------------------------------------
        self._surround = make_spatializer(preset)

        # -- Room reverb ------------------------------------------------------
        self._reverb = TheaterReverb(fs=fs, preset=preset)

        # -- Dynamics ---------------------------------------------------------
        self._comp = MultibandCompressor(
            fs=fs,
            drive=preset["mb_compress_drive"],
        )
        self._trans = TransientEnhancer(
            fs=fs,
            amount=preset["transient_amount"],
        )

        # -- Output -----------------------------------------------------------
        self._limiter = PeakLimiter(
            threshold=preset["limiter_threshold"],
            release_ms=preset["limiter_release_ms"],
            fs=fs,
        )

    # -------------------------------------------------------------------------

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        block : (N, 2) float32  stereo PCM in [-1, +1]
        Returns (N, 2) float32
        """
        if block.ndim == 1:
            block = np.stack([block, block], axis=1)
        if block.shape[1] != 2:
            raise ValueError(f"Expected stereo block, got shape {block.shape}")

        x = block.astype(np.float32, copy=False)

        x = self._bass_enh.process(x)   # harmonic bass synthesis
        x = self._air_exc.process(x)    # air-band exciter
        x = self._eq.process(x)         # cinema X-curve EQ
        x = self._surround.process(x)   # virtual surround / widening
        x = self._reverb.process(x)     # theater room acoustics
        x = self._comp.process(x)       # multi-band compression
        x = self._trans.process(x)      # transient enhancement
        x = x * self._master_gain       # master trim
        x = self._limiter.process(x)    # peak limiter

        return x

    def reset(self):
        for stage in (self._bass_enh, self._air_exc, self._eq, self._surround,
                      self._reverb, self._comp, self._trans, self._limiter):
            stage.reset()

    @property
    def fs(self) -> int:
        return self._fs

    @property
    def mode(self) -> str:
        return self._preset.get("mode", "headphones")

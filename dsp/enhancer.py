"""
Harmonic enhancement processors.

HarmonicBassEnhancer
--------------------
Generates missing psychoacoustic sub-bass for headphones.

Most headphones cannot reproduce 20-60 Hz physically. The "missing fundamental"
psychoacoustic effect allows the brain to reconstruct the perceived bass octave
from its harmonics. We add 2nd/3rd harmonics of the sub-bass to produce the
chest-impact sensation of a cinema subwoofer.

AirBandExciter
--------------
Subtle even-harmonic saturation of the air band (>= 8 kHz). Adds sparkle and
"large tweeter" extension to the sound.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def _sos_stereo_zi(sos):
    zi = sosfilt_zi(sos)
    return np.stack([zi, zi], axis=-1)   # (n_sec, 2, 2)


class _StereoSOS:
    """Stateful SOS filter for stereo (2-channel) blocks."""

    def __init__(self, sos):
        self._sos = sos
        self._zi  = _sos_stereo_zi(sos)

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        for ch in range(2):
            col, self._zi[:, :, ch] = sosfilt(self._sos, x[:, ch],
                                               zi=self._zi[:, :, ch])
            out[:, ch] = col
        return out

    def reset(self):
        self._zi = _sos_stereo_zi(self._sos)


class HarmonicBassEnhancer:
    """
    Psychoacoustic bass extension via sub-harmonic synthesis.

    Parameters
    ----------
    cutoff : float  sub-bass LP cutoff frequency in Hz (default 120)
    drive  : float  saturation drive amount (1.5 - 4.0)
    level  : float  harmonics mix level (0.0 - 1.0)
    fs     : int    sample rate
    """

    def __init__(self, cutoff: float = 120.0, drive: float = 2.8,
                 level: float = 0.50, fs: int = 48000):
        self._drive      = float(drive)
        self._level      = float(level)
        self._tanh_drive = float(np.tanh(drive))

        # Extract sub-bass
        self._lp = _StereoSOS(butter(4, cutoff/(fs/2), btype='low',  output='sos'))
        # High-pass the saturation products to isolate NEW harmonics only
        # (removes the fundamental so we don't double-amplify sub-bass)
        self._hp_harm = _StereoSOS(butter(4, cutoff*1.2/(fs/2), btype='high', output='sos'))

    def process(self, stereo: np.ndarray) -> np.ndarray:
        x   = stereo.astype(np.float64)
        sub = self._lp.process(x)                             # fundamental

        # Soft-clip waveshaper: tanh(drive*x) / tanh(drive) -- range-preserving
        saturated  = np.tanh(self._drive * sub) / self._tanh_drive

        # Isolate harmonics: new content = saturated - original fundamental
        harmonics  = self._hp_harm.process(saturated - sub)  # only above cutoff

        return (x + self._level * harmonics).astype(stereo.dtype)

    def reset(self):
        self._lp.reset()
        self._hp_harm.reset()


class AirBandExciter:
    """
    Subtle harmonic saturation of high-frequency content (>= cutoff Hz).

    Parameters
    ----------
    cutoff : float  air-band start frequency in Hz (default 8000)
    level  : float  exciter mix level (0.0 - 0.5)
    fs     : int    sample rate
    """

    def __init__(self, cutoff: float = 8000.0, level: float = 0.18,
                 fs: int = 48000):
        self._level = float(level)

        # Two separate HP instances: one for extraction, one for harmonic isolation
        self._hp_extract = _StereoSOS(butter(2, cutoff/(fs/2), btype='high', output='sos'))
        self._hp_isolate = _StereoSOS(butter(2, cutoff/(fs/2), btype='high', output='sos'))

    def process(self, stereo: np.ndarray) -> np.ndarray:
        x   = stereo.astype(np.float64)
        air = self._hp_extract.process(x)

        # Even-harmonic saturation: x + 0.25 * x^2 * sign(x)  (2nd harmonic)
        excited = air + 0.25 * (air * air) * np.sign(air)

        # Isolate new harmonics using a separate HP instance (no state aliasing)
        delta = self._hp_isolate.process(excited - air)

        return (x + self._level * delta).astype(stereo.dtype)

    def reset(self):
        self._hp_extract.reset()
        self._hp_isolate.reset()

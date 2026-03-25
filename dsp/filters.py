"""
Stateful biquad IIR filter bank.

All coefficients follow the Audio EQ Cookbook (Robert Bristow-Johnson).
Filters operate on float64 numpy arrays and maintain per-channel state so
they can be called block-by-block in a real-time stream.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi


# -- Coefficient factories ----------------------------------------------------

def _highshelf(fc: float, gain_db: float, q: float, fs: float):
    """High-shelf biquad (b, a) coefficients."""
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    cw = np.cos(w0)
    sw = np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / q - 1.0) + 2.0)

    b0 =  A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 =  A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 =       (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 =  2 *  ((A - 1) - (A + 1) * cw)
    a2 =       (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha

    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _lowshelf(fc: float, gain_db: float, q: float, fs: float):
    """Low-shelf biquad (b, a) coefficients."""
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    cw = np.cos(w0)
    sw = np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / q - 1.0) + 2.0)

    b0 =  A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha)
    b1 =  2 * A * ((A - 1) - (A + 1) * cw)
    b2 =  A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)
    a0 =       (A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha
    a1 = -2 *  ((A - 1) + (A + 1) * cw)
    a2 =       (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha

    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _peaking(fc: float, gain_db: float, q: float, fs: float):
    """Peaking EQ biquad (b, a) coefficients."""
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)

    b0 =  1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 =  1 - alpha * A
    a0 =  1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 =  1 - alpha / A

    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _lowpass(fc: float, q: float, fs: float):
    """2nd-order Butterworth-like low-pass (b, a)."""
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)

    b0 = (1 - cw) / 2
    b1 =  1 - cw
    b2 = (1 - cw) / 2
    a0 =  1 + alpha
    a1 = -2 * cw
    a2 =  1 - alpha

    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _highpass(fc: float, q: float, fs: float):
    """2nd-order high-pass (b, a)."""
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)

    b0 =  (1 + cw) / 2
    b1 = -(1 + cw)
    b2 =  (1 + cw) / 2
    a0 =   1 + alpha
    a1 =  -2 * cw
    a2 =   1 - alpha

    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


# -- Stateful filter class ----------------------------------------------------

class BiquadFilter:
    """
    Single biquad section with persistent per-channel state.

    Parameters
    ----------
    b, a : array-like, shape (3,)
        Feed-forward and feed-back coefficients.
    num_channels : int
        Number of audio channels this filter will process in parallel.
    """

    def __init__(self, b, a, num_channels: int = 2):
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        zi_single = lfilter_zi(self.b, self.a)            # shape (2,)
        self._zi = np.stack([zi_single] * num_channels, axis=-1)  # (2, C)

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Filter block x in-place and update internal state.

        Parameters
        ----------
        x : ndarray, shape (N, C)  float32 or float64
        Returns ndarray same shape.
        """
        x64 = x.astype(np.float64, copy=False)
        out = np.empty_like(x64)
        for c in range(x64.shape[1]):
            col, self._zi[:, c] = lfilter(self.b, self.a, x64[:, c], zi=self._zi[:, c])
            out[:, c] = col
        return out.astype(x.dtype, copy=False)

    def reset(self):
        zi_single = lfilter_zi(self.b, self.a)
        self._zi = np.stack([zi_single] * self._zi.shape[1], axis=-1)


class FilterChain:
    """Cascade of BiquadFilter objects, applied in series."""

    def __init__(self, filters: list[BiquadFilter]):
        self.filters = filters

    def process(self, x: np.ndarray) -> np.ndarray:
        for f in self.filters:
            x = f.process(x)
        return x

    def reset(self):
        for f in self.filters:
            f.reset()


# -- Convenience constructors -------------------------------------------------

def make_highshelf(fc, gain_db, q=0.707, fs=48000, ch=2) -> BiquadFilter:
    b, a = _highshelf(fc, gain_db, q, fs)
    return BiquadFilter(b, a, ch)

def make_lowshelf(fc, gain_db, q=0.707, fs=48000, ch=2) -> BiquadFilter:
    b, a = _lowshelf(fc, gain_db, q, fs)
    return BiquadFilter(b, a, ch)

def make_peaking(fc, gain_db, q=1.0, fs=48000, ch=2) -> BiquadFilter:
    b, a = _peaking(fc, gain_db, q, fs)
    return BiquadFilter(b, a, ch)

def make_lowpass(fc, q=0.707, fs=48000, ch=2) -> BiquadFilter:
    b, a = _lowpass(fc, q, fs)
    return BiquadFilter(b, a, ch)

def make_highpass(fc, q=0.707, fs=48000, ch=2) -> BiquadFilter:
    b, a = _highpass(fc, q, fs)
    return BiquadFilter(b, a, ch)

"""
Theater room acoustics: early reflections + Feedback Delay Network (FDN) reverb tail.

Theater acoustic profile:
  * RT60 ~ 1.2 s  (large reflective room, mostly parallel walls + tiered seating)
  * First reflection arrives ~28 ms after direct sound (30-35 m wall distance)
  * HF decays faster than LF (air absorption: RT60_HF ~ 0.5-0.7 s)

Two-stage model:

  +- Early Reflections (0-80 ms) -----------------------------------------------+
  |  Six discrete reflections at realistic theater delay times.                  |
  |  Each reflection has independent level and a tonal LPF to simulate          |
  |  wall-material absorption (carpet, concrete, screen mesh).                   |
  +------------------------------------------------------------------------------+
                          +
  +- FDN Reverb tail (80 ms -> RT60) -------------------------------------------+
  |  8-line Feedback Delay Network with Hadamard feedback matrix.                |
  |  One-pole absorption filter per delay line controls RT60 and models          |
  |  frequency-dependent energy decay (air absorption).                          |
  +------------------------------------------------------------------------------+

Performance note
---------------
All sample-level Python loops have been replaced with numpy array indexing.
This is valid because every delay line is >= 1481 samples - much longer than
the block size (typically 512 samples) - so there is no within-block feedback
and the entire block can be read/processed/written in a single vectorized pass.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi


# -- Hadamard matrix (N=8) ----------------------------------------------------

def _hadamard8() -> np.ndarray:
    """Normalised 8x8 Hadamard matrix (unitary, flat eigenvalue spectrum)."""
    H = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1,-1, 1,-1, 1,-1, 1,-1],
        [1, 1,-1,-1, 1, 1,-1,-1],
        [1,-1,-1, 1, 1,-1,-1, 1],
        [1, 1, 1, 1,-1,-1,-1,-1],
        [1,-1, 1,-1,-1, 1,-1, 1],
        [1, 1,-1,-1,-1,-1, 1, 1],
        [1,-1,-1, 1,-1, 1, 1,-1],
    ], dtype=np.float64)
    return H / np.sqrt(8)


# Delay lengths chosen to be mutually coprime (avoids mode clustering).
# Spread across 30-110 ms at 48 kHz -> 1440-5280 samples.
# All values > max block size (512) so vectorised block processing is safe.
_FDN_DELAYS = np.array(
    [1481, 1823, 2273, 2683, 3251, 3923, 4637, 5213], dtype=np.int64
)


# -- FDN Reverb ---------------------------------------------------------------

class FDNReverb:
    """
    8-channel Feedback Delay Network reverb.

    Parameters
    ----------
    rt60     : float - reverberation time at low frequencies (seconds)
    rt60_hf  : float - reverberation time at high frequencies (seconds)
    fs       : int   - sample rate
    """

    N = 8

    def __init__(self, rt60: float = 1.2, rt60_hf: float = 0.6, fs: int = 48000):
        self._fs   = fs
        self._H    = _hadamard8()
        self._delays = _FDN_DELAYS.copy()

        bsize = int(np.max(self._delays)) + 1
        self._buf  = np.zeros((bsize, self.N), dtype=np.float64)
        self._bsize = bsize
        self._ptr   = 0

        # Per-delay-line feedback gains from RT60
        delay_s    = self._delays / fs
        self._g_lf = 10.0 ** (-3.0 * delay_s / rt60)
        self._g_hf = 10.0 ** (-3.0 * delay_s / rt60_hf)
        self._tc   = 0.85   # tonal correction coefficient

        # 1-pole filter state, one per delay line  (zi format: shape (1, N))
        self._flt_zi = np.zeros((1, self.N), dtype=np.float64)

        # Input injection: L -> even delay lines, R -> odd delay lines
        self._in_gain = np.zeros((2, self.N), dtype=np.float64)
        self._in_gain[0, 0::2] = 1.0
        self._in_gain[1, 1::2] = 1.0

        # Output mix: (N, 2) - each delay line contributes equally to L/R
        self._out_gain = np.zeros((self.N, 2), dtype=np.float64)
        self._out_gain[0::2, 0] = 1.0
        self._out_gain[1::2, 1] = 1.0
        self._out_gain /= (self.N / 2.0)

    # -- vectorised block processing -------------------------------------------

    def process(self, stereo: np.ndarray) -> np.ndarray:
        """
        stereo : (n, 2) float32/64 - input
        Returns (n, 2) float64  - reverb tail (no dry signal)

        Since all delay lengths > block size, no within-block feedback exists.
        The full block is read from the history buffer, processed through the
        Hadamard + absorption filter, then written back in one vectorised pass.
        """
        n      = stereo.shape[0]
        x      = stereo.astype(np.float64)
        buf    = self._buf
        bsize  = self._bsize
        ptr    = self._ptr

        # -- Read delayed samples -- (n, N)
        # For delay line k, we read the n samples that were written
        # n+delays[k] ... delays[k] steps ago.
        feedback = np.empty((n, self.N), dtype=np.float64)
        for k, d in enumerate(self._delays):
            r_idx = np.arange(ptr - int(d) - n, ptr - int(d), dtype=np.int64) % bsize
            feedback[:, k] = buf[r_idx, k]

        # -- One-pole tonal absorption filter per delay line ------------------
        # H(z) = g_lf / (1 - (g_hf - g_lf)*tc * z^-1)
        # Applied per column using scipy.signal.lfilter (C speed)
        filtered = np.empty_like(feedback)
        for k in range(self.N):
            b = np.array([self._g_lf[k]])
            c = (self._g_hf[k] - self._g_lf[k]) * self._tc
            a = np.array([1.0, -c])
            col, self._flt_zi[:, k] = lfilter(b, a, feedback[:, k],
                                               zi=self._flt_zi[:, k])
            filtered[:, k] = col

        # -- Hadamard mix  (n, N) ---------------------------------------------
        mixed = filtered @ self._H.T   # (n, N)

        # -- Input injection ---------------------------------------------------
        inp = x @ self._in_gain        # (n, N)

        # -- Write to circular buffer ------------------------------------------
        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % bsize
        buf[w_idx] = mixed + inp

        # -- Output mix (n, 2) -------------------------------------------------
        out = filtered @ self._out_gain

        self._ptr = int((ptr + n) % bsize)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._ptr     = 0
        self._flt_zi[:] = 0.0


# -- Early Reflections --------------------------------------------------------

# (delay_ms, gain_L, gain_R, lpf_hz)
_ER_TAPS = [
    ( 9.0,  0.85,  0.60, 12000),
    (14.0,  0.60,  0.80, 10000),
    (22.0, -0.55,  0.55,  9000),
    (31.0,  0.70,  0.65,  8000),
    (47.0,  0.50,  0.45,  7000),
    (68.0,  0.35,  0.40,  6000),
]


class EarlyReflections:
    """
    Vectorised discrete-tap early reflection generator with per-tap 1-pole LPF.
    """

    def __init__(self, fs: int = 48000):
        self._fs    = fs
        self._ntaps = len(_ER_TAPS)

        max_delay = int(max(t[0] for t in _ER_TAPS) * fs / 1000) + 2
        self._bsize = max_delay + 1

        # Circular buffer: shape (bsize, 2) - store stereo input
        self._buf = np.zeros((self._bsize, 2), dtype=np.float64)
        self._ptr = 0

        # Pre-compute tap parameters
        self._tap_delays = np.array(
            [int(round(d * fs / 1000)) for d, *_ in _ER_TAPS], dtype=np.int64
        )
        self._gains = np.array(
            [[gL, gR] for _, gL, gR, _ in _ER_TAPS], dtype=np.float64
        )  # (T, 2)

        # 1-pole LPF coefficients and zi state: (T, 2) for 2 channels
        self._lpf_c  = np.array(
            [np.exp(-2 * np.pi * fc / fs) for *_, fc in _ER_TAPS]
        )  # (T,)
        self._lpf_zi = np.zeros((self._ntaps, 2), dtype=np.float64)

    def process(self, stereo: np.ndarray) -> np.ndarray:
        """
        stereo : (n, 2)  ->  (n, 2) early reflections
        """
        n     = stereo.shape[0]
        x     = stereo.astype(np.float64)
        buf   = self._buf
        bsize = self._bsize
        ptr   = self._ptr

        # Write current block to circular buffer
        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % bsize
        buf[w_idx] = x

        # Accumulate contributions from all taps
        out = np.zeros((n, 2), dtype=np.float64)

        for j, d in enumerate(self._tap_delays):
            r_idx = np.arange(ptr - int(d) - n, ptr - int(d), dtype=np.int64) % bsize
            raw = buf[r_idx] * self._gains[j]    # (n, 2)

            # One-pole LPF per channel: y[n] = (1-c)*x[n] + c*y[n-1]
            c = self._lpf_c[j]
            b_lp = np.array([1.0 - c])
            a_lp = np.array([1.0, -c])

            for ch in range(2):
                col, self._lpf_zi[j, ch:ch+1] = lfilter(
                    b_lp, a_lp, raw[:, ch], zi=[self._lpf_zi[j, ch]]
                )
                out[:, ch] += col

        out *= (1.0 / self._ntaps)
        self._ptr = int((ptr + n) % bsize)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._ptr = 0
        self._lpf_zi[:] = 0.0


# -- Combined theater reverb --------------------------------------------------

class TheaterReverb:
    """
    Combines early reflections and FDN reverb tail.
    output = dry*(1 - rev_mix) + early_ref*er_mix + fdn_tail*rev_mix
    """

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        if preset is None:
            from config import THEATER_PRESET
            preset = THEATER_PRESET

        self._er_mix  = float(preset["early_ref_mix"])
        self._rev_mix = float(preset["reverb_mix"])
        self._er      = EarlyReflections(fs)
        self._fdn     = FDNReverb(rt60=preset["rt60"], rt60_hf=preset["rt60_hf"], fs=fs)

    def process(self, stereo: np.ndarray) -> np.ndarray:
        er  = self._er.process(stereo)
        rev = self._fdn.process(stereo)
        dry_gain = 1.0 - self._rev_mix
        out = (stereo.astype(np.float64) * dry_gain
               + er  * self._er_mix
               + rev * self._rev_mix)
        return out.astype(stereo.dtype)

    def reset(self):
        self._er.reset()
        self._fdn.reset()

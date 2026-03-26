"""
Theater room acoustics: early reflections + Feedback Delay Network (FDN) reverb tail.

Theater acoustic profile:
  * RT60 ~ 1.2-1.4 s  (large reflective room, mostly parallel walls + tiered seating)
  * Pre-delay ~22 ms  (direct sound arrives, then silence, then first reflection)
  * First reflection ~25-30 ms  (side wall near screen)
  * HF decays faster than LF (air absorption: RT60_HF ~ 0.5-0.7 s)

Three-stage model:

  +- Pre-delay (22 ms) ----------------------------------------------------------------+
  |  The gap between direct sound and first reflection is the strongest cue for        |
  |  perceived room size.  Without it, the reverb sounds like a small bathroom.        |
  +------------------------------------------------------------------------------------+
                          +
  +- Early Reflections (25-110 ms) ---------------------------------------------------+
  |  Eight reflections at cinema-geometry-appropriate delays.                         |
  |  Each tap has independent L/R gain and a tonal LPF for surface absorption.        |
  +------------------------------------------------------------------------------------+
                          +
  +- FDN Reverb tail (110 ms -> RT60) ------------------------------------------------+
  |  8-line Feedback Delay Network (Hadamard feedback matrix).                        |
  |  One-pole absorption filter per delay line: RT60 and HF decay are independent.    |
  +------------------------------------------------------------------------------------+

Performance note
----------------
All delay-line reads/writes use numpy array indexing (vectorised per block).
This is valid because every delay line and tap delay is longer than the block size
(512 samples), so no within-block feedback exists.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi


# -- Hadamard matrix (N=8) -----------------------------------------------------

def _hadamard8():
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


# Mutually coprime delay lengths in samples at 48 kHz (30-110 ms)
# All > 512 (max block size) so vectorised block processing is safe.
_FDN_DELAYS = np.array(
    [1481, 1823, 2273, 2683, 3251, 3923, 4637, 5213], dtype=np.int64
)


# -- FDN Reverb ----------------------------------------------------------------

class FDNReverb:
    """8-channel Feedback Delay Network reverb."""

    N = 8

    def __init__(self, rt60=1.3, rt60_hf=0.65, fs=48000):
        self._fs      = fs
        self._H       = _hadamard8()
        self._delays  = _FDN_DELAYS.copy()
        bsize         = int(np.max(self._delays)) + 1
        self._buf     = np.zeros((bsize, self.N), dtype=np.float64)
        self._bsize   = bsize
        self._ptr     = 0

        delay_s       = self._delays / fs
        self._g_lf    = 10.0 ** (-3.0 * delay_s / rt60)
        self._g_hf    = 10.0 ** (-3.0 * delay_s / rt60_hf)
        self._tc      = 0.85
        self._flt_zi  = np.zeros((1, self.N), dtype=np.float64)

        self._in_gain          = np.zeros((2, self.N), dtype=np.float64)
        self._in_gain[0, 0::2] = 1.0
        self._in_gain[1, 1::2] = 1.0

        self._out_gain          = np.zeros((self.N, 2), dtype=np.float64)
        self._out_gain[0::2, 0] = 1.0
        self._out_gain[1::2, 1] = 1.0
        self._out_gain         /= (self.N / 2.0)

    def process(self, stereo):
        n, x = stereo.shape[0], stereo.astype(np.float64)
        buf, bsize, ptr = self._buf, self._bsize, self._ptr

        feedback = np.empty((n, self.N), dtype=np.float64)
        for k, d in enumerate(self._delays):
            r_idx = np.arange(ptr - int(d) - n, ptr - int(d), dtype=np.int64) % bsize
            feedback[:, k] = buf[r_idx, k]

        filtered = np.empty_like(feedback)
        for k in range(self.N):
            b = np.array([self._g_lf[k]])
            c = (self._g_hf[k] - self._g_lf[k]) * self._tc
            a = np.array([1.0, -c])
            col, self._flt_zi[:, k] = lfilter(b, a, feedback[:, k],
                                               zi=self._flt_zi[:, k])
            filtered[:, k] = col

        mixed = filtered @ self._H.T
        inp   = x @ self._in_gain
        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % bsize
        buf[w_idx] = mixed + inp

        out       = filtered @ self._out_gain
        self._ptr = int((ptr + n) % bsize)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._ptr     = 0
        self._flt_zi[:] = 0.0


# -- Early Reflections (theater geometry) --------------------------------------
#
# Delays based on a 500-seat commercial cinema (~25 m wide, ~35 m long):
#   Screen-end side wall:  ~25-32 ms
#   Ceiling (low point):   ~38 ms
#   Mid-hall side walls:   ~47-56 ms
#   Second ceiling bounce: ~68 ms
#   Rear hall early:       ~85 ms
#   Rear side:             ~110 ms
#
# All delays > 512 samples (10.7 ms) -> vectorised block reads are safe.
#
# (delay_ms, gain_L, gain_R, lpf_hz)
_ER_TAPS = [
    ( 25.0,  0.80,  0.55, 10000),   # screen-end left side wall
    ( 32.0,  0.55,  0.78,  9500),   # screen-end right side wall
    ( 38.0, -0.65,  0.60,  9000),   # ceiling (polarity inversion = diffusion)
    ( 47.0,  0.68,  0.62,  8000),   # mid-hall left side wall
    ( 56.0,  0.52,  0.60,  7500),   # mid-hall right side wall
    ( 68.0, -0.48,  0.50,  7000),   # second ceiling bounce
    ( 85.0,  0.40,  0.38,  6500),   # rear-hall early
    (110.0,  0.30,  0.32,  5500),   # rear side reflection
]


class EarlyReflections:
    """Vectorised discrete-tap early reflections with per-tap 1-pole LPF."""

    def __init__(self, fs=48000):
        self._fs    = fs
        self._ntaps = len(_ER_TAPS)
        max_delay   = int(max(t[0] for t in _ER_TAPS) * fs / 1000) + 2
        self._bsize = max_delay + 1
        self._buf   = np.zeros((self._bsize, 2), dtype=np.float64)
        self._ptr   = 0

        self._tap_delays = np.array(
            [int(round(d * fs / 1000)) for d, *_ in _ER_TAPS], dtype=np.int64
        )
        self._gains = np.array([[gL, gR] for _, gL, gR, _ in _ER_TAPS])
        self._lpf_c = np.array(
            [np.exp(-2 * np.pi * fc / fs) for *_, fc in _ER_TAPS]
        )
        self._lpf_zi = np.zeros((self._ntaps, 2), dtype=np.float64)

    def process(self, stereo):
        n, x = stereo.shape[0], stereo.astype(np.float64)
        buf, bsize, ptr = self._buf, self._bsize, self._ptr

        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % bsize
        buf[w_idx] = x

        out = np.zeros((n, 2), dtype=np.float64)
        for j, d in enumerate(self._tap_delays):
            r_idx = np.arange(ptr - int(d) - n, ptr - int(d), dtype=np.int64) % bsize
            raw   = buf[r_idx] * self._gains[j]
            c     = self._lpf_c[j]
            b_lp  = np.array([1.0 - c])
            a_lp  = np.array([1.0, -c])
            for ch in range(2):
                col, self._lpf_zi[j, ch:ch+1] = lfilter(
                    b_lp, a_lp, raw[:, ch], zi=[self._lpf_zi[j, ch]]
                )
                out[:, ch] += col

        out *= 1.0 / self._ntaps
        self._ptr = int((ptr + n) % bsize)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._ptr = 0
        self._lpf_zi[:] = 0.0


# -- Pre-delay (simple delay line) --------------------------------------------

class PreDelay:
    """
    Inserts silence between the direct sound and reverb entry.
    This gap is the single strongest perceptual cue for large room size.
    Without pre-delay, a reverb sounds like a small bathroom.
    """

    def __init__(self, delay_ms=22.0, fs=48000):
        self._delay  = int(round(delay_ms * fs / 1000))
        max_d        = max(self._delay, 1)
        self._buf    = np.zeros((max_d + 1, 2), dtype=np.float64)
        self._bsize  = len(self._buf)
        self._ptr    = 0

    def process(self, stereo):
        n, x = stereo.shape[0], stereo.astype(np.float64)
        buf, bsize, ptr = self._buf, self._bsize, self._ptr

        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % bsize
        buf[w_idx] = x

        r_idx = np.arange(ptr - self._delay - n, ptr - self._delay,
                          dtype=np.int64) % bsize
        out      = buf[r_idx].copy()
        self._ptr = int((ptr + n) % bsize)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._ptr = 0


# -- Combined theater reverb --------------------------------------------------

class TheaterReverb:
    """
    Full reverb chain:  direct_sound -> pre_delay -> (ER + FDN) -> mix
    """

    def __init__(self, fs=48000, preset=None):
        if preset is None:
            from config import THEATER_PRESET
            preset = THEATER_PRESET

        predelay_ms   = float(preset.get("reverb_predelay_ms", 22.0))
        self._er_mix  = float(preset["early_ref_mix"])
        self._rev_mix = float(preset["reverb_mix"])

        self._pre = PreDelay(predelay_ms, fs)
        self._er  = EarlyReflections(fs)
        self._fdn = FDNReverb(rt60=preset["rt60"], rt60_hf=preset["rt60_hf"], fs=fs)

    def process(self, stereo):
        # Pre-delay the signal fed into ER and FDN
        pd  = self._pre.process(stereo)
        er  = self._er.process(pd)
        rev = self._fdn.process(pd)

        dry      = 1.0 - self._rev_mix
        out      = (stereo.astype(np.float64) * dry
                    + er  * self._er_mix
                    + rev * self._rev_mix)
        return out.astype(stereo.dtype)

    def reset(self):
        self._pre.reset()
        self._er.reset()
        self._fdn.reset()

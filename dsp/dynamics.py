"""
Theater dynamics processing suite.

MultibandCompressor  - 4-band feed-forward RMS compressor (theater loudness)
TransientEnhancer    - Vectorised transient attack enhancer
PeakLimiter          - Output ceiling protection
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, lfilter, lfilter_zi


# -- Crossover filter design ---------------------------------------------------

def _lr2_lp(fc, fs):
    return butter(2, fc / (fs/2), btype='low',  output='sos')

def _lr2_hp(fc, fs):
    return butter(2, fc / (fs/2), btype='high', output='sos')


def _sos_stereo_zi(sos):
    """Initial condition for a stereo (2-ch) SOS filter."""
    zi1 = sosfilt_zi(sos)            # (n_sec, 2)
    return np.stack([zi1, zi1], axis=-1)   # (n_sec, 2, 2)


def _apply_sos_stereo(sos, x, zi):
    """Apply SOS filter to stereo block in-place; returns (filtered, new_zi)."""
    out = np.empty_like(x)
    for ch in range(2):
        col, zi[:, :, ch] = sosfilt(sos, x[:, ch], zi=zi[:, :, ch])
        out[:, ch] = col
    return out, zi


# -- Per-band compressor -------------------------------------------------------

class _Band:
    def __init__(self, threshold_db, ratio, attack_ms, release_ms, makeup_db, fs):
        self._thr     = threshold_db
        self._ratio   = ratio
        self._attack  = np.exp(-1.0 / (attack_ms  * 1e-3 * fs))
        self._release = np.exp(-1.0 / (release_ms * 1e-3 * fs))
        self._makeup  = 10 ** (makeup_db / 20.0)
        self._gain_db = 0.0

    def process(self, band: np.ndarray) -> np.ndarray:
        # Fast RMS: single dot-product, no intermediate array
        ss   = float(np.einsum('ij,ij', band, band))
        rms_db = 10.0 * np.log10(ss / band.size + 1e-24)

        if rms_db > self._thr:
            gr = self._thr + (rms_db - self._thr) / self._ratio - rms_db
        else:
            gr = 0.0

        if gr < self._gain_db:
            self._gain_db = self._attack  * self._gain_db + (1 - self._attack)  * gr
        else:
            self._gain_db = self._release * self._gain_db + (1 - self._release) * gr

        return band * (10 ** (self._gain_db / 20.0) * self._makeup)

    def reset(self):
        self._gain_db = 0.0


# Band boundaries [Hz] and parameters
_XOVER = [200.0, 1200.0, 8000.0]

# (threshold_db, ratio, attack_ms, release_ms, makeup_db)
_BAND_PARAMS = [
    (-20.0, 3.5, 40.0, 250.0, 1.5),
    (-22.0, 2.5, 15.0, 120.0, 1.0),
    (-24.0, 3.0,  8.0,  80.0, 1.5),
    (-18.0, 2.0,  5.0,  50.0, 0.5),
]


class MultibandCompressor:
    """
    4-band compressor.
    drive > 1.0 lowers thresholds and increases gain reduction.
    """

    def __init__(self, fs: int = 48000, drive: float = 1.6):
        self._fs = fs

        # Pre-build SOS for each crossover
        self._xsos = [(_lr2_lp(f, fs), _lr2_hp(f, fs)) for f in _XOVER]

        # Filter states: list of [lp_zi, hp_zi], each (n_sec, 2, 2)
        self._xzi = [
            [_sos_stereo_zi(lp), _sos_stereo_zi(hp)]
            for lp, hp in self._xsos
        ]

        # Build band processors with drive-adjusted thresholds
        thresh_offset = (drive - 1.0) * (-3.0)
        self._bands = [
            _Band(thr + thresh_offset, ratio, atk, rel, mu, fs)
            for thr, ratio, atk, rel, mu in _BAND_PARAMS
        ]

    def _split(self, x, idx):
        """Apply crossover at index idx -> (lo, hi)."""
        lp, hp = self._xsos[idx]
        lp_zi, hp_zi = self._xzi[idx]
        lo, lp_zi = _apply_sos_stereo(lp, x, lp_zi)
        hi, hp_zi = _apply_sos_stereo(hp, x, hp_zi)
        self._xzi[idx] = [lp_zi, hp_zi]
        return lo, hi

    def process(self, stereo: np.ndarray) -> np.ndarray:
        x = stereo.astype(np.float64)
        lo1, hi1 = self._split(x,   0)
        lo2, hi2 = self._split(hi1, 1)
        lo3, hi3 = self._split(hi2, 2)
        bands = [lo1, lo2, lo3, hi3]
        out = sum(self._bands[i].process(b) for i, b in enumerate(bands))
        return out.astype(stereo.dtype)

    def reset(self):
        for b in self._bands:
            b.reset()
        self._xzi = [
            [_sos_stereo_zi(lp), _sos_stereo_zi(hp)]
            for lp, hp in self._xsos
        ]


# -- Transient enhancer (fully vectorised) ------------------------------------

class TransientEnhancer:
    """
    Detects transient attacks (fast envelope > slow envelope) and applies a
    proportional gain boost.  Fully vectorised using scipy.signal.lfilter.

    amount = 0.0 -> bypass
    amount = 1.0 -> aggressive cinema-level punch enhancement
    """

    def __init__(self, fs: int = 48000, amount: float = 0.5):
        self._amount = float(amount)

        # Fast envelope IIR: y[n] = (1-a)*y[n-1] + a*|x[n]|, tc = 3 ms
        a_fast = 1.0 - np.exp(-1.0 / (3e-3 * fs))
        # Slow envelope IIR: tc = 80 ms
        a_slow = 1.0 - np.exp(-1.0 / (80e-3 * fs))

        self._b_fast = np.array([a_fast])
        self._a_fast = np.array([1.0, -(1.0 - a_fast)])
        self._b_slow = np.array([a_slow])
        self._a_slow = np.array([1.0, -(1.0 - a_slow)])

        zi0_f = lfilter_zi(self._b_fast, self._a_fast)
        zi0_s = lfilter_zi(self._b_slow, self._a_slow)
        self._zi_fast = zi0_f * 0.0   # shape (1,)
        self._zi_slow = zi0_s * 0.0

    def process(self, stereo: np.ndarray) -> np.ndarray:
        if self._amount < 0.01:
            return stereo

        x = stereo.astype(np.float64)

        # Mono envelope: mean absolute across channels  (N,)
        env = np.abs(x).mean(axis=1)

        # Fast and slow IIR smoothing (scipy lfilter - C speed, no Python loop)
        fast_env, self._zi_fast = lfilter(self._b_fast, self._a_fast, env,
                                           zi=self._zi_fast)
        slow_env, self._zi_slow = lfilter(self._b_slow, self._a_slow, env,
                                           zi=self._zi_slow)

        # Transient ratio: how much faster is the fast envelope than the slow?
        transient = np.maximum(0.0, fast_env - slow_env) / (slow_env + 1e-8)

        # Gain: 1.0 on sustain, up to 1 + amount*3 on strong transients
        gain = 1.0 + self._amount * np.minimum(transient, 1.0) * 3.0

        return (x * gain[:, np.newaxis]).astype(stereo.dtype)

    def reset(self):
        self._zi_fast = lfilter_zi(self._b_fast, self._a_fast) * 0.0
        self._zi_slow = lfilter_zi(self._b_slow, self._a_slow) * 0.0


# -- Peak limiter --------------------------------------------------------------

class PeakLimiter:
    """One-pole envelope peak limiter — vectorised."""

    def __init__(self, threshold: float = 0.93, release_ms: float = 80.0,
                 fs: int = 48000):
        self._threshold = threshold
        self._release   = np.exp(-1.0 / (release_ms * 1e-3 * fs))
        self._env       = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        x64  = x.astype(np.float64)
        peak = np.abs(x64).max(axis=1)
        out  = np.empty_like(x64)
        thr, rel, env = self._threshold, self._release, self._env
        for i in range(len(peak)):
            env = max(peak[i], env * rel)
            out[i] = x64[i] * thr / max(env, thr)
        self._env = env
        return out.astype(x.dtype)

    def reset(self):
        self._env = 0.0

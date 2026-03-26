"""
Binaural HRTF renderer using the Brown-Duda (1998) spherical head model
with multi-notch pinna simulation.

This replaces the simple Woodworth ITD + broadband ILD model with a
perceptually validated, frequency-dependent model that dramatically improves
sound externalization (sounds feel outside the head, not inside it).

Model components per virtual speaker:

  1. ITD  - Woodworth spherical-head formula
             Delay in samples for the contralateral (far) ear.

  2. Head shadow - First-order high-shelf LOW-PASS applied to contralateral ear.
             Models acoustic diffraction around the head:
             * Bass (<600 Hz) diffracts freely -> minimal attenuation
             * Treble (>2 kHz) is blocked     -> up to -20 dB at 90 deg azimuth
             Characteristic frequency: f_0 = c/(2*pi*a) ~= 624 Hz

  3. Concha resonance - +3.5 dB peak at 3.5 kHz on the ipsilateral (near) ear.
             The concha bowl of the pinna resonates at 3-4 kHz for direct-path
             sound, boosting presence and helping forward localisation.

  4. Pinna notch 1 - Notch at 8.5 kHz (both ears).
             The most important elevation/externalization cue. The pinna creates
             a deep notch near 8-10 kHz that shifts with elevation and is the
             primary cue preventing in-head localisation.

  5. Pinna notch 2 - Secondary notch at 11 kHz (both ears).
             Further improves externalization; typical for frontal sources.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from config import HEAD_RADIUS_M, SOUND_SPEED_MS


# -- Internal coefficient helpers ----------------------------------------------

def _highshelf_ba(fc, gain_db, q, fs):
    """High-shelf biquad (b, a)."""
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0/A) * (1.0/q - 1.0) + 2.0)
    b0 =  A*((A+1)+(A-1)*cw+2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1)+(A+1)*cw)
    b2 =  A*((A+1)+(A-1)*cw-2*np.sqrt(A)*alpha)
    a0 =    (A+1)-(A-1)*cw+2*np.sqrt(A)*alpha
    a1 =  2*((A-1)-(A+1)*cw)
    a2 =    (A+1)-(A-1)*cw-2*np.sqrt(A)*alpha
    return np.array([b0,b1,b2])/a0, np.array([1.0, a1/a0, a2/a0])

def _peaking_ba(fc, gain_db, q, fs):
    """Peaking EQ biquad (b, a)."""
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2*q)
    b0 = 1 + alpha*A;  b1 = -2*np.cos(w0);  b2 = 1 - alpha*A
    a0 = 1 + alpha/A;  a1 = -2*np.cos(w0);  a2 = 1 - alpha/A
    return np.array([b0,b1,b2])/a0, np.array([1.0, a1/a0, a2/a0])


# -- Mono stateful filter ------------------------------------------------------

class _MonoFilter:
    """Single biquad, mono (N,) in/out, maintains state across blocks."""

    def __init__(self, b, a):
        self._b  = np.asarray(b, dtype=np.float64)
        self._a  = np.asarray(a, dtype=np.float64)
        self._zi = lfilter_zi(self._b, self._a).copy()

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self._zi = lfilter(self._b, self._a, x.astype(np.float64), zi=self._zi)
        return y

    def reset(self):
        self._zi = lfilter_zi(self._b, self._a).copy()


class _MonoFilterChain:
    def __init__(self, filters):
        self._filters = filters

    def process(self, x):
        for f in self._filters:
            x = f.process(x)
        return x

    def reset(self):
        for f in self._filters:
            f.reset()


# -- Delay line (vectorised) ---------------------------------------------------

class _DelayLine:
    MAX_DELAY = 128   # samples - covers max Woodworth ITD (~25 samples at 48 kHz)

    def __init__(self):
        self._buf  = np.zeros(self.MAX_DELAY + 2, dtype=np.float64)
        self._size = len(self._buf)
        self._pos  = 0

    def process(self, x: np.ndarray, delay: int) -> np.ndarray:
        n, size, ptr = len(x), self._size, self._pos
        np.put(self._buf, np.arange(ptr, ptr+n) % size, x)
        r_idx = np.arange(ptr - delay - n, ptr - delay, dtype=np.int64) % size
        out = self._buf[r_idx].copy()
        self._pos = int((ptr + n) % size)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._pos = 0


# -- Woodworth ITD -------------------------------------------------------------

def _woodworth_itd_samples(azimuth_deg: float, fs: int) -> int:
    """
    Interaural time delay in samples (Woodworth 1938 spherical-head model).
    Positive = left ear delayed (sound from right side).
    """
    theta = np.deg2rad(np.clip(abs(azimuth_deg), 0, 180))
    if theta <= np.pi / 2:
        tau = (HEAD_RADIUS_M / SOUND_SPEED_MS) * (np.sin(theta) + theta)
    else:
        theta_m = np.pi - theta
        tau = (HEAD_RADIUS_M / SOUND_SPEED_MS) * (np.sin(theta_m) + theta_m)
    sign = 1 if azimuth_deg >= 0 else -1
    return sign * int(round(tau * fs))


# -- Per-speaker binaural renderer (Brown-Duda model) -------------------------

class BinauralSpeakerRenderer:
    """
    Renders one mono virtual speaker to a stereo (L_ear, R_ear) pair.

    Uses the Brown-Duda (1998) spherical head model which gives perceptually
    validated, frequency-dependent ILD - a major improvement over simple
    broadband gain differences.
    """

    def __init__(self, azimuth_deg: float, elevation_deg: float = 0.0,
                 fs: int = 48000):
        self._az = azimuth_deg
        self._el = elevation_deg
        self._fs = fs

        # -- ITD ---------------------------------------------------------------
        itd = _woodworth_itd_samples(azimuth_deg, fs)
        self._left_delay  = max(0,  itd)
        self._right_delay = max(0, -itd)
        self._dl_L = _DelayLine()
        self._dl_R = _DelayLine()

        # -- Head shadow (contralateral ear) -----------------------------------
        # Brown-Duda: contralateral ear sees a high-shelf LP.
        # Shadow grows with sin(azimuth) - max at 90 deg (side), zero at 0/180.
        shadow_db = -20.0 * abs(np.sin(np.deg2rad(abs(azimuth_deg))))
        b_shad, a_shad = _highshelf_ba(1000, shadow_db, q=0.55, fs=fs)

        # -- Concha resonance (ipsilateral ear) --------------------------------
        # The concha bowl boosts 3-4 kHz for direct-path (near-ear) sounds.
        b_con, a_con = _peaking_ba(3500, 3.5, q=1.6, fs=fs)

        # -- Pinna notch 1 (both ears) -----------------------------------------
        # Main externalization cue: deep notch around 8-9 kHz.
        b_n1, a_n1 = _peaking_ba(8500, -10.0, q=3.0, fs=fs)

        # -- Pinna notch 2 (both ears) -----------------------------------------
        # Secondary notch: helps prevent "inside the head" sensation.
        b_n2, a_n2 = _peaking_ba(11000, -6.0, q=2.5, fs=fs)

        # Assign filters: ipsilateral = direct/near ear, contralateral = far ear
        # For az >= 0: right ear is ipsilateral (near), left is contralateral
        if azimuth_deg >= 0:
            self._filt_L = _MonoFilterChain([
                _MonoFilter(b_shad, a_shad),  # head shadow
                _MonoFilter(b_n1,   a_n1),    # pinna notch
                _MonoFilter(b_n2,   a_n2),
            ])
            self._filt_R = _MonoFilterChain([
                _MonoFilter(b_con,  a_con),   # concha resonance
                _MonoFilter(b_n1,   a_n1),    # pinna notch (both ears)
                _MonoFilter(b_n2,   a_n2),
            ])
        else:
            self._filt_L = _MonoFilterChain([
                _MonoFilter(b_con,  a_con),
                _MonoFilter(b_n1,   a_n1),
                _MonoFilter(b_n2,   a_n2),
            ])
            self._filt_R = _MonoFilterChain([
                _MonoFilter(b_shad, a_shad),
                _MonoFilter(b_n1,   a_n1),
                _MonoFilter(b_n2,   a_n2),
            ])

    def process(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        mono : (N,) float32/64
        Returns (left_ear, right_ear) each (N,) float64
        """
        m = mono.astype(np.float64)
        left  = self._dl_L.process(m, self._left_delay)
        right = self._dl_R.process(m, self._right_delay)
        left  = self._filt_L.process(left)
        right = self._filt_R.process(right)
        return left, right

    def reset(self):
        self._dl_L.reset()
        self._dl_R.reset()
        self._filt_L.reset()
        self._filt_R.reset()

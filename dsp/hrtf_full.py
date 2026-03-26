"""
Full-sphere binaural HRTF renderer.

Extends Brown-Duda (1998) with:

  1. Algazi (2001) elevation-aware ITD
       τ = (a/c) * [cos(φ)·sin(θ) + θ]   (horizontal plane Woodworth term,
                                            scaled by cos(elevation))

  2. Elevation-dependent primary pinna notch (Blauert 1997)
       Notch frequency rises with elevation:
         -45° → ~6.2 kHz   0° → ~8.5 kHz   +30° → ~10 kHz   +60° → 11.5 kHz
       This is the #1 cue for elevation perception.

  3. Front / back spectral discrimination (~4.5 kHz)
       Front sources: ear-canal resonance adds +2.5 dB presence peak.
       Rear  sources: pinna folds create -4.0 dB notch at same frequency.

  4. Elevation-modulated head shadow
       Shadow strength reduces with elevation (cos-weighted) because sound
       can arrive over/under the head at non-horizontal angles.

  5. Concha resonance on ipsilateral (near) ear
       +3.5 dB peak at ~3.2 kHz — frontal presence and localisation cue.

Usage
-----
    r = FullSphereHRTFRenderer(azimuth_deg=-110.0, elevation_deg=5.0, fs=48000)
    left, right = r.process(mono_block)   # (N,) → (N,), (N,) float64
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from config import HEAD_RADIUS_M, SOUND_SPEED_MS


# ---------------------------------------------------------------------------
# Coefficient helpers (matching hrtf.py style)
# ---------------------------------------------------------------------------

def _highshelf_ba(fc: float, gain_db: float, q: float, fs: float):
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt((A + 1.0/A) * (1.0/q - 1.0) + 2.0)
    b0 =  A * ((A+1) + (A-1)*cw + 2*np.sqrt(A)*alpha)
    b1 = -2 * A * ((A-1) + (A+1)*cw)
    b2 =  A * ((A+1) + (A-1)*cw - 2*np.sqrt(A)*alpha)
    a0 =       (A+1) - (A-1)*cw + 2*np.sqrt(A)*alpha
    a1 =  2 *  ((A-1) - (A+1)*cw)
    a2 =       (A+1) - (A-1)*cw - 2*np.sqrt(A)*alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1/a0, a2/a0])


def _peaking_ba(fc: float, gain_db: float, q: float, fs: float):
    A  = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    b0 = 1 + alpha*A;  b1 = -2*np.cos(w0);  b2 = 1 - alpha*A
    a0 = 1 + alpha/A;  a1 = -2*np.cos(w0);  a2 = 1 - alpha/A
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1/a0, a2/a0])


def _lowpass_ba(fc: float, q: float, fs: float):
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b0 = (1 - cw) / 2;  b1 = 1 - cw;  b2 = (1 - cw) / 2
    a0 = 1 + alpha;      a1 = -2*cw;   a2 = 1 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1/a0, a2/a0])


# ---------------------------------------------------------------------------
# Internal filter helpers (same pattern as hrtf.py _MonoFilter / chain)
# ---------------------------------------------------------------------------

class _MonoFilter:
    """Single stateful biquad, mono (N,) in/out."""
    def __init__(self, b, a):
        self._b  = np.asarray(b, dtype=np.float64)
        self._a  = np.asarray(a, dtype=np.float64)
        self._zi = lfilter_zi(self._b, self._a).copy()

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self._zi = lfilter(self._b, self._a, x.astype(np.float64), zi=self._zi)
        return y

    def reset(self):
        self._zi = lfilter_zi(self._b, self._a).copy()


class _FilterChain:
    def __init__(self, filters: list):
        self._filters = filters

    def process(self, x: np.ndarray) -> np.ndarray:
        for f in self._filters:
            x = f.process(x)
        return x

    def reset(self):
        for f in self._filters:
            f.reset()


# ---------------------------------------------------------------------------
# Delay line (vectorised ring buffer, matching hrtf.py _DelayLine)
# ---------------------------------------------------------------------------

class _DelayLine:
    MAX_DELAY = 256   # samples — covers ~5.3 ms max ITD with safety margin

    def __init__(self):
        self._buf  = np.zeros(self.MAX_DELAY + 512, dtype=np.float64)
        self._size = len(self._buf)
        self._pos  = 0

    def process(self, x: np.ndarray, delay: int) -> np.ndarray:
        n, sz, ptr = len(x), self._size, self._pos
        np.put(self._buf, np.arange(ptr, ptr + n) % sz, x)
        r_idx = np.arange(ptr - delay - n, ptr - delay, dtype=np.int64) % sz
        out = self._buf[r_idx].copy()
        self._pos = int((ptr + n) % sz)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._pos    = 0


# ---------------------------------------------------------------------------
# ITD
# ---------------------------------------------------------------------------

def _algazi_itd_samples(azimuth_deg: float, elevation_deg: float, fs: int) -> int:
    """
    Interaural time delay in samples (Algazi 2001 model with elevation).

    τ = (a/c) * [cos(φ)·sin(|θ|) + |θ|]   for |θ| ≤ 90°
    τ = (a/c) * [cos(φ)·sin(π−|θ|) + (π−|θ|)]   for |θ| > 90°

    Positive return → left ear delayed (source from right side).
    """
    az_abs = np.deg2rad(np.clip(abs(azimuth_deg), 0, 180))
    el     = np.deg2rad(np.clip(elevation_deg, -90, 90))
    cos_el = np.cos(el)

    if az_abs <= np.pi / 2:
        tau = (HEAD_RADIUS_M / SOUND_SPEED_MS) * (cos_el * np.sin(az_abs) + az_abs)
    else:
        az_m = np.pi - az_abs
        tau = (HEAD_RADIUS_M / SOUND_SPEED_MS) * (cos_el * np.sin(az_m) + az_m)

    samples = int(round(tau * fs))
    samples = min(samples, _DelayLine.MAX_DELAY - 1)
    sign    = 1 if azimuth_deg >= 0 else -1
    return sign * samples


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

class FullSphereHRTFRenderer:
    """
    Full-sphere binaural HRTF for one virtual speaker.

    Parameters
    ----------
    azimuth_deg   : horizontal angle, degrees
                    Convention: positive = right, negative = left.
    elevation_deg : vertical angle, degrees (+90 = directly above).
    fs            : sample rate in Hz.
    """

    def __init__(self, azimuth_deg: float, elevation_deg: float = 0.0,
                 fs: int = 48000):
        self._az = azimuth_deg
        self._el = elevation_deg
        self._fs = fs

        # -- ITD ----------------------------------------------------------
        itd = _algazi_itd_samples(azimuth_deg, elevation_deg, fs)
        self._left_delay  = max(0,  itd)
        self._right_delay = max(0, -itd)
        self._dl_L = _DelayLine()
        self._dl_R = _DelayLine()

        # -- Head shadow (contralateral ear) ------------------------------
        # Elevation weakens the shadow because the acoustic path around
        # the head is shorter at non-horizontal angles.
        az_r   = np.deg2rad(abs(azimuth_deg))
        el_r   = np.deg2rad(abs(elevation_deg))
        shadow_db = -20.0 * abs(np.sin(az_r)) * np.cos(el_r * 0.7)
        b_shad, a_shad = _highshelf_ba(900.0, shadow_db, q=0.52, fs=fs)

        # -- Concha resonance (ipsilateral ear) ---------------------------
        b_con, a_con = _peaking_ba(3200.0, 3.5, q=1.8, fs=fs)

        # -- Primary pinna notch ------------------------------------------
        # Frequency shifts with elevation (Blauert 1997).
        # Linear approximation: 8500 + elev_deg * 50 Hz.
        #   0° → 8500 Hz,  +30° → 10 000 Hz,  −30° → 7 000 Hz
        n1_hz = 8500.0 + float(np.clip(elevation_deg, -45.0, 65.0)) * 50.0
        n1_hz = float(np.clip(n1_hz, 5500.0, 13500.0))
        n1_hz = min(n1_hz, fs * 0.45)          # stay below Nyquist
        b_n1, a_n1 = _peaking_ba(n1_hz, -11.0, q=3.2, fs=fs)

        # -- Secondary pinna notch (both ears, ~11 kHz) -------------------
        n2_hz = min(11_000.0, fs * 0.45)
        b_n2, a_n2 = _peaking_ba(n2_hz, -7.0, q=2.5, fs=fs)

        # -- Front / back spectral cue at ~4.5 kHz -----------------------
        # Frontal ear-canal resonance: +2.5 dB boost.
        # Rear pinna shadow: −4.0 dB notch.
        is_front = abs(azimuth_deg) <= 90.0
        fb_db    = 2.5 if is_front else -4.0
        b_fb, a_fb = _peaking_ba(4500.0, fb_db, q=1.8, fs=fs)

        # -- Assign filter chains per ear ---------------------------------
        # Source from right (az ≥ 0): right ear is ipsilateral (near),
        # left ear is contralateral (far / shadowed).
        from_right = azimuth_deg >= 0
        if from_right:
            self._filt_L = _FilterChain([          # contralateral
                _MonoFilter(b_shad, a_shad),
                _MonoFilter(b_n1,   a_n1),
                _MonoFilter(b_n2,   a_n2),
                _MonoFilter(b_fb,   a_fb),
            ])
            self._filt_R = _FilterChain([          # ipsilateral
                _MonoFilter(b_con,  a_con),
                _MonoFilter(b_n1,   a_n1),
                _MonoFilter(b_n2,   a_n2),
                _MonoFilter(b_fb,   a_fb),
            ])
        else:
            self._filt_L = _FilterChain([          # ipsilateral
                _MonoFilter(b_con,  a_con),
                _MonoFilter(b_n1,   a_n1),
                _MonoFilter(b_n2,   a_n2),
                _MonoFilter(b_fb,   a_fb),
            ])
            self._filt_R = _FilterChain([          # contralateral
                _MonoFilter(b_shad, a_shad),
                _MonoFilter(b_n1,   a_n1),
                _MonoFilter(b_n2,   a_n2),
                _MonoFilter(b_fb,   a_fb),
            ])

    # ------------------------------------------------------------------

    def process(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        mono : (N,) float32 or float64
        Returns (left_ear, right_ear) each (N,) float64.
        """
        m = mono.astype(np.float64)
        l = self._dl_L.process(m, self._left_delay)
        r = self._dl_R.process(m, self._right_delay)
        l = self._filt_L.process(l)
        r = self._filt_R.process(r)
        return l, r

    def reset(self):
        self._dl_L.reset()
        self._dl_R.reset()
        self._filt_L.reset()
        self._filt_R.reset()

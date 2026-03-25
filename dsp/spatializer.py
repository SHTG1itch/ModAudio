"""
Virtual surround spatialization for stereo headphones.

Architecture (hybrid of user's frequency-band idea + HRTF binaural rendering):

  Stereo in
      |
      +- [Sub-bass LPF < 120 Hz]  -> center / omnidirectional  (user's idea:
      |                                bass is non-directional in theaters)
      |
      +- [Mid-band 120-8000 Hz]   -> matrix-decoded 5-channel upmix:
      |                               L, C, R, LS, RS
      |                               Each channel -> HRTF binaural render
      |
      +- [Air band > 8000 Hz]     -> extra stereo widening
                                     (user's idea: HF feels spatially diffuse)

  Mix all bands -> stereo binaural output

HRTF model - analytical spherical-head + ILD model (no external data files):
  * ITD  : Woodworth spherical-head formula  (interaural time delay)
  * ILD  : frequency-dependent gain model    (interaural level difference)
  * Pinna: simplified notch filter at ~8 kHz (elevation / externalization cue)

Speaker layout - ITU-R BS.775 / 5.0 surround:
  L: -30 deg   R: +30 deg   C: 0 deg   LS: -110 deg   RS: +110 deg
"""

from __future__ import annotations
import numpy as np
from .filters import make_lowpass, make_highpass, make_peaking, BiquadFilter


# -- Constants -----------------------------------------------------------------

HEAD_RADIUS  = 0.0875   # m
SOUND_SPEED  = 343.0    # m/s


# -- Woodworth ITD ------------------------------------------------------------

def _woodworth_itd_samples(azimuth_deg: float, fs: int) -> int:
    """
    Return the interaural time delay in samples using the Woodworth (1938)
    spherical-head model.

    Positive result -> left ear is delayed (sound comes from the right).
    """
    theta = np.deg2rad(np.clip(abs(azimuth_deg), 0, 180))
    if theta <= np.pi / 2:
        tau = (HEAD_RADIUS / SOUND_SPEED) * (np.sin(theta) + theta)
    else:
        # Mirror: ITD decreases as source moves behind the head
        theta_m = np.pi - theta
        tau = (HEAD_RADIUS / SOUND_SPEED) * (np.sin(theta_m) + theta_m)
    samples = int(round(tau * fs))
    sign = 1 if azimuth_deg >= 0 else -1
    return sign * samples          # positive -> left ear delayed


# -- ILD model ----------------------------------------------------------------

def _ild_db(azimuth_deg: float, freq_hz: float) -> float:
    """
    Frequency-dependent interaural level difference for ipsilateral ear.
    Based on a simplified Duda-Martens sphere model approximation.
    """
    theta = np.deg2rad(abs(azimuth_deg))
    # Magnitude of ILD grows with frequency and angle
    f0 = 1000.0
    ild_max_db = 20.0 * np.sin(theta)          # up to +/-20 dB at 90 deg
    freq_shape  = 1.0 - np.exp(-freq_hz / f0)   # 0 at DC, 1 at high freq
    return ild_max_db * freq_shape               # dB, positive = ipsilateral louder


# -- Integer delay line -------------------------------------------------------

class _DelayLine:
    """
    Circular-buffer delay line - fully vectorised for block processing.
    Valid as long as block_size <= delay (no within-block feedback needed).
    """

    def __init__(self, max_samples: int):
        self._buf  = np.zeros(max_samples + 2, dtype=np.float64)
        self._size = len(self._buf)
        self._pos  = 0

    def process(self, x: np.ndarray, delay: int) -> np.ndarray:
        n    = len(x)
        size = self._size
        ptr  = self._pos

        # Write n samples starting at ptr
        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % size
        self._buf[w_idx] = x

        # Read n samples that were written delay samples before ptr
        # (i.e. positions ptr-delay-n  to  ptr-delay-1)
        r_idx = np.arange(ptr - delay - n, ptr - delay, dtype=np.int64) % size
        out = self._buf[r_idx].copy()

        self._pos = int((ptr + n) % size)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._pos = 0


# -- Pinna notch filter -------------------------------------------------------

def _make_pinna_filter(fs: int, ch: int = 1) -> BiquadFilter:
    """
    Simplified pinna HRTF: a notch near 8 kHz helps externalize sound
    (prevents the "inside the head" sensation common with headphones).
    """
    return make_peaking(8000, -7.0, q=2.5, fs=fs, ch=ch)


# -- Per-speaker binaural renderer --------------------------------------------

class _SpeakerRenderer:
    """
    Render one mono virtual speaker to stereo (L_ear, R_ear).
    Applies ITD, frequency-dependent ILD, and pinna notch filter.
    """

    MAX_DELAY = 128   # samples (> max Woodworth ITD at 48 kHz)

    def __init__(self, azimuth_deg: float, fs: int):
        self._az  = azimuth_deg
        self._fs  = fs

        itd = _woodworth_itd_samples(azimuth_deg, fs)
        # itd > 0 -> left ear delayed; itd < 0 -> right ear delayed
        self._left_delay  = max(0,  itd)
        self._right_delay = max(0, -itd)

        self._dl_L = _DelayLine(self.MAX_DELAY)
        self._dl_R = _DelayLine(self.MAX_DELAY)

        # Broadband ILD approximation at 3 kHz (perceptually weighted)
        ild = _ild_db(azimuth_deg, 3000.0)
        sign = 1 if azimuth_deg >= 0 else -1
        self._gain_ipsi   = 10 ** ( sign * ild / 20.0)   # louder near ear
        self._gain_contra = 10 ** (-sign * ild / 20.0)   # quieter far ear

        # One pinna filter per ear (mono each)
        self._pinna_L = _make_pinna_filter(fs, ch=1)
        self._pinna_R = _make_pinna_filter(fs, ch=1)

    def process(self, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        mono : (N,) float32/64
        Returns left_ear, right_ear  each (N,)
        """
        m = mono.astype(np.float64)

        # Apply delay to contralateral channel
        left  = self._dl_L.process(m, self._left_delay)
        right = self._dl_R.process(m, self._right_delay)

        # ILD (broadband approximation)
        if self._az >= 0:          # sound from right
            right *= self._gain_ipsi
            left  *= self._gain_contra
        else:                      # sound from left
            left  *= self._gain_ipsi
            right *= self._gain_contra

        # Pinna notch (reshape to (N,1) for filter, then squeeze)
        left  = self._pinna_L.process(left[:, None])[:, 0]
        right = self._pinna_R.process(right[:, None])[:, 0]

        return left, right

    def reset(self):
        self._dl_L.reset()
        self._dl_R.reset()
        self._pinna_L.reset()
        self._pinna_R.reset()


# -- Stereo upmixer -----------------------------------------------------------

class _StereoUpmix:
    """
    Dolby Pro Logic II-inspired matrix decoder.
    Extracts: L, C, R, LS, RS from a stereo pair.

    Matrix equations
    ----------------
    C   =  (L + R) * 0.5  (sum)
    LS  =  L - C          (left difference)
    RS  =  R - C          (right difference)

    For the surrounds, a 90-deg all-pass phase shift is applied to the difference
    to reduce colouration when summed back.
    """

    def __init__(self, fs: int = 48000):
        # First-order all-pass for 90-deg phase shift on surrounds (~1 kHz crossover)
        # y[n] = -x[n] + x[n-1] + y[n-1]  (unit-circle all-pass)
        from scipy.signal import lfilter, lfilter_zi
        self._lfilter  = lfilter
        self._b_ap = np.array([-0.6, 1.0])   # first-order all-pass
        self._a_ap = np.array([ 1.0, -0.6])
        zi = lfilter_zi(self._b_ap, self._a_ap)
        self._zi_ls = zi.copy()
        self._zi_rs = zi.copy()

    def process(self, stereo: np.ndarray):
        """
        stereo : (N, 2) - columns [L, R]
        Returns dict: {L, C, R, LS, RS}  each (N,)
        """
        L = stereo[:, 0].astype(np.float64)
        R = stereo[:, 1].astype(np.float64)

        C  = (L + R) * 0.5
        diff_L = L - C
        diff_R = R - C

        # All-pass phase rotate surrounds to reduce comb filtering
        LS, self._zi_ls = self._lfilter(self._b_ap, self._a_ap, diff_L, zi=self._zi_ls)
        RS, self._zi_rs = self._lfilter(self._b_ap, self._a_ap, diff_R, zi=self._zi_rs)

        return {"L": L, "C": C, "R": R, "LS": LS, "RS": RS}

    def reset(self):
        from scipy.signal import lfilter_zi
        zi = lfilter_zi(self._b_ap, self._a_ap)
        self._zi_ls = zi.copy()
        self._zi_rs = zi.copy()


# -- Main virtualizer ---------------------------------------------------------

class VirtualSurroundProcessor:
    """
    Full virtual surround pipeline:

      1. Split stereo into three frequency bands
         * Sub-bass  (<120 Hz)  -> omnidirectional center pan (theater LFE)
         * Mid-band  (120-8 kHz)-> 5-channel upmix + HRTF binaural render
         * Air band  (>8 kHz)   -> extra stereo widening

      2. Binaural render each virtual speaker with ITD, ILD, pinna filter

      3. Sum all bands into final stereo output
    """

    CROSSOVER_LO = 120.0    # Hz - LFE / main split
    CROSSOVER_HI = 8000.0   # Hz - mid / air split

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        if preset is None:
            from config import THEATER_PRESET
            preset = THEATER_PRESET

        self._fs = fs
        self._p  = preset

        # -- Band-split filters ------------------------------------------------
        # Sub-bass: LP at 120 Hz
        self._lp_sub = make_lowpass(self.CROSSOVER_LO, q=0.707, fs=fs, ch=2)
        # Mid complement: HP at 120 Hz
        self._hp_mid = make_highpass(self.CROSSOVER_LO, q=0.707, fs=fs, ch=2)
        # Air split: HP at 8 kHz
        self._hp_air = make_highpass(self.CROSSOVER_HI, q=0.707, fs=fs, ch=2)
        # Mid-band: LP at 8 kHz (= mid_full minus air)
        self._lp_mid = make_lowpass(self.CROSSOVER_HI, q=0.707, fs=fs, ch=2)

        # -- Matrix upmixer ----------------------------------------------------
        self._upmix = _StereoUpmix(fs)

        # -- Per-speaker binaural renderers ------------------------------------
        self._renderers = {
            "L":  _SpeakerRenderer(preset["speaker_L_az"],  fs),
            "C":  _SpeakerRenderer(preset["speaker_C_az"],  fs),
            "R":  _SpeakerRenderer(preset["speaker_R_az"],  fs),
            "LS": _SpeakerRenderer(preset["speaker_LS_az"], fs),
            "RS": _SpeakerRenderer(preset["speaker_RS_az"], fs),
        }

        # Channel levels
        self._lvl = {
            "L":  1.0,
            "C":  preset["center_level"],
            "R":  1.0,
            "LS": preset["surround_level"],
            "RS": preset["surround_level"],
        }

        # Width gain for air band (user's idea: HF feels diffuse in a theater)
        self._air_width = preset["stereo_width"]

    def process(self, stereo: np.ndarray) -> np.ndarray:
        """
        stereo : (N, 2)  float32
        Returns (N, 2) float32 - binaural stereo
        """
        N = stereo.shape[0]

        # -- 1. Band split -----------------------------------------------------
        sub  = self._lp_sub.process(stereo)          # (N, 2) - below 120 Hz
        full = self._hp_mid.process(stereo)           # (N, 2) - above 120 Hz
        air  = self._hp_air.process(full)             # (N, 2) - above 8 kHz
        mid  = self._lp_mid.process(full)             # (N, 2) - 120-8000 Hz

        # -- 2. Sub-bass: omnidirectional (theater LFE feel) ------------------
        lfe_level  = self._p["lfe_level"]
        # Mono bass panned equally to both ears (no spatial cue at LF)
        sub_mono   = (sub[:, 0] + sub[:, 1]) * 0.5 * lfe_level
        out_L = sub_mono.copy()
        out_R = sub_mono.copy()

        # -- 3. Mid-band: upmix + binaural render ------------------------------
        channels = self._upmix.process(mid)
        for name, mono in channels.items():
            lvl = self._lvl[name]
            l, r = self._renderers[name].process(mono * lvl)
            out_L += l
            out_R += r

        # -- 4. Air band: widen (user's insight: HF is spatially diffuse) -----
        # M/S width expansion
        air_M  = (air[:, 0] + air[:, 1]) * 0.5
        air_S  = (air[:, 0] - air[:, 1]) * 0.5
        air_wL = air_M + self._air_width * air_S
        air_wR = air_M - self._air_width * air_S
        out_L += air_wL
        out_R += air_wR

        # -- 5. Normalise mix gain (5 channels summing) -----------------------
        mix_gain = 1.0 / 3.5
        out = np.stack([out_L * mix_gain, out_R * mix_gain], axis=1)
        return out.astype(stereo.dtype)

    def reset(self):
        for f in (self._lp_sub, self._hp_mid, self._hp_air, self._lp_mid):
            f.reset()
        self._upmix.reset()
        for r in self._renderers.values():
            r.reset()

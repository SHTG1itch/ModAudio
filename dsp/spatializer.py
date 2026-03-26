"""
Virtual surround spatialization - headphone and speaker modes.

HEADPHONE MODE  (BinauralSurroundProcessor)
-------------------------------------------
Frequency-band split + 5-channel upmix + Brown-Duda binaural HRTF rendering.

Architecture (hybrid of user's frequency-band idea + HRTF binaural rendering):

  Stereo in
      |
      +- [Sub-bass LPF < 120 Hz]  -> center / omnidirectional
      |                               (bass is non-directional in theaters)
      |
      +- [Mid-band 120-8000 Hz]   -> matrix-decoded 5-channel upmix:
      |                               L, C, R, LS, RS
      |                               Each channel -> Brown-Duda binaural render
      |
      +- [Air band > 8000 Hz]     -> M/S width expansion
                                     (HF feels spatially diffuse)

  Mix all bands -> stereo binaural output

SPEAKER MODE  (StereoWidenerProcessor)
---------------------------------------
For playback through actual stereo speakers. The room already provides the
head-related filtering, so HRTF would double-process and cause comb filtering.
Instead: stereo widening + Haas-effect depth + same reverb/EQ chain.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .filters import make_lowpass, make_highpass
from .hrtf    import BinauralSpeakerRenderer


# -- Stereo upmix (Dolby Pro Logic II-inspired) --------------------------------

class _StereoUpmix:
    """
    Matrix decoder: extract L, C, R, LS, RS from a stereo pair.

    C   = (L + R) * 0.5          (sum)
    LS  = phase-rotated(L - C)   (left surround)
    RS  = phase-rotated(R - C)   (right surround)
    """

    def __init__(self, fs=48000):
        from scipy.signal import lfilter, lfilter_zi
        self._lfilter = lfilter
        self._b_ap = np.array([-0.6, 1.0])
        self._a_ap = np.array([ 1.0, -0.6])
        zi = lfilter_zi(self._b_ap, self._a_ap)
        self._zi_ls = zi.copy()
        self._zi_rs = zi.copy()

    def process(self, stereo):
        L = stereo[:, 0].astype(np.float64)
        R = stereo[:, 1].astype(np.float64)
        C = (L + R) * 0.5
        LS, self._zi_ls = self._lfilter(self._b_ap, self._a_ap, L - C, zi=self._zi_ls)
        RS, self._zi_rs = self._lfilter(self._b_ap, self._a_ap, R - C, zi=self._zi_rs)
        return {"L": L, "C": C, "R": R, "LS": LS, "RS": RS}

    def reset(self):
        zi = lfilter_zi(self._b_ap, self._a_ap)
        self._zi_ls = zi.copy()
        self._zi_rs = zi.copy()


# -- Headphone mode: full binaural rendering -----------------------------------

class BinauralSurroundProcessor:
    """
    Full binaural virtual surround for headphones.
    Uses Brown-Duda HRTF for each of 5 virtual speakers.
    """

    CROSSOVER_LO = 120.0
    CROSSOVER_HI = 8000.0

    def __init__(self, fs=48000, preset=None):
        if preset is None:
            from config import THEATER_PRESET
            preset = THEATER_PRESET

        self._fs  = fs
        self._p   = preset

        self._lp_sub = make_lowpass(self.CROSSOVER_LO, q=0.707, fs=fs, ch=2)
        self._hp_mid = make_highpass(self.CROSSOVER_LO, q=0.707, fs=fs, ch=2)
        self._hp_air = make_highpass(self.CROSSOVER_HI, q=0.707, fs=fs, ch=2)
        self._lp_mid = make_lowpass(self.CROSSOVER_HI, q=0.707, fs=fs, ch=2)

        self._upmix = _StereoUpmix(fs)

        self._renderers = {
            "L":  BinauralSpeakerRenderer(preset["speaker_L_az"],  0.0, fs),
            "C":  BinauralSpeakerRenderer(preset["speaker_C_az"],  0.0, fs),
            "R":  BinauralSpeakerRenderer(preset["speaker_R_az"],  0.0, fs),
            "LS": BinauralSpeakerRenderer(preset["speaker_LS_az"], 0.0, fs),
            "RS": BinauralSpeakerRenderer(preset["speaker_RS_az"], 0.0, fs),
        }

        self._lvl = {
            "L":  1.0,
            "C":  preset["center_level"],
            "R":  1.0,
            "LS": preset["surround_level"],
            "RS": preset["surround_level"],
        }

        self._air_width = preset["stereo_width"]
        self._lfe_level = preset["lfe_level"]

    def process(self, stereo):
        sub  = self._lp_sub.process(stereo)
        full = self._hp_mid.process(stereo)
        air  = self._hp_air.process(full)
        mid  = self._lp_mid.process(full)

        # Sub-bass: mono LFE (bass is non-directional)
        sub_mono = (sub[:, 0] + sub[:, 1]) * 0.5 * self._lfe_level
        out_L    = sub_mono.copy()
        out_R    = sub_mono.copy()

        # Mid-band: 5-channel upmix + binaural render
        channels = self._upmix.process(mid)
        for name, mono in channels.items():
            l, r = self._renderers[name].process(mono * self._lvl[name])
            out_L += l
            out_R += r

        # Air band: M/S width expansion
        air_M  = (air[:, 0] + air[:, 1]) * 0.5
        air_S  = (air[:, 0] - air[:, 1]) * 0.5
        out_L += air_M + self._air_width * air_S
        out_R += air_M - self._air_width * air_S

        # Normalise (5 channels summing)
        mix_gain = 1.0 / 3.5
        out = np.stack([out_L * mix_gain, out_R * mix_gain], axis=1)
        return out.astype(stereo.dtype)

    def reset(self):
        for f in (self._lp_sub, self._hp_mid, self._hp_air, self._lp_mid):
            f.reset()
        self._upmix.reset()
        for r in self._renderers.values():
            r.reset()


# -- Speaker mode: stereo widening + Haas depth --------------------------------

class StereoWidenerProcessor:
    """
    Speaker-mode spatialization. No HRTF (room provides natural head-filtering).

    Techniques:
      1. M/S stereo width expansion  (wider soundstage between speakers)
      2. Haas-effect depth           (short delay on one channel = apparent depth)
      3. Sub-bass management         (mono bass below crossover)
      4. Air-band width boost        (HF sounds wider than speakers)
    """

    CROSSOVER_LO = 120.0
    CROSSOVER_HI = 8000.0

    def __init__(self, fs=48000, preset=None):
        if preset is None:
            from config import SPEAKERS_PRESET
            preset = SPEAKERS_PRESET

        self._fs  = fs
        self._p   = preset

        self._lp_sub = make_lowpass(self.CROSSOVER_LO, q=0.707, fs=fs, ch=2)
        self._hp_mid = make_highpass(self.CROSSOVER_LO, q=0.707, fs=fs, ch=2)
        self._hp_air = make_highpass(self.CROSSOVER_HI, q=0.707, fs=fs, ch=2)
        self._lp_mid = make_lowpass(self.CROSSOVER_HI, q=0.707, fs=fs, ch=2)

        self._width     = preset["stereo_width"]
        self._lfe_level = preset["lfe_level"]
        self._air_width = preset.get("stereo_width", 1.8)

        # Haas-effect delay line (mono, applied to mid-band right channel)
        haas_ms = float(preset.get("haas_delay_ms", 22.0))
        haas_n  = max(1, int(round(haas_ms * fs / 1000)))
        bsize   = haas_n + 1
        self._haas_buf   = np.zeros((bsize, 2), dtype=np.float64)
        self._haas_delay = haas_n
        self._haas_bsize = bsize
        self._haas_ptr   = 0

    def _haas_process(self, mid):
        """Apply Haas delay to the mid-band stereo signal."""
        n, buf, bsize, ptr = mid.shape[0], self._haas_buf, self._haas_bsize, self._haas_ptr
        x = mid.astype(np.float64)
        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % bsize
        buf[w_idx] = x
        # Delay the right channel to create depth
        d = self._haas_delay
        r_idx = np.arange(ptr - d - n, ptr - d, dtype=np.int64) % bsize
        delayed_R = buf[r_idx, 1]
        self._haas_ptr = int((ptr + n) % bsize)

        out = x.copy()
        out[:, 1] = (x[:, 1] + 0.4 * delayed_R) / 1.4   # blend delayed
        return out

    def process(self, stereo):
        sub  = self._lp_sub.process(stereo)
        full = self._hp_mid.process(stereo)
        air  = self._hp_air.process(full)
        mid  = self._lp_mid.process(full)

        # Sub-bass: mono (no directional information at LF for speakers)
        sub_mono = (sub[:, 0] + sub[:, 1]) * 0.5 * self._lfe_level
        sub_stereo = np.stack([sub_mono, sub_mono], axis=1)

        # Mid-band: M/S width + Haas depth
        M   = (mid[:, 0] + mid[:, 1]) * 0.5
        S   = (mid[:, 0] - mid[:, 1]) * 0.5
        mid_w = np.stack([M + self._width * S, M - self._width * S], axis=1)
        mid_w = self._haas_process(mid_w)

        # Air band: wider M/S expansion
        air_M = (air[:, 0] + air[:, 1]) * 0.5
        air_S = (air[:, 0] - air[:, 1]) * 0.5
        air_w = np.stack([air_M + self._air_width * air_S,
                          air_M - self._air_width * air_S], axis=1)

        out = (sub_stereo + mid_w + air_w) / 3.0
        return out.astype(stereo.dtype)

    def reset(self):
        for f in (self._lp_sub, self._hp_mid, self._hp_air, self._lp_mid):
            f.reset()
        self._haas_buf[:] = 0.0
        self._haas_ptr = 0


# -- Factory -------------------------------------------------------------------

def make_spatializer(preset):
    """Return the correct spatializer for the given preset mode."""
    mode = preset.get("mode", "headphones")
    if mode == "speakers":
        return StereoWidenerProcessor(fs=48000, preset=preset)
    return BinauralSurroundProcessor(fs=48000, preset=preset)

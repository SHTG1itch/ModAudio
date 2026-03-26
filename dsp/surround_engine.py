"""
Virtual Surround Sound Engine — 7.1 binaural and mono-compatible modes.

Two rendering paths:

VirtualSurroundBinaural  (mode = "surround")
────────────────────────────────────────────
Stereo input → 7-channel upmix → FullSphereHRTFRenderer per channel → binaural out.

Virtual 7.1 speaker layout:
  FL  −30°  0°     LB  −150°  −5°
  FR  +30°  0°     RB  +150°  −5°
  C     0°  0°
  LS  −110°  +5°   (slightly elevated: surrounds feel "around" not just sideways)
  RS  +110°  +5°

Each speaker uses FullSphereHRTFRenderer which provides:
  • Algazi ITD (elevation-aware)
  • Elevation-shifted pinna notch (6.5–13 kHz range)
  • Front/back spectral cue at 4.5 kHz
  • Head shadow on contralateral ear

VirtualSurroundMono  (mode = "surround_mono")
─────────────────────────────────────────────
Works with a SINGLE physical speaker. Strategy:

  1. Run full 7.1 binaural rendering (above).
  2. Sum L+R ears to mono.
     The HRTF spectral coloring — pinna notches, front/back cues — survives
     the mono sum and provides measurable directional hints (Blauert 1997).
  3. Add six directional early reflections from virtual room walls.
     Even in mono, the reflection delay pattern strongly implies a large,
     enveloping acoustic space (Haas / precedence effect).
     Walls absorb high frequencies → LP filter on each reflection tap.

Factory
───────
    from dsp.surround_engine import make_virtual_surround
    proc = make_virtual_surround(fs=48000, preset=preset_dict)
    out  = proc.process(stereo_block)   # (N,2) → (N,2)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .hrtf_full import (
    FullSphereHRTFRenderer,
    _lowpass_ba,
    _MonoFilter,
)
from .filters import make_lowpass, make_highpass


# ---------------------------------------------------------------------------
# 7-channel up-mix (Dolby Pro Logic II style, extended to 7.1)
# ---------------------------------------------------------------------------

class _Upmix71:
    """
    Stereo → 7-channel matrix decoder.

    Returns dict keys: FL, FR, C, LS, RS, LB, RB.

    Algorithm
    ---------
    C  = (L + R) / 2                     — correlated center content
    LS = AllPass90( L − 0.7·C )          — left-side decorrelated material
    RS = AllPass90( R − 0.7·C )          — right-side decorrelated material
    LB = AllPass30( LS · 0.8 )           — rear-left (second decorrelation)
    RB = AllPass30( RS · 0.8 )           — rear-right

    The two all-pass stages (90° and 30° corner frequencies) produce the
    phase quadrature that prevents rear channels from folding back into
    the front image while sounding temporally coherent.
    """

    def __init__(self, fs: int = 48000):
        # Stage-1 all-pass (surround extraction, ~90° rotation at mid-bass)
        b1 = np.array([-0.6, 1.0])
        a1 = np.array([ 1.0, -0.6])
        zi1 = lfilter_zi(b1, a1)
        self._b1, self._a1 = b1, a1
        self._zi_ls = zi1.copy()
        self._zi_rs = zi1.copy()

        # Stage-2 all-pass (rear channels, additional decorrelation)
        b2 = np.array([-0.3, 1.0])
        a2 = np.array([ 1.0, -0.3])
        zi2 = lfilter_zi(b2, a2)
        self._b2, self._a2 = b2, a2
        self._zi_lb = zi2.copy()
        self._zi_rb = zi2.copy()

    def process(self, stereo: np.ndarray) -> dict[str, np.ndarray]:
        L = stereo[:, 0].astype(np.float64)
        R = stereo[:, 1].astype(np.float64)
        C = (L + R) * 0.5

        ls_src = L - C * 0.7
        rs_src = R - C * 0.7

        LS, self._zi_ls = lfilter(self._b1, self._a1, ls_src, zi=self._zi_ls)
        RS, self._zi_rs = lfilter(self._b1, self._a1, rs_src, zi=self._zi_rs)
        LB, self._zi_lb = lfilter(self._b2, self._a2, LS * 0.8, zi=self._zi_lb)
        RB, self._zi_rb = lfilter(self._b2, self._a2, RS * 0.8, zi=self._zi_rb)

        return {"FL": L, "FR": R, "C": C,
                "LS": LS, "RS": RS,
                "LB": LB, "RB": RB}

    def reset(self):
        zi1 = lfilter_zi(self._b1, self._a1)
        zi2 = lfilter_zi(self._b2, self._a2)
        self._zi_ls = zi1.copy()
        self._zi_rs = zi1.copy()
        self._zi_lb = zi2.copy()
        self._zi_rb = zi2.copy()


# ---------------------------------------------------------------------------
# Default 7.1 speaker positions  (azimuth°, elevation°)
# ---------------------------------------------------------------------------

_DEFAULT_LAYOUT: dict[str, tuple[float, float]] = {
    "FL": (-30.0,   0.0),
    "FR": ( 30.0,   0.0),
    "C":  (  0.0,   0.0),
    "LS": (-110.0,  5.0),   # slight upward tilt — surrounds feel enveloping
    "RS": ( 110.0,  5.0),
    "LB": (-150.0, -5.0),   # rear channels slightly below ear level
    "RB": ( 150.0, -5.0),
}


# ---------------------------------------------------------------------------
# Binaural virtual surround (headphones / near-field speakers)
# ---------------------------------------------------------------------------

class VirtualSurroundBinaural:
    """
    7.1 virtual surround via full-sphere HRTF for headphones or speakers.

    Signal flow
    -----------
    Sub-bass (<120 Hz)  → mono LFE  (bass is non-directional in cinema)
    Mid (120–8000 Hz)   → 7-ch upmix → FullSphereHRTFRenderer per channel
    Air  (>8000 Hz)     → M/S width expansion (HF perceived as spatially diffuse)
    """

    LO = 120.0    # Hz
    HI = 8000.0   # Hz

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        p = preset or {}

        self._lp_sub = make_lowpass(self.LO,  q=0.707, fs=fs, ch=2)
        self._hp_mid = make_highpass(self.LO,  q=0.707, fs=fs, ch=2)
        self._lp_mid = make_lowpass(self.HI,  q=0.707, fs=fs, ch=2)
        self._hp_air = make_highpass(self.HI,  q=0.707, fs=fs, ch=2)

        self._upmix     = _Upmix71(fs)
        self._lfe_level = float(p.get("lfe_level",      0.85))
        self._air_width = float(p.get("stereo_width",   2.0))

        sl   = float(p.get("surround_level",  0.72))
        cl   = float(p.get("center_level",    0.88))
        rear = float(p.get("rear_level",      0.60))

        # Build renderer + level per virtual channel
        layout: dict[str, tuple[float, float]] = {
            "FL": (float(p.get("speaker_L_az",  -30.0)),   0.0),
            "FR": (float(p.get("speaker_R_az",   30.0)),   0.0),
            "C":  (float(p.get("speaker_C_az",    0.0)),   0.0),
            "LS": (float(p.get("speaker_LS_az", -110.0)),  5.0),
            "RS": (float(p.get("speaker_RS_az",  110.0)),  5.0),
            "LB": (-150.0, -5.0),
            "RB": ( 150.0, -5.0),
        }
        levels: dict[str, float] = {
            "FL": 1.0, "FR": 1.0, "C": cl,
            "LS": sl,  "RS": sl,
            "LB": rear, "RB": rear,
        }

        self._renderers: dict[str, FullSphereHRTFRenderer] = {}
        self._levels: dict[str, float] = {}
        for name, (az, el) in layout.items():
            self._renderers[name] = FullSphereHRTFRenderer(az, el, fs)
            self._levels[name]    = levels[name]

    def process(self, stereo: np.ndarray) -> np.ndarray:
        """(N, 2) float32 → (N, 2) float32"""
        sub  = self._lp_sub.process(stereo)
        full = self._hp_mid.process(stereo)
        mid  = self._lp_mid.process(full)
        air  = self._hp_air.process(full)

        # Sub-bass: mono (below 120 Hz is non-directional in cinema)
        sub_m = (sub[:, 0] + sub[:, 1]) * 0.5 * self._lfe_level
        out_L = sub_m.copy()
        out_R = sub_m.copy()

        # Mid-band: 7-ch upmix + per-channel HRTF rendering
        channels = self._upmix.process(mid)
        for name, mono in channels.items():
            l, r = self._renderers[name].process(mono * self._levels[name])
            out_L += l
            out_R += r

        # Air band: M/S width expansion
        M = (air[:, 0] + air[:, 1]) * 0.5
        S = (air[:, 0] - air[:, 1]) * 0.5
        out_L += M + self._air_width * S
        out_R += M - self._air_width * S

        # Normalise (7 channels + LFE + air)
        gain = 1.0 / 4.5
        out  = np.stack([out_L * gain, out_R * gain], axis=1)
        return out.astype(stereo.dtype)

    def reset(self):
        for f in (self._lp_sub, self._hp_mid, self._lp_mid, self._hp_air):
            f.reset()
        self._upmix.reset()
        for r in self._renderers.values():
            r.reset()


# ---------------------------------------------------------------------------
# Mono virtual surround (single physical speaker)
# ---------------------------------------------------------------------------

class VirtualSurroundMono:
    """
    Single-speaker virtual surround.

    Works because:
    a) HRTF spectral coloring (pinna notches, front/back cues, concha resonance)
       survives the binaural → mono sum and provides measurable directional
       spectral hints to the auditory system (Blauert 1997).
    b) Six simulated room reflections with individual delays and LP filtering
       strongly imply a large enveloping acoustic space even in mono.
       The brain uses the delay/spectral pattern to infer room geometry.

    Room reflection taps
    --------------------
    Each tap simulates a first-order reflection from a virtual surface:
      left wall  : ~11 ms   (arrives from the left)
      right wall : ~16 ms
      ceiling    : ~21 ms
      rear wall  : ~29 ms
      side corner: ~41 ms
      far corner : ~63 ms
    All taps pass through a low-pass filter (fc ≈ 3.2 kHz) modelling
    surface absorption of high frequencies.
    """

    # (delay_ms, gain)
    _ROOM_TAPS = [
        (11.0, 0.32),
        (16.0, 0.26),
        (21.0, 0.20),
        (29.0, 0.16),
        (41.0, 0.12),
        (63.0, 0.08),
    ]

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        self._binaural = VirtualSurroundBinaural(fs, preset)

        # Room reflection buffer
        max_d = int(max(t for t, _ in self._ROOM_TAPS) * fs / 1000) + 64
        bsz   = max_d + 2048
        self._rbuf = np.zeros(bsz, dtype=np.float64)
        self._rptr = 0
        self._rsz  = bsz

        self._delays = [int(round(t * fs / 1000)) for t, _ in self._ROOM_TAPS]
        self._gains  = [g for _, g in self._ROOM_TAPS]

        # Low-pass for reflections: walls absorb high frequencies
        b_lp, a_lp = _lowpass_ba(3200.0, 0.707, float(fs))
        self._rm_lpf = _MonoFilter(b_lp, a_lp)

    def process(self, stereo: np.ndarray) -> np.ndarray:
        """(N, 2) float32 → (N, 2) float32  [both channels identical: mono]"""
        # Full binaural processing preserves spatial spectral coloring
        binaural = self._binaural.process(stereo)

        # Collapse to mono — HRTF coloring provides directional hints
        mono = (binaural[:, 0] + binaural[:, 1]) * 0.5
        n64  = mono.astype(np.float64)

        # Write to circular room buffer
        buf, sz, ptr = self._rbuf, self._rsz, self._rptr
        w_idx = np.arange(ptr, ptr + len(n64), dtype=np.int64) % sz
        buf[w_idx] = n64

        # Sum directional reflections
        room = np.zeros(len(n64), dtype=np.float64)
        for delay, gain in zip(self._delays, self._gains):
            r_idx  = np.arange(ptr - delay - len(n64), ptr - delay,
                               dtype=np.int64) % sz
            room  += buf[r_idx] * gain

        # Wall absorption filter
        room = self._rm_lpf.process(room)
        self._rptr = int((ptr + len(n64)) % sz)

        # Blend: direct + room (0.45 keeps the room present but not dominant)
        out_mono = (n64 + room * 0.45).astype(stereo.dtype)
        return np.stack([out_mono, out_mono], axis=1)

    def reset(self):
        self._binaural.reset()
        self._rbuf[:] = 0.0
        self._rptr = 0
        self._rm_lpf.reset()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_virtual_surround(fs: int, preset: dict):
    """
    Return the appropriate virtual surround processor for preset['mode'].

    'surround'      → VirtualSurroundBinaural  (headphones / speakers)
    'surround_mono' → VirtualSurroundMono       (single speaker)
    """
    mode = preset.get("mode", "surround")
    if mode == "surround_mono":
        return VirtualSurroundMono(fs=fs, preset=preset)
    return VirtualSurroundBinaural(fs=fs, preset=preset)

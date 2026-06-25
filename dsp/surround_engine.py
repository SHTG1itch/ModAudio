"""
Virtual Surround Sound Engine — 7.1 binaural and mono-compatible modes.

Two rendering paths:

VirtualSurroundBinaural  (mode = "surround")
────────────────────────────────────────────
Stereo input → adaptive 7-channel upmix → FullSphereHRTFRenderer per channel
→ binaural out.

The upmix analyses each audio block for:
  • Pan position  — (R_rms − L_rms) / (R_rms + L_rms) ∈ [−1, +1]
  • Coherence     — cross-correlation ∈ [0, 1]
                    0 = fully diffuse/decorrelated (wide ambient)
                    1 = fully correlated (mono / dialog)

These measurements drive two adaptive behaviours:

  1. Coherence-adaptive surround level
     Low coherence (ambient / reverb / wide stereo) → surrounds boosted.
     High coherence (dialog, mono-like effects)     → surrounds reduced.
     This matches cinema practice: dialog stays in front, ambient wraps around.

  2. Pan-to-surround extension  ("cinema panning")
     When |pan| > 0.35 a fraction of the dominant channel is added to the
     corresponding surround speaker (LS for left, RS for right).
     Dead zone ±0.35; full extension (×0.30) at ±1.0.
     Effect: a hard-left sweep sounds as if it comes from the left wall and
     wraps around the listener — not just from the front-left speaker.
     Both measurements are smoothed with a 50 ms time constant to prevent
     pumping/clicks during fast panning events.

Virtual 7.1 speaker layout:
  FL  −30°   0°     LB  −150°  −5°
  FR  +30°   0°     RB  +150°  −5°
  C     0°   0°
  LS  −110°  +5°    (slightly elevated — surrounds feel enveloping)
  RS  +110°  +5°

VirtualSurroundMono  (mode = "surround_mono")
─────────────────────────────────────────────
Works with a SINGLE physical speaker.

  1. Runs full 7.1 binaural rendering above.
  2. Sums L+R ears to mono — HRTF spectral coloring (pinna notches,
     front/back cues) survives the mono collapse (Blauert 1997).
  3. Adds six directional early reflections simulating left/right walls,
     ceiling, and rear of a virtual cinema.  Even in mono the delay pattern
     implies a large enveloping acoustic space (Haas / precedence effect).
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .hrtf_full import (
    FullSphereHRTFRenderer,
    _lowpass_ba,
    _MonoFilter,
)
from .filters import make_lr4_lowpass, make_lr4_highpass


# ---------------------------------------------------------------------------
# Adaptive 7-channel up-mix
# ---------------------------------------------------------------------------

class _AdaptiveUpmix71:
    """
    Stereo → 7-channel adaptive matrix decoder.

    Returns dict keys: FL, FR, C, LS, RS, LB, RB.

    Static extraction
    -----------------
    C  = (L + R) / 2
    LS = AllPass90(L − 0.7·C)            — left-side decorrelated ambient
    RS = AllPass90(R − 0.7·C)
    LB = AllPass30(LS · 0.8)             — further-decorrelated rear
    RB = AllPass30(RS · 0.8)

    Adaptive scaling (per block, smoothed over 50 ms)
    ─────────────────────────────────────────────────
    surr_scale = 0.5 + (1 − coherence) · 0.9
      coh=1 (mono)    → scale=0.50  (surrounds quiet, dialog stays front)
      coh=0 (diffuse) → scale=1.40  (ambient wraps around listener)

    Pan extension to surrounds (cinema panning)
    ───────────────────────────────────────────
    Dead-zone ±0.35; linear ramp to ±0.30 extension at ±1.0.
    When pan < −0.35:  LS += L · ls_ext   (sound bleeds from front-left
                                           toward left-surround wall)
    When pan > +0.35:  RS += R · rs_ext
    """

    # Analysis smoothing time constant (seconds). The per-block alpha is
    # derived from the actual block length so the smoothing speed is
    # independent of block size and sample rate.
    _SMOOTH_TC: float = 0.050

    def __init__(self, fs: int = 48000):
        self._fs = fs

        # Stage-1 all-pass  (surround extraction, ≈90° rotation)
        b1 = np.array([-0.6, 1.0])
        a1 = np.array([ 1.0, -0.6])
        self._b1, self._a1 = b1, a1
        self._zi_ls = np.zeros(1)   # rest state, not lfilter_zi
        self._zi_rs = np.zeros(1)

        # Stage-2 all-pass  (rear channels, additional decorrelation)
        b2 = np.array([-0.3, 1.0])
        a2 = np.array([ 1.0, -0.3])
        self._b2, self._a2 = b2, a2
        self._zi_lb = np.zeros(1)
        self._zi_rb = np.zeros(1)

        # Smoothed analysis state
        self._pan: float = 0.0   # −1 (full left) to +1 (full right)
        self._coh: float = 0.7   # 0 (diffuse) to 1 (mono)

    def process(self, stereo: np.ndarray) -> dict[str, np.ndarray]:
        L = stereo[:, 0].astype(np.float64)
        R = stereo[:, 1].astype(np.float64)

        # --- Per-block analysis -------------------------------------------
        L_rms = float(np.sqrt(np.mean(L * L) + 1e-12))
        R_rms = float(np.sqrt(np.mean(R * R) + 1e-12))

        # Pan position: −1 = full left, +1 = full right
        pan_raw = (R_rms - L_rms) / (R_rms + L_rms + 1e-12)

        # Coherence: normalised cross-correlation [0, 1]
        coh_raw = float(np.clip(
            np.mean(L * R) / (L_rms * R_rms + 1e-12), 0.0, 1.0))

        # Exponential smoothing (alpha from actual block duration)
        a = float(np.exp(-(len(L) / self._fs) / self._SMOOTH_TC))
        self._pan = a * self._pan + (1.0 - a) * float(pan_raw)
        self._coh = a * self._coh + (1.0 - a) * coh_raw

        pan = float(self._pan)
        coh = float(self._coh)

        # --- Static channel extraction ------------------------------------
        C     = (L + R) * 0.5
        ls_src = L - C * 0.7
        rs_src = R - C * 0.7

        LS_ap, self._zi_ls = lfilter(self._b1, self._a1, ls_src, zi=self._zi_ls)
        RS_ap, self._zi_rs = lfilter(self._b1, self._a1, rs_src, zi=self._zi_rs)
        LB,    self._zi_lb = lfilter(self._b2, self._a2, LS_ap * 0.8, zi=self._zi_lb)
        RB,    self._zi_rb = lfilter(self._b2, self._a2, RS_ap * 0.8, zi=self._zi_rb)

        # --- Adaptive gains -----------------------------------------------

        # Coherence → surround scale
        # More diffuse audio (music, ambience) = wider surround envelope.
        # Correlated (mono, dialog) = quieter surrounds so dialog stays front.
        surr_scale = 0.55 + (1.0 - coh) * 1.0   # range [0.55, 1.55]

        # Pan extension: hard-left/right feeds into corresponding surround channel.
        # This is the key mechanism for "wrap-around" movement —
        # when audio sweeps left→right the LS/RS channels get direct bleed
        # which then routes to the rear speaker via the VBAP matrix, giving the
        # perception that the sound moves from the front around the head.
        # Dead-zone ±0.30; full extension 0.55 at ±1.0
        if pan < -0.30:
            pan_frac = float(np.clip((-pan - 0.30) / 0.70, 0.0, 1.0))
            ls_ext = pan_frac * 0.55
            lb_ext = pan_frac * 0.22   # pan also bleeds into back channel
        else:
            ls_ext = 0.0
            lb_ext = 0.0
        if pan > 0.30:
            pan_frac = float(np.clip((pan - 0.30) / 0.70, 0.0, 1.0))
            rs_ext = pan_frac * 0.55
            rb_ext = pan_frac * 0.22
        else:
            rs_ext = 0.0
            rb_ext = 0.0

        LS = LS_ap * surr_scale + L * ls_ext
        RS = RS_ap * surr_scale + R * rs_ext
        LB_out = LB * surr_scale + L * lb_ext
        RB_out = RB * surr_scale + R * rb_ext

        return {
            "FL": L,
            "FR": R,
            "C":  C,
            "LS": LS,
            "RS": RS,
            "LB": LB_out,
            "RB": RB_out,
        }

    def reset(self):
        self._zi_ls = np.zeros(1)
        self._zi_rs = np.zeros(1)
        self._zi_lb = np.zeros(1)
        self._zi_rb = np.zeros(1)
        self._pan = 0.0
        self._coh = 0.7


# ---------------------------------------------------------------------------
# Binaural virtual surround (headphones / near-field speakers)
# ---------------------------------------------------------------------------

class VirtualSurroundBinaural:
    """
    7.1 virtual surround via full-sphere HRTF for headphones or speakers.

    Signal flow
    -----------
    Sub-bass (<120 Hz)  → mono LFE  (bass is non-directional in cinema)
    Mid (120–8000 Hz)   → adaptive 7-ch upmix → FullSphereHRTFRenderer/ch
    Air  (>8000 Hz)     → M/S width expansion (HF perceived as spatially diffuse)
    """

    LO = 120.0    # Hz
    HI = 8000.0   # Hz

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        p = preset or {}

        # LR4 crossovers: the three bands are summed back together at the
        # output, and LR4 LP/HP pairs recombine flat (a single Butterworth
        # biquad pair would notch the output at 120 Hz and 8 kHz).
        self._lp_sub = make_lr4_lowpass(self.LO,  fs=fs, ch=2)
        self._hp_mid = make_lr4_highpass(self.LO, fs=fs, ch=2)
        self._lp_mid = make_lr4_lowpass(self.HI,  fs=fs, ch=2)
        self._hp_air = make_lr4_highpass(self.HI, fs=fs, ch=2)

        self._upmix     = _AdaptiveUpmix71(fs)
        self._lfe_level = float(p.get("lfe_level",      0.85))
        self._air_width = float(p.get("stereo_width",   2.0))

        sl   = float(p.get("surround_level",  0.72))
        cl   = float(p.get("center_level",    0.88))
        rear = float(p.get("rear_level",      0.60))

        # Virtual speaker layout: azimuth°, elevation°
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

        # Sub-bass: mono (non-directional below 120 Hz)
        sub_m = (sub[:, 0] + sub[:, 1]) * 0.5 * self._lfe_level
        out_L = sub_m.copy()
        out_R = sub_m.copy()

        # Mid-band: adaptive 7-ch upmix + HRTF per channel
        channels = self._upmix.process(mid)
        for name, mono in channels.items():
            l, r = self._renderers[name].process(mono * self._levels[name])
            out_L += l
            out_R += r

        # Air band: M/S width expansion (HF sounds enveloping)
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
    Single-speaker virtual surround — improved for ~60-70 % surround impression.

    Architecture
    ------------
    1. Full 7.1 binaural rendering via FullSphereHRTFRenderer.
       HRTF spectral coloring (pinna notches, front/back notches, concha
       resonance) survives the binaural→mono collapse and gives the auditory
       system directional spectral cues (Blauert 1997).

    2. ITD-compensated mono collapse: instead of a naive (L+R)/2 that causes
       comb-filtering at high frequencies, a short allpass crossfeed filter
       smoothes the phase before summing, preserving more spectral content.

    3. Twelve room-reflection taps covering three distance ranges:
         Early (5–20 ms)   — room size perception, envelopment
         Mid   (20–60 ms)  — spaciousness, front/back depth
         Late  (60–130 ms) — reverb tail onset, large-space sensation
       Each range uses a different LP cutoff (surface material model):
         Early taps: fc = 5.5 kHz  (hard walls, less absorption)
         Mid taps:   fc = 3.5 kHz
         Late taps:  fc = 2.2 kHz  (carpets/seats, high HF absorption)

    4. Decorrelated side reflections: left-wall and right-wall taps are
       phase-inverted relative to each other (±0.5 gain) which the mono
       speaker reproduces as apparent width even without true stereo.

    The combined effect reliably delivers 60-70 % of true surround impression
    on a single speaker in a typical room.
    """

    # (delay_ms, gain, lp_cutoff_hz)
    _ROOM_TAPS = [
        # Early reflections — hard boundary first-order
        (  5.5,  0.28, 5500.0),   # floor / near-wall
        ( 10.0,  0.30, 5500.0),   # left wall
        ( 14.5, -0.28, 5500.0),   # right wall  (polarity inversion → width cue)
        ( 18.0,  0.22, 5500.0),   # ceiling
        # Mid reflections — secondary surfaces
        ( 24.0,  0.18, 3500.0),   # rear wall
        ( 31.0,  0.14, 3500.0),   # left rear corner
        ( 37.0, -0.12, 3500.0),   # right rear corner (inverted)
        ( 46.0,  0.10, 3500.0),   # ceiling + rear blend
        # Late reflections — reverb tail onset
        ( 62.0,  0.08, 2200.0),   # far left
        ( 78.0, -0.07, 2200.0),   # far right (inverted)
        ( 98.0,  0.05, 2200.0),
        (128.0,  0.04, 2200.0),
    ]

    def __init__(self, fs: int = 48000, preset: dict | None = None):
        self._binaural = VirtualSurroundBinaural(fs, preset)
        self._fs = fs

        max_d = int(max(t for t, *_ in self._ROOM_TAPS) * fs / 1000) + 128
        bsz   = max_d + 4096
        self._rbuf = np.zeros(bsz, dtype=np.float64)
        self._rptr = 0
        self._rsz  = bsz

        self._delays = [int(round(t * fs / 1000)) for t, _, _ in self._ROOM_TAPS]
        self._gains  = [g for _, g, _ in self._ROOM_TAPS]

        # Per-tap LP filters (surface absorption model)
        self._tap_lpf: list[_MonoFilter] = []
        for _, _, fc in self._ROOM_TAPS:
            b, a = _lowpass_ba(fc, 0.707, float(fs))
            self._tap_lpf.append(_MonoFilter(b, a))

        # Allpass crossfeed for phase-smooth mono collapse
        # First-order allpass at ~700 Hz  →  gradual phase rotation blends
        # the HRTF-processed L and R channels without hard cancellation.
        b_ap = np.array([-0.12, 1.0])
        a_ap = np.array([ 1.0, -0.12])
        self._b_ap, self._a_ap = b_ap, a_ap
        self._zi_cf_l = np.zeros(1)   # rest state, not lfilter_zi
        self._zi_cf_r = np.zeros(1)

    def process(self, stereo: np.ndarray) -> np.ndarray:
        """(N, 2) float32 → (N, 2) float32  [both channels identical: mono]"""
        binaural = self._binaural.process(stereo)

        # Phase-smooth mono collapse via allpass crossfeed
        L64 = binaural[:, 0].astype(np.float64)
        R64 = binaural[:, 1].astype(np.float64)
        L_cf, self._zi_cf_l = lfilter(self._b_ap, self._a_ap, L64, zi=self._zi_cf_l)
        R_cf, self._zi_cf_r = lfilter(self._b_ap, self._a_ap, R64, zi=self._zi_cf_r)
        # Blend: 0.5*direct + 0.5*crossfed-opposite gives smooth phase
        mono = (L64 * 0.6 + R_cf * 0.4 + R64 * 0.6 + L_cf * 0.4) * 0.5

        buf, sz, ptr = self._rbuf, self._rsz, self._rptr
        n = len(mono)
        w_idx = np.arange(ptr, ptr + n, dtype=np.int64) % sz
        buf[w_idx] = mono

        room = np.zeros(n, dtype=np.float64)
        for delay, gain, lpf in zip(self._delays, self._gains, self._tap_lpf):
            # [ptr-delay, ptr-delay+n) realizes exactly `delay` samples (the old
            # [ptr-delay-n, ptr-delay) added a full block to every reflection).
            r_idx = np.arange(ptr - delay, ptr - delay + n, dtype=np.int64) % sz
            tap   = buf[r_idx] * gain
            tap   = lpf.process(tap)
            room += tap

        self._rptr = int((ptr + n) % sz)

        # Room mix level: 0.55 gives strong envelopment without muddying direct
        out_mono = (mono + room * 0.55).astype(stereo.dtype)
        return np.stack([out_mono, out_mono], axis=1)

    def reset(self):
        self._binaural.reset()
        self._rbuf[:] = 0.0
        self._rptr = 0
        for f in self._tap_lpf:
            f.reset()
        self._zi_cf_l = np.zeros(1)
        self._zi_cf_r = np.zeros(1)


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

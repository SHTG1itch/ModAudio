"""
Multi-speaker theater chain — two physical speaker buses.

Unlike the regular TheaterChain (which renders to binaural via HRTF),
this chain routes the 7-channel upmix to two separate stereo output buses
so each bus can be played through a different physical speaker:

  Front bus  — FL + FR + Center blend  (speaker facing the listener)
  Rear  bus  — LS + RS + LB/RB blend   (speaker placed behind the listener)

VBAP Routing
------------
Each of the 7 virtual channels (FL -30°, FR +30°, C 0°, LS -110°, RS +110°,
LB -150°, RB +150°) is routed to the 4 physical bus outputs (front_L, front_R,
rear_L, rear_R) using constant-power Vector-Base Amplitude Panning on a circle.

Physical driver positions are:
  front_L : -30°
  front_R : +30°
  rear_L  : -rear_az_deg  (left-behind direction)
  rear_R  : +rear_az_deg  (right-behind direction)

The stream-level swap_rear_lr flag handles whether the rear speaker faces the
listener (facing → swap) or away (away → no swap).

Behind Spectral Coloring
------------------------
A mild psychoacoustic filter is applied to the rear bus:
  +1.5 dB low shelf  @ 280 Hz  — bass fullness of a rear-wall boundary
  −3.0 dB high shelf @ 4 kHz   — pinna / HF shadowing from behind
This helps the auditory system localise the rear speaker as coming from
behind even when it is physically close to the listener.

Signal flow
-----------
input (N,2) float32
    ↓  HarmonicBassEnhancer  (psychoacoustic sub-bass synthesis)
    ↓  AirBandExciter         (HF sparkle)
    ↓  CinemaEqualizer        (X-curve)
    ↓  TheaterReverb          (pre-delay + early reflections + FDN tail)
    ↓  MultibandCompressor    (4-band theater dynamics)
    ↓  TransientEnhancer      (punch)
    ↓  × master_gain
    ↓  sub/mid crossover split
    ↓  _AdaptiveUpmix71       (FL, FR, C, LS, RS, LB, RB + coherence steering)
    ↓  VBAP routing matrix  → 4 bus outputs
    ↓  rear bus: behind-spectral coloring filter
    ↓  PeakLimiter per bus
"""

from __future__ import annotations
import numpy as np

from .equalizer        import CinemaEqualizer
from .reverb           import TheaterReverb
from .dynamics         import MultibandCompressor, TransientEnhancer, PeakLimiter
from .enhancer         import HarmonicBassEnhancer, AirBandExciter
from .filters          import (make_lowpass, make_highpass,
                               make_lowshelf, make_highshelf, FilterChain)
from .surround_engine  import _AdaptiveUpmix71


# ---------------------------------------------------------------------------
# VBAP helpers
# ---------------------------------------------------------------------------

def _vbap_circle(src_az: float, spk_az_list: list) -> np.ndarray:
    """
    Constant-power VBAP on a circle.

    src_az      : source azimuth in degrees (0=front, +90=right, ±180=behind)
    spk_az_list : list of N speaker azimuths in degrees
    Returns     : gain array length N (sum-of-squares = 1)
    """
    n = len(spk_az_list)
    src      = float(src_az) % 360.0
    spk_norm = [float(a) % 360.0 for a in spk_az_list]

    # Sort counterclockwise (increasing angle)
    order = sorted(range(n), key=lambda i: spk_norm[i])
    spk_s = [spk_norm[order[i]] for i in range(n)]

    gains = np.zeros(n, dtype=np.float32)

    for i in range(n):
        a1 = spk_s[i]
        j  = (i + 1) % n
        a2 = spk_s[j]

        # Ensure a2 > a1 going counterclockwise
        if a2 <= a1:
            a2 += 360.0

        # Bring source into this sector's range
        src_adj = src
        if src_adj < a1:
            src_adj += 360.0

        if a1 <= src_adj <= a2:
            span = a2 - a1
            frac = (src_adj - a1) / span if span > 1e-6 else 0.5
            g1 = float(np.cos(frac * np.pi * 0.5))
            g2 = float(np.sin(frac * np.pi * 0.5))
            gains[order[i]] = g1
            gains[order[j]] = g2
            return gains

    # Fallback: nearest speaker
    diffs = [min(abs(src - a), 360.0 - abs(src - a)) for a in spk_norm]
    gains[int(np.argmin(diffs))] = 1.0
    return gains


def _build_routing_matrix(rear_az_deg: float) -> np.ndarray:
    """
    Build 7×4 VBAP routing matrix.

    Rows    : FL, FR, C, LS, RS, LB, RB  (virtual 7.1 channels)
    Columns : front_L, front_R, rear_L, rear_R  (physical bus channels)

    Physical driver azimuths:
      front_L : -30°
      front_R : +30°
      rear_L  : -rear_az_deg   (left-side rear driver)
      rear_R  : +rear_az_deg   (right-side rear driver)

    Note: rear_L/R are semantic names.  Whether rear_L physically points
    left or right depends on speaker orientation (handled by stream-level
    swap_rear_lr, not this matrix).
    """
    phys_az = [-30.0, +30.0, -rear_az_deg, +rear_az_deg]

    # Virtual 7.1 channel azimuths (fixed by cinema standard)
    virt_az = [
        -30.0,   # FL
        +30.0,   # FR
          0.0,   # C
       -110.0,   # LS  (left surround)
       +110.0,   # RS  (right surround)
       -150.0,   # LB  (left back)
       +150.0,   # RB  (right back)
    ]

    M = np.zeros((7, 4), dtype=np.float32)
    for i, az in enumerate(virt_az):
        M[i] = _vbap_circle(az, phys_az)
    return M


# ---------------------------------------------------------------------------
# MultiSpeakerChain
# ---------------------------------------------------------------------------

class MultiSpeakerChain:
    """
    Theater DSP chain for 2-physical-speaker setups.

    Parameters
    ----------
    fs           : sample rate (Hz)
    preset       : theater preset dict (from PRESETS in app.py)
    bass_priority: "equal" | "front" | "rear"
    rear_az_deg  : azimuth of rear speaker drivers (90–170°, default 150°).
                   150° = directly behind.  90° = side-mounted.
                   Smaller values move the rear image forward toward the sides.
    """

    _LO = 120.0   # sub-bass crossover (Hz)

    def __init__(self, fs: int = 48000, preset: dict | None = None,
                 bass_priority: str = "equal",
                 rear_az_deg: float = 150.0,
                 rear_direct_blend: float = 0.0):
        if preset is None:
            from config import HEADPHONES_PRESET
            preset = dict(HEADPHONES_PRESET)

        self._fs          = fs
        self._master_gain = 10 ** (float(preset.get("output_gain_db", 0.0)) / 20.0)

        # Rear speaker azimuth: how far behind (90–170°)
        self._rear_az_deg = float(np.clip(rear_az_deg, 60.0, 170.0))

        # Keep for compat but no longer used in routing (VBAP handles it)
        self._rear_direct_blend = float(rear_direct_blend)

        # bass_priority: "equal" | "front" | "rear"
        self._bass_priority = bass_priority

        # -- Enhancement & EQ -------------------------------------------------
        self._bass_enh = HarmonicBassEnhancer(
            cutoff=120.0,
            drive=float(preset.get("bass_harm_drive",   2.8)),
            level=float(preset.get("bass_harm_level",   0.50)),
            fs=fs,
        )
        self._air_exc = AirBandExciter(
            cutoff=8000.0,
            level=float(preset.get("air_exciter_level", 0.18)),
            fs=fs,
        )
        self._eq = CinemaEqualizer(fs=fs, num_channels=2, preset=preset)

        # -- Room acoustics ---------------------------------------------------
        self._reverb = TheaterReverb(fs=fs, preset=preset)

        # -- Dynamics ---------------------------------------------------------
        self._comp  = MultibandCompressor(
            fs=fs, drive=float(preset.get("mb_compress_drive", 1.6)))
        self._trans = TransientEnhancer(
            fs=fs, amount=float(preset.get("transient_amount", 0.55)))

        # -- Per-bus limiters -------------------------------------------------
        thr = float(preset.get("limiter_threshold",  0.93))
        rel = float(preset.get("limiter_release_ms", 80.0))
        self._lim_front = PeakLimiter(threshold=thr, release_ms=rel, fs=fs)
        self._lim_rear  = PeakLimiter(threshold=thr, release_ms=rel, fs=fs)

        # -- Sub-bass crossover -----------------------------------------------
        self._lp_sub = make_lowpass(self._LO,  q=0.707, fs=fs, ch=2)
        self._hp_mid = make_highpass(self._LO,  q=0.707, fs=fs, ch=2)

        # -- Adaptive 7-channel upmix -----------------------------------------
        self._upmix = _AdaptiveUpmix71(fs)

        # -- Channel levels ---------------------------------------------------
        self._lfe_level  = float(preset.get("lfe_level",      0.85))
        self._surr_level = float(preset.get("surround_level", 0.72))
        self._rear_level = float(preset.get("rear_level",     0.60))
        self._cl         = float(preset.get("center_level",   0.88))

        # -- VBAP routing matrix (7 virtual channels → 4 bus outputs) ---------
        self._routing_matrix = _build_routing_matrix(self._rear_az_deg)

        # -- "Behind" spectral coloring for rear bus --------------------------
        # Psychoacoustic cues that help localise the rear speaker as coming
        # from behind:
        #   • Low shelf +1.5 dB @ 280 Hz: room boundary bass buildup from rear
        #   • High shelf −3 dB  @ 4 kHz:  pinna HF shadow / distance rolloff
        self._rear_color = FilterChain([
            make_lowshelf( 280.0, +1.5, q=0.707, fs=fs, ch=2),
            make_highshelf(4000.0, -3.0, q=0.707, fs=fs, ch=2),
        ])

        # Normalization divisors — calibrated from steady-state RMS testing.
        # fn=1.5 rn=1.2 gives gain_vs_in ≈ -3.3 dB with +4.5 dB output_gain_db,
        # and rear/front ≈ -7.8 dB (rear slightly quieter, as expected for a
        # diffuse surround field vs direct front sound).
        self._front_norm = 1.0 / 1.5
        self._rear_norm  = 1.0 / 1.2

    # -------------------------------------------------------------------------

    def process(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        block : (N, 2) float32  stereo PCM

        Returns
        -------
        front_out : (N, 2) float32  — front speaker bus
        rear_out  : (N, 2) float32  — rear speaker bus
        """
        if block.ndim == 1:
            block = np.stack([block, block], axis=1)
        x = block.astype(np.float32, copy=False)

        # Full-signal theater processing
        x = self._bass_enh.process(x)
        x = self._air_exc.process(x)
        x = self._eq.process(x)
        x = self._reverb.process(x)
        x = self._comp.process(x)
        x = self._trans.process(x)
        x = x * self._master_gain

        # Sub-bass: mono, shared by both buses
        sub      = self._lp_sub.process(x)
        mid      = self._hp_mid.process(x)
        sub_mono = ((sub[:, 0] + sub[:, 1]) * 0.5 * self._lfe_level
                    ).astype(np.float64)

        # 7-channel adaptive upmix on mid/high band
        ch = self._upmix.process(mid)

        # Stack channels into a matrix (N, 7) for vectorised routing
        # Apply per-channel levels
        sig = np.stack([
            ch["FL"].astype(np.float64),
            ch["FR"].astype(np.float64),
            ch["C"].astype(np.float64)  * self._cl,
            ch["LS"].astype(np.float64) * self._surr_level,
            ch["RS"].astype(np.float64) * self._surr_level,
            ch["LB"].astype(np.float64) * self._rear_level,
            ch["RB"].astype(np.float64) * self._rear_level,
        ], axis=1)   # (N, 7)

        # VBAP routing: (N,7) @ (7,4) → (N,4) = [front_L, front_R, rear_L, rear_R]
        M    = self._routing_matrix.astype(np.float64)
        buses = sig @ M   # (N, 4)

        # Sub-bass distribution by bass_priority
        bp = self._bass_priority
        if bp == "front":
            fs_l, fs_r = sub_mono * 1.15, sub_mono * 1.15
            rs_l, rs_r = sub_mono * 0.06, sub_mono * 0.06
        elif bp == "rear":
            fs_l, fs_r = sub_mono * 0.06, sub_mono * 0.06
            rs_l, rs_r = sub_mono * 1.15, sub_mono * 1.15
        else:  # "equal"
            fs_l, fs_r = sub_mono, sub_mono
            rs_l, rs_r = sub_mono * 0.40, sub_mono * 0.40

        front_L = (buses[:, 0] + fs_l) * self._front_norm
        front_R = (buses[:, 1] + fs_r) * self._front_norm
        rear_L  = (buses[:, 2] + rs_l) * self._rear_norm
        rear_R  = (buses[:, 3] + rs_r) * self._rear_norm

        front = np.stack([front_L, front_R], axis=1).astype(np.float32)
        rear  = np.stack([rear_L,  rear_R],  axis=1).astype(np.float32)

        # Apply "behind" spectral coloring to rear bus
        rear = self._rear_color.process(rear)

        # Per-bus limiting
        front = self._lim_front.process(front)
        rear  = self._lim_rear.process(rear)

        return front, rear

    # -------------------------------------------------------------------------

    def set_bass_priority(self, priority: str):
        """Update bass priority at runtime ('equal', 'front', 'rear')."""
        self._bass_priority = priority

    def update_rear_az(self, rear_az_deg: float):
        """Update rear speaker azimuth and rebuild routing matrix."""
        self._rear_az_deg    = float(np.clip(rear_az_deg, 60.0, 170.0))
        self._routing_matrix = _build_routing_matrix(self._rear_az_deg)

    def reset(self):
        for stage in (self._bass_enh, self._air_exc, self._eq, self._reverb,
                      self._comp, self._trans, self._lim_front, self._lim_rear,
                      self._lp_sub, self._hp_mid, self._rear_color):
            stage.reset()
        self._upmix.reset()

    @property
    def fs(self) -> int:
        return self._fs

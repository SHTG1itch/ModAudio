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
import math
import numpy as np

from .equalizer        import CinemaEqualizer
from .reverb           import TheaterReverb
from .dynamics         import MultibandCompressor, TransientEnhancer, PeakLimiter
from .enhancer         import HarmonicBassEnhancer, AirBandExciter
from .filters          import (make_lowpass, make_highpass,
                               make_lowshelf, make_highshelf, FilterChain)
from .surround_engine  import _AdaptiveUpmix71


# ---------------------------------------------------------------------------
# 3-D vector helpers  (mirror room_canvas_3d conventions)
# ---------------------------------------------------------------------------

def _norm3(v: tuple) -> tuple:
    x, y, z = v
    m = math.sqrt(x*x + y*y + z*z)
    return (x/m, y/m, z/m) if m > 1e-10 else (0., 0., 1.)

def _cross3(a: tuple, b: tuple) -> tuple:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _az_el_to_dir(az_deg: float, el_deg: float) -> tuple:
    """(azimuth°, elevation°) → unit direction vector.  0°=front(−Z), 90°=right(+X)."""
    az_r = math.radians(az_deg)
    el_r = math.radians(el_deg)
    ce   = math.cos(el_r)
    return (math.sin(az_r)*ce, math.sin(el_r), -math.cos(az_r)*ce)

def _dir_to_az(dx: float, dz: float) -> float:
    """World (dx, dz) → azimuth degrees, -180..180."""
    return math.degrees(math.atan2(dx, -dz))


# ---------------------------------------------------------------------------
# VBAP helpers
# ---------------------------------------------------------------------------

def _vbap_sphere(src_az: float, src_el: float, spk_az_el_list: list) -> np.ndarray:
    """
    Constant-power panning on a sphere using 3D great-circle angular distances.

    Finds the two nearest speakers to the source by angular distance on the
    unit sphere and interpolates with constant power between them.  Reduces to
    the same result as a horizontal circle VBAP when all elevations are zero,
    but correctly handles height speakers (e.g. Dolby Atmos overhead).

    src_az, src_el   : source azimuth/elevation in degrees
    spk_az_el_list   : list of (az_deg, el_deg) tuples — one per speaker
    Returns          : gain array length N, sum-of-squares ≈ 1
    """
    n = len(spk_az_el_list)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    src_v = _az_el_to_dir(src_az, src_el)

    def _ang(az, el):
        v = _az_el_to_dir(az, el)
        dot = max(-1.0, min(1.0, src_v[0]*v[0] + src_v[1]*v[1] + src_v[2]*v[2]))
        return math.acos(dot)   # radians, 0 = exactly at speaker

    dists = [_ang(az, el) for az, el in spk_az_el_list]

    # Nearest pair
    order = sorted(range(n), key=lambda i: dists[i])
    i0, i1 = order[0], order[1]
    d0, d1 = dists[i0], dists[i1]
    span = d0 + d1

    gains = np.zeros(n, dtype=np.float32)
    if span < 1e-9:
        gains[i0] = 1.0
        return gains

    frac = d0 / span          # 0 = at i0, 1 = at i1
    gains[i0] = float(math.cos(frac * math.pi * 0.5))
    gains[i1] = float(math.sin(frac * math.pi * 0.5))
    return gains


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


def _speaker_driver_az_el(
    az_deg: float, el_deg: float,
    face_az_deg: float, face_el_deg: float,
    half_width_deg: float = 15.0,
) -> tuple:
    """
    Compute L and R virtual-driver positions for a physical speaker.

    The "right" vector of the speaker (cross(world_up, facing)) determines
    which physical direction the R driver is in, and vice-versa for L.
    This mirrors the convention used by the 3D room canvas visualiser.

    Parameters
    ----------
    az_deg, el_deg          : speaker azimuth/elevation from listener (degrees)
    face_az_deg, face_el_deg: speaker facing direction (degrees)
    half_width_deg          : angular half-separation of L and R drivers (°)

    Returns
    -------
    (L_az, L_el, R_az, R_el) in degrees
    """
    # Facing unit vector  (0° = front −Z, 90° = right +X)
    fv = _az_el_to_dir(face_az_deg, face_el_deg)

    # Gimbal-safe world up
    world_up = (0., 1., 0.)
    if abs(fv[1]) > 0.9:
        world_up = (0., 0., 1.)

    # Speaker's physical right vector: cross(world_up, forward)
    right = _norm3(_cross3(world_up, fv))

    # Speaker's centre direction from listener
    centre = _az_el_to_dir(az_deg, el_deg)

    hw_r   = math.radians(half_width_deg)
    cos_hw = math.cos(hw_r)
    sin_hw = math.sin(hw_r)

    # R driver = centre rotated +half_width toward right
    r_dir = _norm3((
        centre[0]*cos_hw + right[0]*sin_hw,
        centre[1]*cos_hw + right[1]*sin_hw,
        centre[2]*cos_hw + right[2]*sin_hw,
    ))
    # L driver = centre rotated −half_width (toward −right)
    l_dir = _norm3((
        centre[0]*cos_hw - right[0]*sin_hw,
        centre[1]*cos_hw - right[1]*sin_hw,
        centre[2]*cos_hw - right[2]*sin_hw,
    ))

    l_az = _dir_to_az(l_dir[0], l_dir[2])
    r_az = _dir_to_az(r_dir[0], r_dir[2])
    l_el = math.degrees(math.asin(max(-1., min(1., l_dir[1]))))
    r_el = math.degrees(math.asin(max(-1., min(1., r_dir[1]))))
    return l_az, l_el, r_az, r_el


def _build_routing_matrix(front_info: tuple, rear_info: tuple,
                           half_width: float = 15.0) -> np.ndarray:
    """
    Build 7×4 VBAP routing matrix for a 2-speaker stereo setup.

    Each speaker contributes 2 columns (L and R virtual drivers):
      Columns [0, 1] = front speaker's L and R drivers
      Columns [2, 3] = rear  speaker's L and R drivers

    The driver positions are computed from each speaker's actual 3-D position
    and facing direction, so the routing is correct for any speaker placement
    (front/rear, side/side, diagonal, etc.).

    Convention: columns are in device-physical order (L = speaker's left driver,
    R = speaker's right driver). The stream-level swap_rear_lr flag can still be
    used to correct physical wiring differences.

    Parameters
    ----------
    front_info : (az, el, face_az, face_el) for the front speaker (degrees)
    rear_info  : same for the rear speaker
    half_width : angular half-width between L/R virtual drivers (degrees)
    """
    _VIRT_AZ = [-30.0, 30.0, 0.0, -110.0, 110.0, -150.0, 150.0]

    f_az, f_el, f_faz, f_fel = (float(x) for x in front_info)
    r_az, r_el, r_faz, r_fel = (float(x) for x in rear_info)

    fl_az, _, fr_az, _ = _speaker_driver_az_el(f_az, f_el, f_faz, f_fel, half_width)
    rl_az, _, rr_az, _ = _speaker_driver_az_el(r_az, r_el, r_faz, r_fel, half_width)

    phys_az = [fl_az, fr_az, rl_az, rr_az]

    M = np.zeros((7, 4), dtype=np.float32)
    for i, vaz in enumerate(_VIRT_AZ):
        M[i] = _vbap_circle(vaz, phys_az)
    return M


def _build_routing_matrix_n_stereo(
    speaker_info: list, half_width: float = 15.0
) -> np.ndarray:
    """
    Build 7×2N VBAP routing matrix for N physical stereo speakers.

    Each speaker contributes two columns:
      Column 2*i   = L driver (speaker's left)
      Column 2*i+1 = R driver (speaker's right)

    Uses two-step routing to avoid VBAP degeneracy caused by adjacent
    inward-facing speakers sharing coincident inner driver positions:

      1. VBAP across N speaker center-azimuths → per-speaker gain (constant power)
      2. Within each speaker, constant-power pan between its L and R driver positions

    This preserves constant overall power (each step is constant-power) and
    correctly images the stereo field for any speaker placement and facing.

    Parameters
    ----------
    speaker_info : list of (az, el, face_az, face_el) for each speaker (degrees)
    half_width   : angular half-width between L/R virtual drivers (degrees)
    """
    _VIRT_AZ = [-30.0, 30.0, 0.0, -110.0, 110.0, -150.0, 150.0]
    N = len(speaker_info)

    # Pre-compute speaker center positions and L/R driver azimuths.
    # center_az_els stores (az, el) for each speaker so that step-1 VBAP
    # uses true 3D angular distance — this correctly routes to height speakers
    # (e.g. Dolby Atmos overhead) instead of relying on azimuth alone.
    center_az_els: list = []
    driver_pairs: list = []          # [(L_az, R_az), ...]
    for az, el, face_az, face_el in speaker_info:
        center_az_els.append((float(az), float(el)))
        l_az, _, r_az, _ = _speaker_driver_az_el(az, el, face_az, face_el, half_width)
        driver_pairs.append((l_az, r_az))

    # Virtual channels are defined on the horizontal plane (el=0°).
    _VIRT_EL = 0.0

    M = np.zeros((7, 2 * N), dtype=np.float32)

    for j, vaz in enumerate(_VIRT_AZ):
        # Step 1: route virtual source to speakers using 3D great-circle VBAP.
        # Handles elevation (e.g. Atmos height speakers) correctly while
        # being identical to horizontal VBAP when all elevations are zero.
        spk_gains = _vbap_sphere(vaz, _VIRT_EL, center_az_els)  # shape (N,)

        for i, (l_az, r_az) in enumerate(driver_pairs):
            g = float(spk_gains[i])
            if g < 1e-9:
                continue

            # Step 2: constant-power pan between this speaker's L and R drivers.
            # span = clockwise arc from L to R (always ≥ 0).
            # signed_diff = signed arc from L to source in (−180, +180]; clamping
            # it to [0, span] means sources left-of-L → all L, right-of-R → all R.
            span = (r_az - l_az) % 360.0       # arc from L to R going CW (>0)
            if span < 0.5:
                # Degenerate speaker (L≈R): split evenly
                M[j, 2*i]   += g * 0.7071068
                M[j, 2*i+1] += g * 0.7071068
                continue
            signed_diff = ((vaz - l_az + 180.0) % 360.0) - 180.0
            frac = max(0.0, min(1.0, signed_diff / span))
            M[j, 2*i]   += g * math.cos(frac * math.pi * 0.5)   # L gain
            M[j, 2*i+1] += g * math.sin(frac * math.pi * 0.5)   # R gain

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
    rear_az_deg  : azimuth magnitude of rear speaker (legacy, default 150°).
                   Used only if front_info / rear_info are not supplied.
    front_info   : (az, el, face_az, face_el) of the front speaker.
                   Defaults to a center-front speaker facing the listener.
    rear_info    : (az, el, face_az, face_el) of the rear speaker.
                   Defaults to a directly-behind speaker facing the listener.
    """

    _LO = 120.0   # sub-bass crossover (Hz)

    def __init__(self, fs: int = 48000, preset: dict | None = None,
                 bass_priority: str = "equal",
                 rear_az_deg: float = 150.0,
                 rear_direct_blend: float = 0.0,
                 front_info: tuple | None = None,
                 rear_info:  tuple | None = None):
        if preset is None:
            from config import HEADPHONES_PRESET
            preset = dict(HEADPHONES_PRESET)

        self._fs          = fs
        self._master_gain = 10 ** (float(preset.get("output_gain_db", 0.0)) / 20.0)

        # Keep for legacy update_rear_az compat
        self._rear_az_deg = float(np.clip(rear_az_deg, 60.0, 170.0))

        # Build speaker info tuples  (az, el, face_az, face_el)
        rr = float(self._rear_az_deg)
        self._front_info: tuple = front_info if front_info else (0.0, 0.0, 180.0, 0.0)
        self._rear_info:  tuple = rear_info  if rear_info  else (rr, 0.0, (rr+180.0)%360.0, 0.0)

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
        # Uses actual speaker positions and facing directions so the routing is
        # accurate for any layout (front/rear, side/side, diagonal, etc.).
        self._routing_matrix = _build_routing_matrix(self._front_info, self._rear_info)

        # -- "Behind" spectral coloring for rear bus --------------------------
        # Psychoacoustic cues that help localise the rear speaker as coming
        # from behind:
        #   • Low shelf +1.5 dB @ 280 Hz: room boundary bass buildup from rear
        #   • High shelf −3 dB  @ 4 kHz:  pinna HF shadow / distance rolloff
        self._rear_color = FilterChain([
            make_lowshelf( 280.0, +0.8, q=0.707, fs=fs, ch=2),
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

        # Apply "behind" spectral coloring to the rear VBAP mid-band signal
        # BEFORE mixing in sub-bass.  Doing it after would boost the sub through
        # the 280 Hz low-shelf, causing bass-induced limiter pumping on the rear bus.
        rear_mid = np.stack(
            [buses[:, 2] * self._rear_norm,
             buses[:, 3] * self._rear_norm], axis=1
        ).astype(np.float32)
        rear_mid = self._rear_color.process(rear_mid)

        # Add the (un-filtered) sub-bass contribution to the coloured mid signal
        rear_L = rear_mid[:, 0].astype(np.float64) + rs_l * self._rear_norm
        rear_R = rear_mid[:, 1].astype(np.float64) + rs_r * self._rear_norm

        front = np.stack([front_L, front_R], axis=1).astype(np.float32)
        rear  = np.stack([rear_L,  rear_R],  axis=1).astype(np.float32)

        # Per-bus limiting
        front = self._lim_front.process(front)
        rear  = self._lim_rear.process(rear)

        return front, rear

    # -------------------------------------------------------------------------

    def set_bass_priority(self, priority: str):
        """Update bass priority at runtime ('equal', 'front', 'rear')."""
        self._bass_priority = priority

    def update_rear_az(self, rear_az_deg: float):
        """Update rear speaker azimuth (legacy) and rebuild routing matrix."""
        rr = float(np.clip(rear_az_deg, 60.0, 170.0))
        self._rear_az_deg = rr
        # Keep facing toward listener by default
        self._rear_info = (rr, self._rear_info[1],
                           (rr + 180.0) % 360.0, self._rear_info[3])
        self._routing_matrix = _build_routing_matrix(self._front_info, self._rear_info)

    def update_speaker_info(self, front_info: tuple, rear_info: tuple):
        """Update both speakers' position and orientation, rebuild routing matrix.

        Parameters
        ----------
        front_info : (az_deg, el_deg, face_az_deg, face_el_deg)
        rear_info  : same
        """
        self._front_info = tuple(float(x) for x in front_info)
        self._rear_info  = tuple(float(x) for x in rear_info)
        self._rear_az_deg = abs(self._rear_info[0])
        self._routing_matrix = _build_routing_matrix(self._front_info, self._rear_info)

    def reset(self):
        for stage in (self._bass_enh, self._air_exc, self._eq, self._reverb,
                      self._comp, self._trans, self._lim_front, self._lim_rear,
                      self._lp_sub, self._hp_mid, self._rear_color):
            stage.reset()
        self._upmix.reset()

    @property
    def fs(self) -> int:
        return self._fs


# ---------------------------------------------------------------------------
# N-speaker DSP chain
# ---------------------------------------------------------------------------


class MultiSpeakerChainN:
    """
    Theater DSP chain for N physical speakers at arbitrary positions.

    Each call to process() returns a list of N (frames, 2) float32 arrays —
    one **stereo** bus per physical speaker.  The L and R channels of each bus
    correspond to the speaker's L and R virtual drivers as determined by the
    speaker's actual position and facing direction.  This produces correct
    directional imaging within each speaker's coverage zone for any layout.

    Parameters
    ----------
    fs                : sample rate (Hz)
    preset            : theater preset dict (same keys as HEADPHONES_PRESET)
    speaker_azimuths  : list of N azimuth angles in degrees
                        (0 = front, 90 = right, 180 = behind, −90 = left)
    speaker_elevations: list of N elevation angles in degrees (optional, default 0)
    speaker_face_azs  : list of N facing azimuths in degrees (optional,
                        defaults to each speaker facing toward the listener)
    speaker_face_els  : list of N facing elevations in degrees (optional, default 0)
    bass_priority     : "equal" | "front" | "rear"
    """

    _LO = 120.0  # sub-bass crossover

    def __init__(
        self,
        fs: int = 48000,
        preset: dict | None = None,
        speaker_azimuths:  list | None = None,
        speaker_elevations: list | None = None,
        speaker_face_azs:  list | None = None,
        speaker_face_els:  list | None = None,
        bass_priority: str = "equal",
    ):
        if preset is None:
            from config import HEADPHONES_PRESET
            preset = dict(HEADPHONES_PRESET)
        if not speaker_azimuths:
            speaker_azimuths = [-30.0, 30.0, -150.0, 150.0]

        self._fs = fs
        self._bass_priority = bass_priority
        self._master_gain = 10 ** (float(preset.get("output_gain_db", 0.0)) / 20.0)

        # Build the combined speaker_info list and rebuild routing matrix
        self._azimuths: list[float] = list(speaker_azimuths)
        self._N = len(self._azimuths)
        self._speaker_info = self._make_speaker_info(
            self._azimuths, speaker_elevations, speaker_face_azs, speaker_face_els)
        # 7 × 2N routing matrix (L and R driver columns per speaker)
        self._routing_matrix = _build_routing_matrix_n_stereo(self._speaker_info)

        # ---- Shared theater DSP (same pipeline as MultiSpeakerChain) ----
        self._bass_enh = HarmonicBassEnhancer(
            cutoff=120.0,
            drive=float(preset.get("bass_harm_drive",  2.8)),
            level=float(preset.get("bass_harm_level",  0.50)),
            fs=fs,
        )
        self._air_exc = AirBandExciter(
            cutoff=8000.0,
            level=float(preset.get("air_exciter_level", 0.18)),
            fs=fs,
        )
        self._eq     = CinemaEqualizer(fs=fs, num_channels=2, preset=preset)
        self._reverb = TheaterReverb(fs=fs, preset=preset)
        self._comp   = MultibandCompressor(
            fs=fs, drive=float(preset.get("mb_compress_drive", 1.6)))
        self._trans  = TransientEnhancer(
            fs=fs, amount=float(preset.get("transient_amount", 0.55)))

        self._lp_sub = make_lowpass(self._LO,  q=0.707, fs=fs, ch=2)
        self._hp_mid = make_highpass(self._LO, q=0.707, fs=fs, ch=2)
        self._upmix  = _AdaptiveUpmix71(fs)

        # ---- Per-bus limiters and behind-coloring filters ---------------
        thr = float(preset.get("limiter_threshold",  0.93))
        rel = float(preset.get("limiter_release_ms", 80.0))
        self._limiters = [
            PeakLimiter(threshold=thr, release_ms=rel, fs=fs)
            for _ in range(self._N)
        ]
        self._behind_filters = self._build_behind_filters(self._azimuths, fs)

        # ---- Channel levels ---------------------------------------------
        self._lfe_level  = float(preset.get("lfe_level",      0.85))
        self._surr_level = float(preset.get("surround_level", 0.72))
        self._rear_level = float(preset.get("rear_level",     0.60))
        self._cl         = float(preset.get("center_level",   0.88))

        # Normalisation: scale so N outputs sum to roughly the same power
        # as the 2-speaker case.
        self._norm = 1.0 / max(1.0, math.sqrt(self._N / 2.0))

    # ---- Static helpers -------------------------------------------------

    @staticmethod
    def _make_speaker_info(azimuths, elevations, face_azs, face_els) -> list:
        """Return [(az, el, face_az, face_el), ...] with sensible defaults."""
        N = len(azimuths)
        elevations = list(elevations) if elevations else [0.0] * N
        face_azs   = list(face_azs)   if face_azs   else [(a+180.0)%360.0 for a in azimuths]
        face_els   = list(face_els)   if face_els   else [0.0] * N
        return [(float(a), float(e), float(fa), float(fe))
                for a, e, fa, fe in zip(azimuths, elevations, face_azs, face_els)]

    @staticmethod
    def _build_behind_filters(azimuths: list, fs: int) -> list:
        """Return a FilterChain per speaker (None if speaker is in front or side).

        Behind-coloring is applied only to speakers that are clearly in the rear
        hemisphere (within 80° of directly-behind = 100°–260° azimuth range).
        Side speakers (≈ ±90°) are intentionally excluded so they do not
        receive the HF rolloff intended for rear-wall coloring.
        """
        filters = []
        for az in azimuths:
            az_n = float(az) % 360.0
            # Angular distance from directly-behind (180°)
            behind_deg = min(abs(az_n - 180.0), 360.0 - abs(az_n - 180.0))
            if behind_deg <= 80.0:   # 100° ≤ az ≤ 260° (rear hemisphere only)
                f = FilterChain([
                    make_lowshelf( 280.0, +0.8, q=0.707, fs=fs, ch=2),
                    make_highshelf(4000.0, -3.0, q=0.707, fs=fs, ch=2),
                ])
            else:
                f = None
            filters.append(f)
        return filters

    # ---- Sub-bass weight per speaker ------------------------------------

    def _lfe_weight(self, speaker_idx: int) -> float:
        az = self._azimuths[speaker_idx] % 360.0
        is_front = az <= 60.0 or az >= 300.0
        is_rear  = 120.0 <= az <= 240.0

        if self._bass_priority == "front":
            return 1.15 if is_front else 0.06
        if self._bass_priority == "rear":
            if is_front:
                return 0.06
            return 1.15 if is_rear else 0.40
        # "equal"
        return 1.0 if is_front else 0.40

    # ---- Main processing ------------------------------------------------

    def process(self, block: np.ndarray) -> list[np.ndarray]:
        """
        block : (frames, 2) float32 stereo input

        Returns
        -------
        List of N (frames, 2) float32 arrays, one per physical speaker.
        """
        if block.ndim == 1:
            block = np.stack([block, block], axis=1)
        x = block.astype(np.float32, copy=False)

        # Shared theater processing
        x = self._bass_enh.process(x)
        x = self._air_exc.process(x)
        x = self._eq.process(x)
        x = self._reverb.process(x)
        x = self._comp.process(x)
        x = self._trans.process(x)
        x = x * self._master_gain

        # Sub-bass crossover
        sub      = self._lp_sub.process(x)
        mid      = self._hp_mid.process(x)
        sub_mono = (sub[:, 0] + sub[:, 1]) * 0.5 * self._lfe_level  # (N_frames,)

        # 7-channel adaptive upmix
        ch = self._upmix.process(mid)

        # Stack into (frames, 7) matrix with per-channel levels applied
        sig = np.stack([
            ch["FL"].astype(np.float64),
            ch["FR"].astype(np.float64),
            ch["C"].astype(np.float64)  * self._cl,
            ch["LS"].astype(np.float64) * self._surr_level,
            ch["RS"].astype(np.float64) * self._surr_level,
            ch["LB"].astype(np.float64) * self._rear_level,
            ch["RB"].astype(np.float64) * self._rear_level,
        ], axis=1)   # (frames, 7)

        # VBAP routing: (frames, 7) @ (7, 2N) → (frames, 2N)
        # Each speaker occupies columns 2*i (L driver) and 2*i+1 (R driver).
        M     = self._routing_matrix.astype(np.float64)
        buses = sig @ M   # (frames, 2N)

        outputs: list[np.ndarray] = []
        sub_d = sub_mono.astype(np.float64)

        for i in range(self._N):
            sub_w = self._lfe_weight(i)
            # L and R come from their respective driver columns — NOT duplicated.
            # This produces genuine per-speaker stereo imaging so that content
            # panned toward one side of a speaker's coverage zone plays louder
            # on the corresponding physical driver.
            l_ch = (buses[:, 2*i]   * self._norm).astype(np.float32)
            r_ch = (buses[:, 2*i+1] * self._norm).astype(np.float32)
            stereo = np.stack([l_ch, r_ch], axis=1)

            # Apply "behind" spectral coloring to the mid-band VBAP signal
            # BEFORE adding sub-bass.  Doing it after would let the 280 Hz
            # low-shelf boost the sub component, driving the limiter into
            # audible pumping on rear/side speakers.
            if self._behind_filters[i] is not None:
                stereo = self._behind_filters[i].process(stereo)

            # Mix in the (un-filtered) sub-bass contribution
            sub_contrib = (sub_d * sub_w * self._norm).astype(np.float32)
            stereo[:, 0] += sub_contrib
            stereo[:, 1] += sub_contrib

            # Per-bus limiter
            stereo = self._limiters[i].process(stereo)
            outputs.append(stereo)

        return outputs

    # ---- Runtime updates ------------------------------------------------

    def update_speakers(
        self,
        azimuths:   list,
        elevations: list | None = None,
        face_azs:   list | None = None,
        face_els:   list | None = None,
    ):
        """
        Hot-update speaker positions/orientations and rebuild VBAP routing matrix.

        Parameters
        ----------
        azimuths   : list of N azimuth angles in degrees
        elevations : list of N elevation angles (degrees, optional → 0°)
        face_azs   : list of N facing azimuths (degrees, optional → toward listener)
        face_els   : list of N facing elevations (degrees, optional → 0°)
        """
        self._azimuths   = list(azimuths)
        self._speaker_info = self._make_speaker_info(
            self._azimuths, elevations, face_azs, face_els)
        n_new = len(self._azimuths)

        self._routing_matrix = _build_routing_matrix_n_stereo(self._speaker_info)

        if n_new != self._N:
            self._N = n_new
            thr    = self._limiters[0]._threshold if self._limiters else 0.93
            rel_ms = 80.0
            self._limiters = [
                PeakLimiter(threshold=thr, release_ms=rel_ms, fs=self._fs)
                for _ in range(self._N)
            ]
            self._norm = 1.0 / max(1.0, math.sqrt(self._N / 2.0))

        self._behind_filters = self._build_behind_filters(self._azimuths, self._fs)

    def update_speaker_azimuths(self, azimuths: list):
        """Backward-compatible alias for update_speakers (azimuth-only)."""
        self.update_speakers(azimuths)

    def set_bass_priority(self, priority: str):
        self._bass_priority = priority

    def reset(self):
        for stage in (self._bass_enh, self._air_exc, self._eq, self._reverb,
                      self._comp, self._trans, self._lp_sub, self._hp_mid):
            stage.reset()
        self._upmix.reset()
        for lim in self._limiters:
            lim.reset()
        for f in self._behind_filters:
            if f is not None:
                f.reset()

    @property
    def fs(self) -> int:
        return self._fs

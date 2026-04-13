"""
room_canvas_3d.py — Interactive 3D room visualisation widget.

Renders a perspective 3D view of a listening room using only tkinter Canvas
(no OpenGL or external 3D library required).  Provides:

  • Orbit camera   — left-drag to rotate, scroll to zoom
  • Listener dot   — red sphere at ear height, draggable on the floor plane
  • Speaker dots   — colour-coded per speaker, draggable in 3D
                     (horizontal drag = floor position, Shift-drag = height)
  • Orientation    — arrow showing speaker facing direction; right-click to
                     rotate; or use the external slider callbacks
  • Sound waves    — concentric spheres in 3 cross-section planes, animated
                     in real time from per-speaker RMS levels
  • Reflections    — mirror waves spawned from wall mirror-image positions
  • Directional emission — wave alpha modulated by cosine of angle from
                     speaker facing direction (cardioid roll-off)

Coordinate system
-----------------
  X = right (+)  /  Y = up (+, floor=0)  /  Z = front (-)  back (+)
  Azimuth  0° = front (-Z),  90° = right (+X),  180° = back (+Z)
  Elevation 0° = horizontal, 90° = straight up (+Y), -90° = down (-Y)
  Listener sits at (0, EAR_HEIGHT, 0).

Public API mirrors room_canvas.RoomCanvas so app.py can swap them.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EAR_HEIGHT     = 1.2    # metres — seated listener ear height
DEFAULT_ROOM_W = 6.0    # metres — room width  (X)
DEFAULT_ROOM_D = 5.0    # metres — room depth  (Z)
DEFAULT_ROOM_H = 2.8    # metres — room height (Y)

_BG         = "#0d1117"
_FLOOR_COL  = "#111823"
_WALL_COL   = "#1c2a3f"
_WALL_DIM   = "#1a2635"
_GRID_COL   = "#161e2e"
_AXIS_COL   = "#1c2840"
_TEXT_COL   = "#e8edf3"
_DIM_COL    = "#3a4a6a"

_LISTENER_COL = "#ef476f"
_SPK_COLS = [
    "#4361ee", "#7209b7", "#3a86ff", "#06d6a0",
    "#ffd166", "#f72585", "#4cc9f0", "#ff6b6b",
]

_WAVE_FPS      = 30      # animation frame interval (ms)

# Directional ray-beam wave system
_BEAM_SPEED    = 6.0    # visual propagation speed (m/s)
_BEAM_MAX_DIST = 7.5    # max ray travel distance before expiry (m)
_BEAM_IVMS     = 280    # ms between pulse emissions per speaker
_BEAM_N_RAYS   = 10     # horizontal rays per pulse
_BEAM_CONE_DEG = 72.0   # half-angle of emission cone (degrees)

# ---------------------------------------------------------------------------
# Speaker layouts with elevation support  (label, azimuth, elevation, distance)
# ---------------------------------------------------------------------------

SPEAKER_LAYOUTS_3D: dict[str, list[tuple[str, float, float, float]]] = {
    "2.0 Stereo": [
        ("FL", -30.0, 0.0, 2.5),
        ("FR",  30.0, 0.0, 2.5),
    ],
    "5.1 Cinema": [
        ("C",    0.0, 0.0, 2.5),
        ("FL", -30.0, 0.0, 2.5),
        ("FR",  30.0, 0.0, 2.5),
        ("SL", -110.0, 0.0, 2.5),
        ("SR",  110.0, 0.0, 2.5),
    ],
    "7.1 IMAX": [
        ("C",    0.0, 0.0, 2.5),
        ("FL", -30.0, 0.0, 2.5),
        ("FR",  30.0, 0.0, 2.5),
        ("SL", -90.0, 0.0, 2.5),
        ("SR",  90.0, 0.0, 2.5),
        ("BL", -150.0, 0.0, 2.0),
        ("BR",  150.0, 0.0, 2.0),
    ],
    "7.1.4 Dolby Atmos": [
        ("C",    0.0,  0.0, 2.5),
        ("FL", -30.0,  0.0, 2.5),
        ("FR",  30.0,  0.0, 2.5),
        ("SL", -90.0,  0.0, 2.5),
        ("SR",  90.0,  0.0, 2.5),
        ("BL", -150.0, 0.0, 2.0),
        ("BR",  150.0, 0.0, 2.0),
        ("TFL", -30.0, 45.0, 2.0),
        ("TFR",  30.0, 45.0, 2.0),
        ("TBL", -150.0, 45.0, 2.0),
        ("TBR",  150.0, 45.0, 2.0),
    ],
    "Home 5.1": [
        ("C",    0.0, 0.0, 1.8),
        ("FL", -30.0, 0.0, 1.8),
        ("FR",  30.0, 0.0, 1.8),
        ("SL", -110.0, 0.0, 1.5),
        ("SR",  110.0, 0.0, 1.5),
    ],
}

# Canonical channel directions for the output-device picker
CHANNEL_DIRECTIONS = [
    "Front L/R",
    "Front L",
    "Front R",
    "Center",
    "Surround L",
    "Surround R",
    "Rear L/R",
    "Rear L",
    "Rear R",
    "Height Front L",
    "Height Front R",
    "Height Rear L",
    "Height Rear R",
    "Subwoofer",
    "Custom",
]

# Map direction → (label, azimuth_deg, elevation_deg, distance_m)
DIRECTION_TO_SPEAKER: dict[str, tuple[str, float, float, float]] = {
    "Front L/R":       ("FL/FR",  0.0,   0.0,  2.5),
    "Front L":         ("FL",    -30.0,  0.0,  2.5),
    "Front R":         ("FR",     30.0,  0.0,  2.5),
    "Center":          ("C",       0.0,  0.0,  2.5),
    "Surround L":      ("SL",   -110.0,  0.0,  2.5),
    "Surround R":      ("SR",    110.0,  0.0,  2.5),
    "Rear L/R":        ("BL/BR", 180.0,  0.0,  2.0),
    "Rear L":          ("BL",   -150.0,  0.0,  2.0),
    "Rear R":          ("BR",    150.0,  0.0,  2.0),
    "Height Front L":  ("TFL",   -30.0, 45.0,  2.0),
    "Height Front R":  ("TFR",    30.0, 45.0,  2.0),
    "Height Rear L":   ("TBL",  -150.0, 45.0,  2.0),
    "Height Rear R":   ("TBR",   150.0, 45.0,  2.0),
    "Subwoofer":       ("SUB",     0.0,-20.0,  1.5),
    "Custom":          ("SPK",     0.0,  0.0,  2.0),
}

# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def _norm(v: tuple) -> tuple:
    x, y, z = v
    m = math.sqrt(x * x + y * y + z * z)
    return (x / m, y / m, z / m) if m > 1e-10 else (0.0, 0.0, 1.0)

def _cross(a: tuple, b: tuple) -> tuple:
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def _dot(a: tuple, b: tuple) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _sub(a: tuple, b: tuple) -> tuple:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _add(a: tuple, b: tuple) -> tuple:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _scale(v: tuple, s: float) -> tuple:
    return (v[0]*s, v[1]*s, v[2]*s)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Speaker3D:
    sid:           int
    label:         str
    x:             float  # world X (right+)
    y:             float  # world Y (up+, floor=0)
    z:             float  # world Z (front-, back+)
    face_az:       float = 0.0    # facing azimuth (0=front/-Z, 90=right/+X)
    face_el:       float = 0.0    # facing elevation (0=horizontal, 90=up)
    device_idx:    Optional[int] = None
    device_label:  str   = "Unassigned"
    level_rms:     float = 0.0
    level_left:    float = 0.0   # left-channel RMS (0–1)
    level_right:   float = 0.0   # right-channel RMS (0–1)
    active:        bool  = False
    # Slow ambient-floor (EWMA ≈ 2 s half-life) and transient spike
    level_avg_l:   float = 0.0   # ambient floor, left
    level_avg_r:   float = 0.0   # ambient floor, right
    spike_l:       float = 0.0   # instantaneous level above ambient
    spike_r:       float = 0.0   # instantaneous level above ambient
    # Stereo pan position: −1 = hard left, 0 = centre, +1 = hard right
    pan_center:    float = 0.0


@dataclass
class _WavePulse:
    """
    One emitted sound-wave pulse from a speaker.

    Carries N pre-traced directional rays.  Each ray is a list of segments:
      segments: list of ((ox,oy,oz), (ex,ey,ez), length_m)
    The animation advances a virtual wavefront along each ray path at
    _BEAM_SPEED m/s, drawing the travelled portion as a glowing line.

    alpha_l/r  — transient spike amplitude; drives bright foreground rays.
    ambient_l/r — slow-average level; used for the persistent speaker halo.
    pan_center  — stereo pan at the moment of emission (−1=L, 0=C, +1=R);
                  was baked into the cone direction at spawn time, stored
                  here so the drawing pass can tint spikes vs. ambient.
    """
    sid:       int
    born:      float      # time.time() at spawn
    alpha_l:   float      # spike-based left amplitude (0–1)
    alpha_r:   float      # spike-based right amplitude (0–1)
    ambient_l: float      # ambient-floor left (0–1)
    ambient_r: float      # ambient-floor right (0–1)
    pan_center: float     # −1..+1 stereo pan at birth
    color:     str
    rays:      list = field(default_factory=list)
    # rays: list of (channel 'L'|'R'|'C', dir_weight, segments)

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class _Camera:
    """Orbiting perspective camera."""

    def __init__(self):
        self.az   = 210.0   # azimuth: 0=camera at +Z, 90=+X, 180=-Z, 270=-X
        self.el   = 22.0    # elevation: 0=horizontal, 90=directly above
        self.dist = 11.0    # metres from target
        self.tx   = 0.0
        self.ty   = EAR_HEIGHT
        self.tz   = 0.0
        self.fov  = 58.0

    def _basis(self):
        az_r = math.radians(self.az)
        el_r = math.radians(max(-80.0, min(80.0, self.el)))
        c_el, s_el = math.cos(el_r), math.sin(el_r)
        c_az, s_az = math.cos(az_r), math.sin(az_r)

        cam_x = self.tx + self.dist * s_az * c_el
        cam_y = self.ty + self.dist * s_el
        cam_z = self.tz + self.dist * c_az * c_el

        fwd = _norm(_sub((self.tx, self.ty, self.tz), (cam_x, cam_y, cam_z)))
        world_up = (0.0, 1.0, 0.0)
        if abs(fwd[1]) > 0.98:
            world_up = (0.0, 0.0, -1.0) if fwd[1] > 0 else (0.0, 0.0, 1.0)
        right = _norm(_cross(fwd, world_up))
        up    = _norm(_cross(right, fwd))

        return (cam_x, cam_y, cam_z), right, up, fwd

    def project(self, x, y, z, cw, ch):
        """Returns (px, py, depth) or None if behind camera."""
        (cx, cy, cz), right, up, fwd = self._basis()
        p = _sub((x, y, z), (cx, cy, cz))
        cam_x = _dot(p, right)
        cam_y = _dot(p, up)
        cam_z = _dot(p, fwd)   # positive = in front of camera

        if cam_z < 0.02:
            return None

        ft = math.tan(math.radians(self.fov / 2))
        s  = (ch / 2) / ft / cam_z
        px = cw / 2 + cam_x * s
        py = ch / 2 - cam_y * s
        return px, py, cam_z

# ---------------------------------------------------------------------------
# Room3DCanvas
# ---------------------------------------------------------------------------

class Room3DCanvas:
    """
    3D interactive room visualisation widget.

    Parameters — identical to RoomCanvas so app.py can swap them.
    Additional kwargs:  room_height_m, on_speaker_rotated
    """

    HIT_R  = 14    # pixel hit-test radius
    SPK_R  = 7     # screen radius of speaker dot

    def __init__(
        self,
        parent,
        canvas_width:  int   = 620,
        canvas_height: int   = 340,
        room_width_m:  float = DEFAULT_ROOM_W,
        room_depth_m:  float = DEFAULT_ROOM_D,
        room_height_m: float = DEFAULT_ROOM_H,
        on_speaker_moved:    Optional[Callable] = None,
        on_listener_moved:   Optional[Callable] = None,
        on_speaker_selected: Optional[Callable] = None,
        on_speaker_rotated:  Optional[Callable] = None,
        on_change:           Optional[Callable] = None,
    ):
        self._cw = canvas_width
        self._ch = canvas_height
        self._room_w = float(room_width_m)
        self._room_d = float(room_depth_m)
        self._room_h = float(room_height_m)

        self._on_speaker_moved    = on_speaker_moved
        self._on_listener_moved   = on_listener_moved
        self._on_speaker_selected = on_speaker_selected
        self._on_speaker_rotated  = on_speaker_rotated
        self._on_change           = on_change

        self._cam = _Camera()

        # Listener world position
        self._lx, self._ly, self._lz = 0.0, EAR_HEIGHT, 0.0

        self._speakers: list[Speaker3D] = []
        self._next_sid = 0
        self._selected_sid: Optional[int] = None

        # Drag state
        self._drag_target = None   # "listener", int (sid), or "camera"
        self._drag_x0 = 0; self._drag_y0 = 0   # screen coords at press
        self._cam_az0 = 0.0; self._cam_el0 = 0.0
        self._drag_spk_x0 = 0.0; self._drag_spk_z0 = 0.0  # world floor coords

        # Rotation drag (shift+drag on selected speaker = face rotation)
        self._rot_mode = False
        self._rot_az0  = 0.0
        self._rot_el0  = 0.0

        # Wave animation
        self._waves: list[_WavePulse] = []
        self._last_emit: dict[int, float] = {}
        self._anim_running = False

        self._cvs = tk.Canvas(
            parent,
            width=canvas_width,
            height=canvas_height,
            bg=_BG,
            highlightthickness=1,
            highlightbackground=_WALL_COL,
            cursor="crosshair",
        )

        self._cvs.bind("<ButtonPress-1>",   self._on_press)
        self._cvs.bind("<B1-Motion>",       self._on_drag)
        self._cvs.bind("<ButtonRelease-1>", self._on_release)
        self._cvs.bind("<ButtonPress-3>",   self._on_right_click)
        self._cvs.bind("<MouseWheel>",      self._on_scroll)
        self._cvs.bind("<Button-4>",        self._on_scroll)   # Linux scroll up
        self._cvs.bind("<Button-5>",        self._on_scroll)   # Linux scroll down
        self._cvs.bind("<Configure>",       self._on_resize)
        self._cvs.bind("<Shift-B1-Motion>", self._on_shift_drag)

        self._draw_all()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def widget(self): return self._cvs
    def pack(self, **kw): self._cvs.pack(**kw)
    def grid(self, **kw): self._cvs.grid(**kw)

    # ---- speakers -------------------------------------------------------

    def add_speaker(
        self,
        label: str,
        azimuth_deg: float,
        elevation_deg: float = 0.0,
        distance_m: float = 2.5,
        device_idx: Optional[int] = None,
        device_label: str = "Unassigned",
    ) -> int:
        x, y, z = self._az_el_dist_to_xyz(azimuth_deg, elevation_deg, distance_m)
        face_az = (azimuth_deg + 180.0) % 360.0   # faces listener by default
        spk = Speaker3D(
            sid=self._next_sid,
            label=label, x=x, y=y, z=z,
            face_az=face_az, face_el=-elevation_deg,
            device_idx=device_idx, device_label=device_label,
        )
        self._speakers.append(spk)
        sid = self._next_sid
        self._next_sid += 1
        self._draw_all()
        if self._on_change: self._on_change()
        return sid

    def remove_speaker(self, sid: int):
        self._speakers = [s for s in self._speakers if s.sid != sid]
        if self._selected_sid == sid:
            self._selected_sid = None
        self._draw_all()
        if self._on_change: self._on_change()

    def clear_speakers(self):
        self._speakers.clear()
        self._selected_sid = None
        self._next_sid = 0
        self._draw_all()
        if self._on_change: self._on_change()

    def load_layout(self, layout_name: str) -> list[int]:
        self.clear_speakers()
        ids = []
        for label, az, el, dist in SPEAKER_LAYOUTS_3D.get(layout_name, []):
            ids.append(self.add_speaker(label, az, el, dist))
        return ids

    def get_speakers(self) -> list[Speaker3D]:
        return list(self._speakers)

    def get_selected_sid(self) -> Optional[int]:
        return self._selected_sid

    def set_selected_sid(self, sid: Optional[int]):
        self._selected_sid = sid
        self._draw_all()

    # ---- speaker properties ---------------------------------------------

    def set_speaker_device(self, sid: int, device_idx: Optional[int],
                           device_label: str):
        spk = self._find(sid)
        if spk:
            spk.device_idx  = device_idx
            spk.device_label = device_label
        self._draw_all()

    def set_speaker_label(self, sid: int, label: str):
        spk = self._find(sid)
        if spk: spk.label = label
        self._draw_all()

    def set_speaker_facing(self, sid: int, face_az: float, face_el: float):
        """Programmatically set speaker facing direction."""
        spk = self._find(sid)
        if spk:
            spk.face_az = float(face_az) % 360.0
            spk.face_el = max(-90.0, min(90.0, float(face_el)))
        self._draw_all()

    def set_speaker_position(self, sid: int, azimuth: float, elevation: float,
                              distance: float):
        """Programmatically reposition speaker by spherical coords from listener."""
        spk = self._find(sid)
        if spk:
            spk.x, spk.y, spk.z = self._az_el_dist_to_xyz(azimuth, elevation, distance)
        self._draw_all()

    def set_speaker_stereo_level(self, sid: int, left_rms: float, right_rms: float):
        """Feed real-time per-channel audio levels (0–1) to drive wave animation.

        Maintains a slow ambient-floor EWMA (≈2 s half-life at 20 Hz call rate)
        so instantaneous spikes above the background can be detected and rendered
        more prominently than steady ambient content.
        """
        spk = self._find(sid)
        if spk:
            L, R = float(left_rms), float(right_rms)
            spk.level_left  = L
            spk.level_right = R
            spk.level_rms   = (L + R) * 0.5
            spk.active      = spk.level_rms > 0.004

            # Slow EWMA for ambient floor.
            # alpha ≈ 0.966 → half-life ≈ 20 calls ≈ 1 second at 50 ms tick rate.
            _A = 0.966
            spk.level_avg_l = _A * spk.level_avg_l + (1.0 - _A) * L
            spk.level_avg_r = _A * spk.level_avg_r + (1.0 - _A) * R

            # Spike = level that exceeds 1.4× the ambient floor.
            spk.spike_l = max(0.0, L - spk.level_avg_l * 1.4)
            spk.spike_r = max(0.0, R - spk.level_avg_r * 1.4)

            # Stereo pan: +1 = all right, −1 = all left.
            tot = L + R
            spk.pan_center = (R - L) / tot if tot > 0.01 else 0.0

    def set_speaker_level(self, sid: int, level_rms: float):
        """Mono fallback — splits equally to both channels."""
        self.set_speaker_stereo_level(sid, level_rms, level_rms)

    # ---- room dimensions ------------------------------------------------

    def set_room_size(self, width_m: float, depth_m: float,
                      height_m: float = DEFAULT_ROOM_H):
        self._room_w = max(1.0, float(width_m))
        self._room_d = max(1.0, float(depth_m))
        self._room_h = max(1.0, float(height_m))
        self._draw_all()

    def get_room_size(self) -> tuple[float, float, float]:
        return self._room_w, self._room_d, self._room_h

    # ---- derived geometry -----------------------------------------------

    def get_speaker_azimuths(self) -> list[float]:
        return [self._xyz_to_azimuth(s.x, s.z) for s in self._speakers]

    def get_speaker_elevations(self) -> list[float]:
        return [self._xyz_to_elevation(s.x, s.y, s.z) for s in self._speakers]

    def get_speaker_distances_m(self) -> list[float]:
        return [math.sqrt((s.x-self._lx)**2 + (s.y-self._ly)**2 + (s.z-self._lz)**2)
                for s in self._speakers]

    def get_speaker_spherical(self, sid: int) -> tuple:
        """Return (azimuth_deg, elevation_deg, distance_m) of speaker from listener."""
        spk = self._find(sid)
        if spk is None:
            return (0.0, 0.0, 2.0)
        az   = self._xyz_to_azimuth(spk.x, spk.z)
        el   = self._xyz_to_elevation(spk.x, spk.y, spk.z)
        dist = math.sqrt((spk.x - self._lx)**2 + (spk.y - self._ly)**2 +
                         (spk.z - self._lz)**2)
        return az, el, max(dist, 0.1)

    # ---- animation ------------------------------------------------------

    def start_animation(self):
        if not self._anim_running:
            self._anim_running = True
            self._animate()

    def stop_animation(self):
        self._anim_running = False
        self._waves.clear()
        for s in self._speakers:
            s.level_rms = 0.0
            s.active    = False
        self._draw_all()

    # ------------------------------------------------------------------ #
    # Coordinate maths
    # ------------------------------------------------------------------ #

    def _az_el_dist_to_xyz(self, az: float, el: float, dist: float) -> tuple:
        """Azimuth/elevation/distance from listener → absolute world XYZ."""
        az_r = math.radians(az)
        el_r = math.radians(el)
        ce   = math.cos(el_r)
        dx   =  math.sin(az_r) * ce * dist
        dy   =  math.sin(el_r) * dist
        dz   = -math.cos(az_r) * ce * dist
        return self._lx + dx, self._ly + dy, self._lz + dz

    def _xyz_to_azimuth(self, x: float, z: float) -> float:
        dx = x - self._lx
        dz = z - self._lz
        return math.degrees(math.atan2(dx, -dz)) % 360.0

    def _xyz_to_elevation(self, x: float, y: float, z: float) -> float:
        dx = x - self._lx
        dy = y - self._ly
        dz = z - self._lz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return math.degrees(math.asin(dy / dist)) if dist > 1e-6 else 0.0

    def _facing_unit_vec(self, face_az: float, face_el: float) -> tuple:
        """Speaker facing unit vector in world space."""
        az_r = math.radians(face_az)
        el_r = math.radians(face_el)
        ce = math.cos(el_r)
        return (math.sin(az_r) * ce, math.sin(el_r), -math.cos(az_r) * ce)

    def _clamp_to_room(self, x: float, y: float, z: float) -> tuple:
        hw, hd = self._room_w / 2.0, self._room_d / 2.0
        margin = 0.2
        x = max(-hw + margin, min(hw - margin, x))
        y = max(0.2,           min(self._room_h - 0.2, y))
        z = max(-hd + margin,  min(hd - margin, z))
        return x, y, z

    # ------------------------------------------------------------------ #
    # 3D drawing helpers
    # ------------------------------------------------------------------ #

    def _proj(self, x, y, z):
        return self._cam.project(x, y, z, self._cw, self._ch)

    def _circle_3d(self, cx, cy, cz, r, axis, n=32):
        """
        Sample n world-space points on a circle of radius r centred at
        (cx,cy,cz) in the plane perpendicular to 'axis' ('x','y','z').
        Returns list of projected (px,py) tuples (skips points behind camera).
        """
        pts = []
        for i in range(n):
            theta = 2 * math.pi * i / n
            c, s  = math.cos(theta), math.sin(theta)
            if axis == 'y':
                wx, wy, wz = cx + r*c, cy, cz + r*s
            elif axis == 'x':
                wx, wy, wz = cx, cy + r*c, cz + r*s
            else:
                wx, wy, wz = cx + r*c, cy + r*s, cz
            pp = self._proj(wx, wy, wz)
            if pp:
                pts.append((pp[0], pp[1]))
        return pts

    def _draw_poly(self, pts, fill="", outline="white", width=1, **kw):
        if len(pts) < 2:
            return
        flat = []
        for p in pts:
            flat.extend(p)
        flat.extend(pts[0])  # close loop
        try:
            self._cvs.create_line(*flat, fill=outline, width=width, smooth=True)
        except tk.TclError:
            pass

    def _draw_line_3d(self, x1, y1, z1, x2, y2, z2, color, width=1, dash=None):
        p1 = self._proj(x1, y1, z1)
        p2 = self._proj(x2, y2, z2)
        if p1 and p2:
            kw = {"fill": color, "width": width}
            if dash:
                kw["dash"] = dash
            self._cvs.create_line(p1[0], p1[1], p2[0], p2[1], **kw)

    def _draw_dot_3d(self, x, y, z, r_px, fill, outline="white", outline_w=1):
        pp = self._proj(x, y, z)
        if pp:
            px, py, depth = pp
            # Depth-scale the radius slightly
            s = max(0.4, min(1.6, 8.0 / depth))
            r = r_px * s
            self._cvs.create_oval(px-r, py-r, px+r, py+r,
                                  fill=fill, outline=outline, width=outline_w)

    def _depth_color(self, base_hex: str, depth: float, max_depth: float = 15.0) -> str:
        """Darken a colour proportionally to depth (atmospheric perspective)."""
        t = max(0.0, min(1.0, depth / max_depth))
        frac = 1.0 - t * 0.5
        h = base_hex.lstrip("#")
        r = max(0, int(int(h[:2], 16) * frac))
        g = max(0, int(int(h[2:4], 16) * frac))
        b = max(0, int(int(h[4:], 16) * frac))
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _blend(fg: str, bg: str, a: float) -> str:
        def _p(h):
            h = h.lstrip("#")
            return int(h[:2],16), int(h[2:4],16), int(h[4:],16)
        try:
            fr,fg2,fb = _p(fg); br,bg2,bb = _p(bg)
            r = max(0,min(255, int(fr*a + br*(1-a))))
            g = max(0,min(255, int(fg2*a + bg2*(1-a))))
            b = max(0,min(255, int(fb*a + bb*(1-a))))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return fg

    # ------------------------------------------------------------------ #
    # Ray-physics helpers (sound-wave propagation)
    # ------------------------------------------------------------------ #

    def _ray_room_hit(self, ox: float, oy: float, oz: float,
                      dx: float, dy: float, dz: float) -> tuple:
        """
        Find the first axis-aligned room wall hit for a ray (origin + direction).
        Returns (t, (nx, ny, nz)) where t is hit distance and n is surface normal.
        """
        hw, hd, rh = self._room_w / 2.0, self._room_d / 2.0, self._room_h
        t_best = 1e9
        n_best = (1, 0, 0)
        for (d_comp, wall_pos, o_comp, normal) in [
            (dx,  hw,  ox, ( 1, 0, 0)),
            (dx, -hw,  ox, (-1, 0, 0)),
            (dy,  rh,  oy, ( 0, 1, 0)),
            (dy,  0.0, oy, ( 0,-1, 0)),
            (dz,  hd,  oz, ( 0, 0, 1)),
            (dz, -hd,  oz, ( 0, 0,-1)),
        ]:
            if abs(d_comp) < 1e-9:
                continue
            t = (wall_pos - o_comp) / d_comp
            if 0.04 < t < t_best:
                t_best = t
                n_best = normal
        return t_best, n_best

    def _trace_ray_segs(self, ox: float, oy: float, oz: float,
                        dx: float, dy: float, dz: float,
                        max_bounces: int = 1) -> list:
        """
        Trace a ray through the room with up to max_bounces wall reflections.
        Returns list of ((ox,oy,oz), (ex,ey,ez), segment_length_m).
        """
        segs = []
        x, y, z = ox, oy, oz
        total = 0.0
        for _ in range(max_bounces + 1):
            t, (nx, ny, nz) = self._ray_room_hit(x, y, z, dx, dy, dz)
            remaining = _BEAM_MAX_DIST - total
            t = min(t, remaining)
            if t <= 0:
                break
            ex, ey, ez = x + dx * t, y + dy * t, z + dz * t
            segs.append(((x, y, z), (ex, ey, ez), t))
            total += t
            if total >= _BEAM_MAX_DIST - 1e-6:
                break
            # Specular reflection: d -= 2*(d·n)*n
            dn = dx * nx + dy * ny + dz * nz
            dx -= 2 * dn * nx
            dy -= 2 * dn * ny
            dz -= 2 * dn * nz
            x, y, z = ex, ey, ez
        return segs

    def _spawn_pulse(self, spk: "Speaker3D", col_idx: int) -> "_WavePulse":
        """
        Build a new _WavePulse from the given speaker.

        Three key improvements over the original:

        1. **Pan-shifted cone** — the emission cone centre is rotated toward
           the speaker's dominant channel (up to ±_PAN_MAX_DEG degrees along
           the speaker's right vector). When audio pans from left to right the
           ray cluster visually sweeps across the speaker face.

        2. **Per-driver origin offsets** — L-channel rays originate from the
           physical left-driver position, R-channel rays from the right-driver
           position (offset ±_DRIVER_OFFSET metres along the right vector).
           This makes the source of moving audio traceable to a specific part
           of the speaker cabinet.

        3. **Spike / ambient split** — alpha is computed from the transient
           *spike* above the ambient floor, making foreground events (e.g. a
           bird flapping) visually distinct from the constant background hum.
           The ambient level is stored separately so _draw_sound_waves can
           draw it as a softer, thicker halo alongside the sharp spike rays.
        """
        _PAN_MAX_DEG   = 28.0   # max cone-centre shift from neutral (degrees)
        _DRIVER_OFFSET = 0.11   # physical L/R driver separation from centre (m)

        col = _SPK_COLS[col_idx % len(_SPK_COLS)]
        fv  = self._facing_unit_vec(spk.face_az, spk.face_el)

        # Speaker-space right vector: cross(world_up, forward) = physical right.
        world_up = (0.0, 1.0, 0.0)
        if abs(fv[1]) > 0.9:
            world_up = (0.0, 0.0, 1.0)
        right = _norm(_cross(world_up, fv))

        # ----- 1. Pan-shifted cone centre ------------------------------------
        # Rotate fv toward 'right' by pan_center * _PAN_MAX_DEG so rays cluster
        # toward whichever driver is playing louder content.
        pan_shift = math.radians(spk.pan_center * _PAN_MAX_DEG)
        c_ps, s_ps = math.cos(pan_shift), math.sin(pan_shift)
        fv_pan = _norm((
            fv[0]*c_ps + right[0]*s_ps,
            fv[1]*c_ps + right[1]*s_ps,
            fv[2]*c_ps + right[2]*s_ps,
        ))

        # ----- 2. Per-driver world-space origins -----------------------------
        L_origin = (spk.x - right[0]*_DRIVER_OFFSET,
                    spk.y - right[1]*_DRIVER_OFFSET,
                    spk.z - right[2]*_DRIVER_OFFSET)
        R_origin = (spk.x + right[0]*_DRIVER_OFFSET,
                    spk.y + right[1]*_DRIVER_OFFSET,
                    spk.z + right[2]*_DRIVER_OFFSET)
        C_origin = (spk.x, spk.y, spk.z)

        # ----- 3. Build rays -------------------------------------------------
        n    = _BEAM_N_RAYS
        cone = math.radians(_BEAM_CONE_DEG)
        rays = []

        for i in range(n):
            az  = -cone + 2.0 * cone * i / (n - 1) if n > 1 else 0.0
            c_az, s_az = math.cos(az), math.sin(az)

            # Rotate pan-shifted facing by az around world_up
            dx = fv_pan[0]*c_az + right[0]*s_az
            dy = fv_pan[1]*c_az + right[1]*s_az
            dz = fv_pan[2]*c_az + right[2]*s_az
            d  = _norm((dx, dy, dz))

            # Cardioid directional weight (exponent 0.4 ≈ wide supercardioid)
            dir_w = max(0.05, c_az ** 0.4)

            # Channel assignment and origin offset.
            # s_az < 0 → rotated toward −right = speaker's left driver.
            # s_az > 0 → rotated toward +right = speaker's right driver.
            if s_az < -0.12:
                ch, origin = 'L', L_origin
            elif s_az >  0.12:
                ch, origin = 'R', R_origin
            else:
                ch, origin = 'C', C_origin

            segs = self._trace_ray_segs(
                origin[0], origin[1], origin[2],
                d[0], d[1], d[2], max_bounces=1)
            rays.append((ch, dir_w, segs))

        # ----- Spike / ambient amplitude encoding ----------------------------
        # spike_l/r: transient above ambient floor → drives bright foreground rays
        # ambient_l/r: slow average → drives the persistent soft halo
        spike_l  = min(1.0, spk.spike_l  * 5.0)
        spike_r  = min(1.0, spk.spike_r  * 5.0)
        amb_l    = min(0.55, spk.level_avg_l * 2.0)
        amb_r    = min(0.55, spk.level_avg_r * 2.0)

        # Combined alpha: spike dominates; ambient provides a soft floor so
        # even constant sounds have *some* visible ray activity.
        alpha_l  = min(1.0, spike_l + amb_l * 0.28)
        alpha_r  = min(1.0, spike_r + amb_r * 0.28)

        return _WavePulse(
            sid=spk.sid,
            born=time.time(),
            alpha_l=alpha_l,
            alpha_r=alpha_r,
            ambient_l=amb_l,
            ambient_r=amb_r,
            pan_center=spk.pan_center,
            color=col,
            rays=rays,
        )

    # ------------------------------------------------------------------ #
    # Main draw
    # ------------------------------------------------------------------ #

    def _draw_all(self):
        c = self._cvs
        c.delete("all")
        self._draw_room_back()
        self._draw_floor_grid()
        self._draw_sound_waves()
        self._draw_room_front()
        self._draw_speakers()
        self._draw_listener()
        self._draw_hud()

    def _room_corners(self):
        hw, hd, rh = self._room_w/2, self._room_d/2, self._room_h
        # (x, y, z) floor corners, then ceiling corners
        return [
            (-hw, 0,  -hd), ( hw, 0,  -hd),  # front-left, front-right (floor)
            ( hw, 0,   hd), (-hw, 0,   hd),  # back-right, back-left (floor)
            (-hw, rh, -hd), ( hw, rh, -hd),  # front ceiling
            ( hw, rh,  hd), (-hw, rh,  hd),  # back ceiling
        ]

    def _draw_room_back(self):
        """Draw the floor, back wall, side walls (behind speakers, ocluded by them)."""
        c = self._cvs
        corners = self._room_corners()

        def edge(i, j, color, width=1, dash=None):
            self._draw_line_3d(*corners[i], *corners[j], color, width, dash)

        # Floor
        hw, hd = self._room_w/2, self._room_d/2
        fp = []
        for cx, cz in [(-hw,-hd),(hw,-hd),(hw,hd),(-hw,hd)]:
            pp = self._proj(cx, 0, cz)
            if pp: fp.extend([pp[0], pp[1]])
        if len(fp) >= 8:
            try:
                c.create_polygon(*fp, fill=_FLOOR_COL, outline=_WALL_COL, width=1)
            except tk.TclError:
                pass

        # Back wall
        bwpts = []
        for x, y, z in [(-hw,0,hd),(hw,0,hd),(hw,self._room_h,hd),(-hw,self._room_h,hd)]:
            pp = self._proj(x, y, z)
            if pp: bwpts.extend([pp[0], pp[1]])
        if len(bwpts) >= 8:
            try:
                c.create_polygon(*bwpts, fill=_WALL_DIM, outline=_WALL_COL, width=1)
            except tk.TclError:
                pass

        # Side walls (left and right)
        for side_x in [-hw, hw]:
            swpts = []
            for x, y, z in [(side_x,0,-hd),(side_x,0,hd),
                             (side_x,self._room_h,hd),(side_x,self._room_h,-hd)]:
                pp = self._proj(x, y, z)
                if pp: swpts.extend([pp[0], pp[1]])
            if len(swpts) >= 8:
                try:
                    c.create_polygon(*swpts, fill=_WALL_DIM, outline=_WALL_COL, width=1)
                except tk.TclError:
                    pass

        # Vertical wall edges (dashed to show depth)
        edge(0, 4, _DIM_COL, 1)   # front-left floor→ceiling
        edge(1, 5, _DIM_COL, 1)   # front-right
        edge(2, 6, _DIM_COL, 1)   # back-right
        edge(3, 7, _DIM_COL, 1)   # back-left

    def _draw_floor_grid(self):
        hw, hd = self._room_w/2, self._room_d/2
        step = 1.0  # 1-metre grid
        z = -hd
        while z <= hd + 0.01:
            self._draw_line_3d(-hw, 0, z, hw, 0, z, _GRID_COL, 1)
            z += step
        x = -hw
        while x <= hw + 0.01:
            self._draw_line_3d(x, 0, -hd, x, 0, hd, _GRID_COL, 1)
            x += step

        # Axis cross through listener
        self._draw_line_3d(self._lx, 0.01, -hd,
                           self._lx, 0.01,  hd, _AXIS_COL, 1)
        self._draw_line_3d(-hw, 0.01, self._lz,
                            hw, 0.01, self._lz, _AXIS_COL, 1)

    def _draw_room_front(self):
        """Draw front wall and ceiling edges (in front of speakers)."""
        corners = self._room_corners()
        hw, hd, rh = self._room_w/2, self._room_d/2, self._room_h

        # Front wall polygon
        fwpts = []
        for x, y, z in [(-hw,0,-hd),(hw,0,-hd),(hw,rh,-hd),(-hw,rh,-hd)]:
            pp = self._proj(x, y, z)
            if pp: fwpts.extend([pp[0], pp[1]])
        if len(fwpts) >= 8:
            try:
                self._cvs.create_polygon(*fwpts, fill=_WALL_DIM,
                                         outline=_WALL_COL, width=1)
            except tk.TclError:
                pass

        # Ceiling
        cpts = []
        for x, y, z in [(-hw,rh,-hd),(hw,rh,-hd),(hw,rh,hd),(-hw,rh,hd)]:
            pp = self._proj(x, y, z)
            if pp: cpts.extend([pp[0], pp[1]])
        if len(cpts) >= 8:
            try:
                self._cvs.create_polygon(*cpts, fill=_WALL_DIM,
                                         outline=_WALL_COL, width=1)
            except tk.TclError:
                pass

        # Ceiling edges
        for a, b in [(4,5),(5,6),(6,7),(7,4)]:
            self._draw_line_3d(*corners[a], *corners[b], _WALL_COL, 1)

    def _draw_sound_waves(self):
        """
        Draw directional sound-wave ray traces in 3D.

        Rendering distinguishes two layers of content:

        **Foreground / spike rays** — alpha driven by spike_l/r (level above the
        ambient floor).  These are bright, may be thicker, and have a vivid
        wavefront dot.  They stand out visually so moving or transient audio
        (e.g. a bird flapping across a nature scene) is easy to track.

        **Ambient rays** — alpha driven by the ambient floor average.  These are
        softer (lower alpha, thinner lines) but still show which channels of
        each speaker carry the persistent background sound.

        The cone direction at spawn time was already shifted toward the dominant
        channel (pan_center), so the entire cluster of rays naturally leans
        toward whichever side of the speaker is currently louder.
        """
        now = time.time()
        for pulse in self._waves:
            dist = _BEAM_SPEED * (now - pulse.born)
            if dist <= 0:
                continue

            # Pulse fades out as wavefront reaches max distance
            global_alpha = max(0.0, 1.0 - dist / _BEAM_MAX_DIST)

            # Is this pulse dominated by a foreground spike?
            avg_amb   = (pulse.ambient_l + pulse.ambient_r) * 0.5
            avg_spike = (pulse.alpha_l   + pulse.alpha_r)   * 0.5
            is_spike  = avg_spike > avg_amb * 1.6 + 0.05

            for ch, dir_w, segs in pulse.rays:
                # Per-channel amplitude
                if ch == 'L':
                    ch_amp = pulse.alpha_l
                elif ch == 'R':
                    ch_amp = pulse.alpha_r
                else:
                    ch_amp = (pulse.alpha_l + pulse.alpha_r) * 0.5

                if ch_amp < 0.008:
                    continue

                # Walk precomputed path segments up to current wavefront
                traveled = 0.0
                for seg_i, (seg_s, seg_e, seg_len) in enumerate(segs):
                    if traveled >= dist:
                        break
                    if seg_len < 1e-6:
                        continue

                    rem    = dist - traveled
                    draw_t = min(rem, seg_len)
                    frac   = draw_t / seg_len

                    tip = (
                        seg_s[0] + (seg_e[0] - seg_s[0]) * frac,
                        seg_s[1] + (seg_e[1] - seg_s[1]) * frac,
                        seg_s[2] + (seg_e[2] - seg_s[2]) * frac,
                    )

                    # Attenuation: directional × inverse-distance × bounce × age
                    mid_dist   = traveled + draw_t * 0.5
                    inv_sq     = 1.0 / (1.0 + mid_dist * 0.4)
                    bounce_fac = 0.52 ** seg_i
                    alpha      = ch_amp * dir_w * inv_sq * bounce_fac * global_alpha

                    if alpha < 0.010:
                        traveled += seg_len
                        continue

                    # Spike rays: brighter colour, thicker at high alpha
                    # Ambient rays: softer blend, always thin
                    if is_spike:
                        line_col = self._blend(pulse.color, _BG, min(1.0, alpha))
                        width    = 3 if alpha > 0.55 else (2 if alpha > 0.28 else 1)
                    else:
                        # Ambient: blend more toward background for subtlety
                        line_col = self._blend(pulse.color, _BG, min(0.72, alpha * 0.78))
                        width    = 1

                    self._draw_line_3d(
                        seg_s[0], seg_s[1], seg_s[2],
                        tip[0],   tip[1],   tip[2],
                        line_col, width=width,
                    )

                    # Wavefront tip dot — larger / brighter for spike content
                    if frac < 1.0:
                        if is_spike:
                            tip_alpha = min(1.0, alpha * 2.4)
                            min_tap   = 0.07
                            dot_scale = 4.0
                        else:
                            tip_alpha = min(0.65, alpha * 1.5)
                            min_tap   = 0.12
                            dot_scale = 2.2
                        if tip_alpha > min_tap:
                            tip_col = self._blend(pulse.color, "#ffffff", tip_alpha)
                            tp = self._proj(tip[0], tip[1], tip[2])
                            if tp:
                                r = max(1, int(tip_alpha * dot_scale))
                                self._cvs.create_oval(
                                    tp[0]-r, tp[1]-r, tp[0]+r, tp[1]+r,
                                    fill=tip_col, outline="")

                    traveled += seg_len

    def _draw_speakers(self):
        c = self._cvs

        # Collect projected positions for depth-sorting
        projected = []
        for i, spk in enumerate(self._speakers):
            pp = self._proj(spk.x, spk.y, spk.z)
            if pp:
                projected.append((pp[2], i, spk, pp))  # (depth, index, spk, pp)

        # Draw back-to-front
        projected.sort(key=lambda t: -t[0])

        col_idx = {s.sid: i % len(_SPK_COLS) for i, s in enumerate(self._speakers)}

        for depth, _, spk, pp in projected:
            px, py, _ = pp
            col = _SPK_COLS[col_idx[spk.sid]]
            sel = (spk.sid == self._selected_sid)
            sr  = self.SPK_R

            # Ambient halo — persistent soft glow for constant background sound.
            # Drawn first (widest radius) so spike glow and speaker body sit above.
            amb = (spk.level_avg_l + spk.level_avg_r) * 0.5
            if amb > 0.015:
                amb_r  = sr + 6 + int(amb * 38)           # 6–44 px radius
                amb_a  = min(0.38, amb * 0.70)            # cap at 38 % opacity
                amb_col = self._blend(col, _BG, amb_a)
                c.create_oval(px - amb_r, py - amb_r,
                              px + amb_r, py + amb_r,
                              fill=amb_col, outline="")

            # Foreground transient glow — only when a spike is detected.
            spk_spike = (spk.spike_l + spk.spike_r) * 0.5
            if spk.active and spk_spike > 0.02:
                gr   = sr + int(spk_spike * 22)
                gcol = self._blend(col, "#ffffff", spk_spike * 0.45)
                c.create_oval(px-gr, py-gr, px+gr, py+gr, fill=gcol, outline="")
            elif spk.active and spk.level_rms > 0.01:
                # Soft active glow even without spike
                gr   = sr + int(spk.level_rms * 9)
                gcol = self._blend(col, _BG, spk.level_rms * 0.28)
                c.create_oval(px-gr, py-gr, px+gr, py+gr, fill=gcol, outline="")

            # Selection halo
            if sel:
                c.create_oval(px-sr-6, py-sr-6, px+sr+6, py+sr+6,
                              fill="", outline="white", width=1)

            # Shadow on floor
            shadow_pp = self._proj(spk.x, 0, spk.z)
            if shadow_pp:
                sx, sy = shadow_pp[0], shadow_pp[1]
                c.create_oval(sx-4, sy-1, sx+4, sy+1,
                              fill="#222", outline="")
                c.create_line(px, py, sx, sy, fill="#333", width=1, dash=(3,3))

            # Orientation arrow
            fv = self._facing_unit_vec(spk.face_az, spk.face_el)
            arrow_len = 0.7
            ax = spk.x + fv[0] * arrow_len
            ay = spk.y + fv[1] * arrow_len
            az = spk.z + fv[2] * arrow_len
            ap = self._proj(ax, ay, az)
            if ap:
                acol = self._blend(col, _BG, 0.65)
                c.create_line(px, py, ap[0], ap[1],
                              fill=acol, width=2,
                              arrow=tk.LAST, arrowshape=(7, 9, 3))

            # Speaker body (slightly depth-shaded)
            body_col = self._depth_color(col, depth)
            if spk.active:
                body_col = self._blend(col, "#ffffff", 0.18)
            rim = "white" if sel else self._blend(col, "#ffffff", 0.4)
            c.create_oval(px-sr, py-sr, px+sr, py+sr,
                          fill=body_col, outline=rim,
                          width=2 if sel else 1)

            # Label
            c.create_text(px, py, text=spk.label,
                          fill="white", font=("Consolas", 6, "bold"),
                          anchor="center")

            # Device label
            dlbl = spk.device_label[:14] if spk.device_label else "—"
            dcol = self._blend(col, _BG, 0.75)
            c.create_text(px, py+sr+9, text=dlbl,
                          fill=dcol, font=("Consolas", 6), anchor="center")

    def _draw_listener(self):
        c   = self._cvs
        pp  = self._proj(self._lx, self._ly, self._lz)
        if not pp:
            return
        px, py, depth = pp
        lr = 9

        # Shadow on floor
        sp = self._proj(self._lx, 0, self._lz)
        if sp:
            sx, sy = sp[0], sp[1]
            c.create_oval(sx-5, sy-2, sx+5, sy+2, fill="#222", outline="")
            c.create_line(px, py, sx, sy, fill="#333", width=1, dash=(3,3))

        # Halo
        c.create_oval(px-lr-6, py-lr-6, px+lr+6, py+lr+6,
                      fill="", outline=self._blend(_LISTENER_COL, _BG, 0.18),
                      width=1)
        # Body
        c.create_oval(px-lr, py-lr, px+lr, py+lr,
                      fill=_LISTENER_COL, outline="#ff8888", width=1)
        # Up indicator
        c.create_text(px, py, text="♦", fill="white",
                      font=("Arial", 7), anchor="center")
        c.create_text(px+lr+6, py, text="YOU",
                      fill=_LISTENER_COL, font=("Consolas", 7, "bold"), anchor="w")

    def _draw_hud(self):
        c = self._cvs
        # Camera info badge
        c.create_text(
            self._cw - 8, 8,
            text=f"az:{self._cam.az:.0f}° el:{self._cam.el:.0f}°  "
                 f"Room {self._room_w:.1f}×{self._room_d:.1f}×{self._room_h:.1f} m",
            fill=_DIM_COL,
            font=("Consolas", 7),
            anchor="ne",
        )
        # Drag help
        c.create_text(
            8, self._ch - 8,
            text="Drag=orbit  Scroll=zoom  Click dot=select  Shift-drag=raise/lower",
            fill=_DIM_COL, font=("Consolas", 7), anchor="sw",
        )
        # Axis labels (project axis tips)
        for x, y, z, lbl in [(1.5,0.05,0,"→X"), (0,1.5,0,"↑Y"), (0,0.05,-1.5,"▲Z front")]:
            pp = self._proj(x+self._lx, y+self._ly, z+self._lz)
            if pp:
                c.create_text(pp[0], pp[1], text=lbl, fill=_DIM_COL,
                              font=("Consolas", 7), anchor="center")

    # ------------------------------------------------------------------ #
    # Animation
    # ------------------------------------------------------------------ #

    def _animate(self):
        if not self._anim_running:
            return

        now = time.time()
        iv  = _BEAM_IVMS / 1000.0

        col_idx = {s.sid: i for i, s in enumerate(self._speakers)}

        # Emit new pulses from active speakers
        for spk in self._speakers:
            if not spk.active:
                continue
            last = self._last_emit.get(spk.sid, 0.0)
            if now - last >= iv:
                self._last_emit[spk.sid] = now
                self._waves.append(
                    self._spawn_pulse(spk, col_idx.get(spk.sid, 0)))

        # Prune expired pulses (wavefront has traveled past max distance)
        self._waves = [
            p for p in self._waves
            if _BEAM_SPEED * (now - p.born) < _BEAM_MAX_DIST
        ]

        self._draw_all()

        try:
            self._cvs.after(_WAVE_FPS, self._animate)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Hit testing
    # ------------------------------------------------------------------ #

    def _hit_test(self, ex: float, ey: float):
        """Return 'listener', speaker sid (int), or None."""
        lpp = self._proj(self._lx, self._ly, self._lz)
        if lpp and math.hypot(ex - lpp[0], ey - lpp[1]) <= self.HIT_R:
            return "listener"
        # Check speakers back-to-front (nearest = shallowest depth)
        hits = []
        for spk in self._speakers:
            pp = self._proj(spk.x, spk.y, spk.z)
            if pp:
                d = math.hypot(ex - pp[0], ey - pp[1])
                if d <= self.HIT_R:
                    hits.append((pp[2], spk.sid))
        if hits:
            hits.sort()
            return hits[0][1]
        return None

    def _screen_to_floor(self, sx: float, sy: float) -> Optional[tuple]:
        """Unproject a screen point onto the Y=0 (floor) plane."""
        (cx, cy, cz), right, up, fwd = self._cam._basis()
        ft = math.tan(math.radians(self._cam.fov / 2))
        scale = (self._ch / 2) / ft

        ray_cx = (sx - self._cw / 2) / scale
        ray_cy = -(sy - self._ch / 2) / scale

        # Ray in world: start at cam, direction = fwd + ray_cx*right + ray_cy*up
        rdx = fwd[0] + ray_cx * right[0] + ray_cy * up[0]
        rdy = fwd[1] + ray_cx * right[1] + ray_cy * up[1]
        rdz = fwd[2] + ray_cx * right[2] + ray_cy * up[2]

        if abs(rdy) < 1e-6:
            return None
        t = -cy / rdy
        if t < 0:
            return None
        return cx + rdx * t, cz + rdz * t  # world X, Z on floor

    def _screen_to_height_plane(self, sx: float, sy: float,
                                spk_x: float, spk_z: float) -> Optional[float]:
        """Unproject screen point onto the vertical plane X=spk_x (for height drag)."""
        (cx, cy, cz), right, up, fwd = self._cam._basis()
        ft = math.tan(math.radians(self._cam.fov / 2))
        scale = (self._ch / 2) / ft

        ray_cx = (sx - self._cw / 2) / scale
        ray_cy = -(sy - self._ch / 2) / scale

        rdx = fwd[0] + ray_cx * right[0] + ray_cy * up[0]
        rdy = fwd[1] + ray_cx * right[1] + ray_cy * up[1]
        rdz = fwd[2] + ray_cx * right[2] + ray_cy * up[2]

        if abs(rdy) < 1e-6:
            return None
        t = (1.2 - cy) / rdy + (sy - self._ch / 2) / scale  # rough
        # Simpler: project screen Y change to world Y
        eye_up = up[1]
        if abs(eye_up) < 0.01:
            return None
        return None  # fallback: use shift-drag delta

    # ------------------------------------------------------------------ #
    # Mouse handlers
    # ------------------------------------------------------------------ #

    def _on_press(self, ev):
        self._drag_x0 = ev.x
        self._drag_y0 = ev.y
        self._cam_az0 = self._cam.az
        self._cam_el0 = self._cam.el

        target = self._hit_test(ev.x, ev.y)

        if target == "listener":
            self._drag_target = "listener"
            self._selected_sid = None

        elif isinstance(target, int):
            self._drag_target = target
            spk = self._find(target)
            if spk:
                self._drag_spk_x0 = spk.x
                self._drag_spk_z0 = spk.z
            prev = self._selected_sid
            self._selected_sid = target
            if prev != target and self._on_speaker_selected:
                self._on_speaker_selected(target)

        else:
            self._drag_target = "camera"
            self._selected_sid = None

        self._draw_all()

    def _on_drag(self, ev):
        dx = ev.x - self._drag_x0
        dy = ev.y - self._drag_y0

        if self._drag_target == "camera":
            self._cam.az = (self._cam_az0 - dx * 0.45) % 360.0
            self._cam.el = max(-80.0, min(80.0, self._cam_el0 + dy * 0.35))
            self._draw_all()

        elif self._drag_target == "listener":
            floor = self._screen_to_floor(ev.x, ev.y)
            if floor:
                nx, nz = floor
                hw, hd = self._room_w/2, self._room_d/2
                self._lx = max(-hw+0.3, min(hw-0.3, nx))
                self._lz = max(-hd+0.3, min(hd-0.3, nz))
                if self._on_listener_moved:
                    self._on_listener_moved(self._lx, self._lz)
            self._draw_all()

        elif isinstance(self._drag_target, int):
            spk = self._find(self._drag_target)
            if spk:
                floor = self._screen_to_floor(ev.x, ev.y)
                if floor:
                    nx, nz = floor
                    spk.x, _, spk.z = self._clamp_to_room(nx, spk.y, nz)
                    if self._on_speaker_moved:
                        az   = self._xyz_to_azimuth(spk.x, spk.z)
                        dist = math.sqrt((spk.x-self._lx)**2 +
                                         (spk.y-self._ly)**2 +
                                         (spk.z-self._lz)**2)
                        self._on_speaker_moved(spk.sid, az, dist)
                self._draw_all()

    def _on_shift_drag(self, ev):
        """Shift+drag on selected speaker: adjust height."""
        if not isinstance(self._drag_target, int):
            return
        spk = self._find(self._drag_target)
        if not spk:
            return
        dy = ev.y - self._drag_y0
        # Map vertical screen movement to world Y (roughly)
        new_y = max(0.1, min(self._room_h - 0.1,
                              spk.y - dy * (self._room_h / self._ch) * 1.5))
        spk.y = new_y
        self._drag_y0 = ev.y
        if self._on_speaker_moved:
            az   = self._xyz_to_azimuth(spk.x, spk.z)
            dist = math.sqrt((spk.x-self._lx)**2 +
                              (spk.y-self._ly)**2 +
                              (spk.z-self._lz)**2)
            self._on_speaker_moved(spk.sid, az, dist)
        self._draw_all()

    def _on_release(self, ev):
        self._drag_target = None

    def _on_scroll(self, ev):
        delta = getattr(ev, "delta", 0)
        if delta == 0:
            delta = -120 if ev.num == 5 else 120
        self._cam.dist = max(2.0, min(25.0, self._cam.dist - delta * 0.008))
        self._draw_all()

    def _on_right_click(self, ev):
        target = self._hit_test(ev.x, ev.y)
        if isinstance(target, int):
            self._selected_sid = target
            self._draw_all()
            if self._on_speaker_selected:
                self._on_speaker_selected(target)
            self._show_context_menu(ev, target)

    def _on_resize(self, ev):
        self._cw = ev.width
        self._ch = ev.height
        self._draw_all()

    # ------------------------------------------------------------------ #
    # Context menu
    # ------------------------------------------------------------------ #

    def _show_context_menu(self, ev, sid: int):
        spk = self._find(sid)
        if not spk:
            return
        m = tk.Menu(self._cvs, tearoff=0,
                    bg="#1c2333", fg="#e8edf3",
                    activebackground="#4361ee", activeforeground="white",
                    bd=0, relief=tk.FLAT, font=("Segoe UI", 10))
        m.add_command(label=f"Speaker: {spk.label}  az={spk.face_az:.0f}° el={spk.face_el:.0f}°",
                      state="disabled")
        m.add_separator()
        # Cardinal orientation presets
        for face_az, face_el, label in [
            (0,   0, "▲  Face front"),
            (90,  0, "▶  Face right"),
            (180, 0, "▼  Face back"),
            (270, 0, "◀  Face left"),
            (0,  90, "⬆  Face up (ceiling)"),
            (0, -90, "⬇  Face down (floor)"),
        ]:
            m.add_command(label=label,
                          command=lambda a=face_az, e=face_el, s=sid:
                          self._set_facing(s, a, e))
        m.add_separator()
        m.add_command(label="⊙  Face listener",
                      command=lambda s=sid: self._face_listener(s))
        m.add_command(label="⊗  Face away from listener",
                      command=lambda s=sid: self._face_away(s))
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()

    def _set_facing(self, sid: int, az: float, el: float):
        spk = self._find(sid)
        if spk:
            spk.face_az = float(az) % 360.0
            spk.face_el = max(-90.0, min(90.0, float(el)))
            if self._on_speaker_rotated:
                self._on_speaker_rotated(sid, spk.face_az, spk.face_el)
        self._draw_all()

    def _face_listener(self, sid: int):
        spk = self._find(sid)
        if spk:
            dx = self._lx - spk.x
            dy = self._ly - spk.y
            dz = self._lz - spk.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            spk.face_az = math.degrees(math.atan2(dx, -dz)) % 360.0
            spk.face_el = math.degrees(math.asin(dy / dist)) if dist > 0 else 0.0
            if self._on_speaker_rotated:
                self._on_speaker_rotated(sid, spk.face_az, spk.face_el)
        self._draw_all()

    def _face_away(self, sid: int):
        spk = self._find(sid)
        if spk:
            dx = self._lx - spk.x
            dy = self._ly - spk.y
            dz = self._lz - spk.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            spk.face_az = (math.degrees(math.atan2(dx, -dz)) + 180.0) % 360.0
            spk.face_el = -math.degrees(math.asin(dy / dist)) if dist > 0 else 0.0
            if self._on_speaker_rotated:
                self._on_speaker_rotated(sid, spk.face_az, spk.face_el)
        self._draw_all()

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #

    def _find(self, sid: int) -> Optional[Speaker3D]:
        return next((s for s in self._speakers if s.sid == sid), None)

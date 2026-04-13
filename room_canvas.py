"""
room_canvas.py — Interactive 2D top-down room visualisation widget.

Shows a listener (red dot) and N speakers (colour-coded blue dots) in a
room.  Speakers and the listener are draggable.  Right-clicking a speaker
opens an orientation context menu.  An animation loop draws expanding sound-
wave rings from active speakers proportional to their current audio level.

Coordinate conventions
----------------------
Canvas: origin top-left, y increases downward (standard tkinter).
Room:   0° = front (top of canvas), 90° = right, 180° = behind,
        −90° / 270° = left.  This matches the cinema/ITU azimuth convention
        used throughout ModAudio's DSP chain.

Azimuth formula (canvas → azimuth):
    dx = cx − lx,  dy = cy − ly  (ly up = negative dy)
    azimuth = atan2(dx, −dy)  [degrees, CCW from north]

Speaker orientation (angle_deg):
    The angle_deg attribute uses the same 0°=up convention.
    The orientation arrow is drawn in that direction from the speaker centre.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_ROOM_BG       = "#0d1117"
_ROOM_FLOOR    = "#111823"
_WALL_COLOR    = "#2a3550"
_GRID_COLOR    = "#161b27"
_LISTENER_COL  = "#ef476f"
_LISTENER_RIM  = "#ff8080"

# 8 distinct speaker colours (cycles if more speakers)
_SPEAKER_COLS = [
    "#4361ee",  # blue
    "#7209b7",  # purple
    "#3a86ff",  # sky-blue
    "#06d6a0",  # teal
    "#ffd166",  # amber
    "#f72585",  # magenta
    "#4cc9f0",  # cyan
    "#ff6b6b",  # coral
]

# ---------------------------------------------------------------------------
# Preset speaker layouts  (label, azimuth_deg, distance_m)
# ---------------------------------------------------------------------------

SPEAKER_LAYOUTS: dict[str, list[tuple[str, float, float]]] = {
    "2.0 Stereo": [
        ("FL", -30.0, 2.0),
        ("FR",  30.0, 2.0),
    ],
    "5.1 Cinema": [
        ("C",    0.0, 2.5),
        ("FL", -30.0, 2.5),
        ("FR",  30.0, 2.5),
        ("SL", -110.0, 2.0),
        ("SR",  110.0, 2.0),
    ],
    "7.1 IMAX": [
        ("C",    0.0, 2.5),
        ("FL", -30.0, 2.5),
        ("FR",  30.0, 2.5),
        ("SL", -90.0, 2.0),
        ("SR",  90.0, 2.0),
        ("BL", -150.0, 1.8),
        ("BR",  150.0, 1.8),
    ],
    "7.1 Dolby": [
        ("C",    0.0, 2.5),
        ("FL", -30.0, 2.5),
        ("FR",  30.0, 2.5),
        ("SL", -110.0, 2.0),
        ("SR",  110.0, 2.0),
        ("BL", -150.0, 1.8),
        ("BR",  150.0, 1.8),
    ],
    "Home 5.1": [
        ("C",    0.0, 1.5),
        ("FL", -30.0, 1.5),
        ("FR",  30.0, 1.5),
        ("SL", -110.0, 1.2),
        ("SR",  110.0, 1.2),
    ],
}

# ---------------------------------------------------------------------------
# Speaker data model
# ---------------------------------------------------------------------------

@dataclass
class Speaker:
    """State of one physical speaker in the room."""
    sid: int                         # unique ID
    label: str                       # display label (e.g. "FL", "SR")
    x: float                         # canvas x position (pixels)
    y: float                         # canvas y position (pixels)
    angle_deg: float = 0.0           # facing direction (0=up/front, 90=right)
    device_idx: Optional[int] = None # assigned sounddevice output index
    device_label: str = "Unassigned" # short display name for assigned device
    level_rms: float = 0.0           # current RMS level 0–1 (from audio thread)
    active: bool = False             # True while audio is playing

# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class RoomCanvas:
    """
    Interactive 2D room top-down view.

    Parameters
    ----------
    parent             : tkinter parent widget
    canvas_width       : initial canvas width in pixels
    canvas_height      : initial canvas height in pixels
    room_width_m       : physical room width in metres (horizontal)
    room_depth_m       : physical room depth in metres (vertical / front-back)
    on_speaker_moved   : callback(speaker_id, azimuth_deg, distance_m)
    on_listener_moved  : callback(canvas_x, canvas_y)
    on_speaker_selected: callback(speaker_id)
    on_change          : callback() — fired on any structural change
    """

    # Layout constants
    ROOM_PAD   = 32     # px padding between canvas edge and room wall
    L_RAD      = 10     # listener dot radius
    S_RAD      = 9      # speaker dot radius
    ARROW_LEN  = 20     # orientation arrow length
    HIT_PAD    = 7      # extra hit-test radius beyond visual radius
    WAVE_SPD   = 75.0   # sound-wave ring expansion speed (px/s)
    WAVE_MAX_R = 130.0  # max ring radius before fade-out
    WAVE_FPS   = 33     # animation frames per second (~30 ms)
    WAVE_IVMS  = 280    # ms between new wave emissions per speaker

    def __init__(
        self,
        parent,
        canvas_width: int  = 620,
        canvas_height: int = 340,
        room_width_m: float  = 6.0,
        room_depth_m: float  = 5.0,
        on_speaker_moved:    Optional[Callable] = None,
        on_listener_moved:   Optional[Callable] = None,
        on_speaker_selected: Optional[Callable] = None,
        on_change:           Optional[Callable] = None,
    ):
        self._cw = canvas_width
        self._ch = canvas_height
        self._room_w = float(room_width_m)
        self._room_d = float(room_depth_m)

        self._on_speaker_moved    = on_speaker_moved
        self._on_listener_moved   = on_listener_moved
        self._on_speaker_selected = on_speaker_selected
        self._on_change           = on_change

        # Create canvas
        self._cvs = tk.Canvas(
            parent,
            width=canvas_width,
            height=canvas_height,
            bg=_ROOM_BG,
            highlightthickness=1,
            highlightbackground=_WALL_COLOR,
            cursor="crosshair",
        )

        # Listener starts at centre of room, slightly toward front
        self._lx = canvas_width  / 2.0
        self._ly = canvas_height * 0.58

        # Speaker list and ID counter
        self._speakers: list[Speaker] = []
        self._next_sid = 0
        self._selected_sid: Optional[int] = None

        # Drag state
        self._drag_target = None   # "listener" | int (speaker id)
        self._drag_ox = 0.0
        self._drag_oy = 0.0

        # Sound-wave animation state
        self._waves: list[dict] = []
        self._last_emit: dict[int, float] = {}   # sid → time of last wave
        self._anim_running = False

        # Bindings
        self._cvs.bind("<ButtonPress-1>",   self._on_press)
        self._cvs.bind("<B1-Motion>",       self._on_drag)
        self._cvs.bind("<ButtonRelease-1>", self._on_release)
        self._cvs.bind("<ButtonPress-3>",   self._on_right_click)
        self._cvs.bind("<Configure>",       self._on_resize)

        self._draw_all()

    # ------------------------------------------------------------------ #
    # Public API — widget packing
    # ------------------------------------------------------------------ #

    @property
    def widget(self) -> tk.Canvas:
        return self._cvs

    def pack(self, **kw):
        self._cvs.pack(**kw)

    def grid(self, **kw):
        self._cvs.grid(**kw)

    # ------------------------------------------------------------------ #
    # Public API — speaker management
    # ------------------------------------------------------------------ #

    def add_speaker(
        self,
        label: str,
        azimuth_deg: float,
        distance_m: float = 2.0,
        device_idx: Optional[int] = None,
        device_label: str = "Unassigned",
    ) -> int:
        """Place a new speaker and return its ID."""
        x, y = self._az_dist_to_canvas(azimuth_deg, distance_m)
        # Default facing: toward listener
        face = (azimuth_deg + 180.0) % 360.0
        spk = Speaker(
            sid=self._next_sid,
            label=label,
            x=x, y=y,
            angle_deg=face,
            device_idx=device_idx,
            device_label=device_label,
        )
        self._speakers.append(spk)
        sid = self._next_sid
        self._next_sid += 1
        self._draw_all()
        if self._on_change:
            self._on_change()
        return sid

    def remove_speaker(self, sid: int):
        """Remove a speaker by ID."""
        self._speakers = [s for s in self._speakers if s.sid != sid]
        if self._selected_sid == sid:
            self._selected_sid = None
        self._draw_all()
        if self._on_change:
            self._on_change()

    def clear_speakers(self):
        self._speakers.clear()
        self._selected_sid = None
        self._next_sid = 0
        self._draw_all()
        if self._on_change:
            self._on_change()

    def load_layout(self, layout_name: str) -> list[int]:
        """
        Clear speakers and load a named preset layout.
        Returns list of new speaker IDs in the same order as the layout.
        """
        self.clear_speakers()
        layout = SPEAKER_LAYOUTS.get(layout_name, [])
        ids = []
        for label, az, dist in layout:
            ids.append(self.add_speaker(label, az, dist))
        return ids

    def get_speakers(self) -> list[Speaker]:
        return list(self._speakers)

    def get_selected_sid(self) -> Optional[int]:
        return self._selected_sid

    def set_selected_sid(self, sid: Optional[int]):
        self._selected_sid = sid
        self._draw_all()

    # ------------------------------------------------------------------ #
    # Public API — speaker properties (callable from outside after select)
    # ------------------------------------------------------------------ #

    def set_speaker_device(self, sid: int, device_idx: Optional[int],
                           device_label: str):
        for s in self._speakers:
            if s.sid == sid:
                s.device_idx    = device_idx
                s.device_label  = device_label
                break
        self._draw_all()

    def set_speaker_label(self, sid: int, label: str):
        for s in self._speakers:
            if s.sid == sid:
                s.label = label
                break
        self._draw_all()

    def set_speaker_stereo_level(self, sid: int, left_rms: float, right_rms: float):
        """Feed real-time per-channel levels (0–1); the 2D canvas uses their average."""
        self.set_speaker_level(sid, (float(left_rms) + float(right_rms)) * 0.5)

    def set_speaker_level(self, sid: int, level_rms: float):
        """Update real-time audio level (0–1); feeds the wave animation."""
        for s in self._speakers:
            if s.sid == sid:
                s.level_rms = float(level_rms)
                s.active    = level_rms > 0.004
                break

    # ------------------------------------------------------------------ #
    # Public API — room & geometry
    # ------------------------------------------------------------------ #

    def set_room_size(self, width_m: float, depth_m: float):
        self._room_w = max(1.0, float(width_m))
        self._room_d = max(1.0, float(depth_m))
        self._draw_all()

    def get_room_size(self) -> tuple[float, float]:
        return self._room_w, self._room_d

    def get_speaker_azimuths(self) -> list[float]:
        return [self._canvas_to_azimuth(s.x, s.y) for s in self._speakers]

    def get_speaker_distances_m(self) -> list[float]:
        return [self._canvas_to_distance_m(s.x, s.y) for s in self._speakers]

    # ------------------------------------------------------------------ #
    # Public API — animation
    # ------------------------------------------------------------------ #

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
    # Coordinate helpers
    # ------------------------------------------------------------------ #

    def _room_rect(self) -> tuple[int, int, int, int]:
        p = self.ROOM_PAD
        return p, p, self._cw - p, self._ch - p

    def _ppm(self) -> float:
        """Pixels per metre (based on smaller of width/height ratio)."""
        x1, y1, x2, y2 = self._room_rect()
        return min((x2 - x1) / self._room_w, (y2 - y1) / self._room_d)

    def _room_centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self._room_rect()
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _az_dist_to_canvas(self, az_deg: float, dist_m: float) -> tuple[float, float]:
        """Azimuth + distance → canvas coords relative to current listener."""
        r = math.radians(az_deg)
        ppm = self._ppm()
        return (
            self._lx + math.sin(r) * dist_m * ppm,
            self._ly - math.cos(r) * dist_m * ppm,
        )

    def _canvas_to_azimuth(self, cx: float, cy: float) -> float:
        """Canvas coords → azimuth degrees (0=front/up, 90=right)."""
        dx = cx - self._lx
        dy = cy - self._ly
        return math.degrees(math.atan2(dx, -dy)) % 360.0

    def _canvas_to_distance_m(self, cx: float, cy: float) -> float:
        ppm = self._ppm()
        if ppm < 1e-6:
            return 0.0
        return math.hypot(cx - self._lx, cy - self._ly) / ppm

    def _clamp_to_room(self, x: float, y: float) -> tuple[float, float]:
        x1, y1, x2, y2 = self._room_rect()
        m = self.S_RAD + 2
        return (
            max(x1 + m, min(x2 - m, x)),
            max(y1 + m, min(y2 - m, y)),
        )

    # ------------------------------------------------------------------ #
    # Colour blending
    # ------------------------------------------------------------------ #

    @staticmethod
    def _blend(fg: str, bg: str, alpha: float) -> str:
        """Alpha-blend two hex colours."""
        def _p(h: str):
            h = h.lstrip("#")
            return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        try:
            fr, fg2, fb = _p(fg)
            br, bg2, bb = _p(bg)
            r = int(fr * alpha + br * (1 - alpha))
            g = int(fg2 * alpha + bg2 * (1 - alpha))
            b = int(fb * alpha + bb * (1 - alpha))
            return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"
        except Exception:
            return fg

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def _draw_all(self):
        c = self._cvs
        c.delete("all")
        self._draw_floor()
        self._draw_grid()
        self._draw_walls()
        self._draw_sound_waves()
        self._draw_speakers()
        self._draw_listener()
        self._draw_hud()

    def _draw_floor(self):
        x1, y1, x2, y2 = self._room_rect()
        self._cvs.create_rectangle(x1, y1, x2, y2, fill=_ROOM_FLOOR, outline="")

    def _draw_grid(self):
        """Faint 1-metre grid lines."""
        c = self._cvs
        x1, y1, x2, y2 = self._room_rect()
        ppm = self._ppm()
        lx, ly = self._lx, self._ly

        # Vertical (columns)
        for i in range(-20, 21):
            gx = lx + i * ppm
            if x1 <= gx <= x2:
                c.create_line(gx, y1, gx, y2, fill=_GRID_COLOR, width=1)
        # Horizontal (rows)
        for i in range(-20, 21):
            gy = ly + i * ppm
            if y1 <= gy <= y2:
                c.create_line(x1, gy, x2, gy, fill=_GRID_COLOR, width=1)

        # Axis cross through listener (slightly brighter)
        axis_col = "#1c2840"
        c.create_line(lx, y1, lx, y2, fill=axis_col, width=1)
        c.create_line(x1, ly, x2, ly, fill=axis_col, width=1)

    def _draw_walls(self):
        c   = self._cvs
        x1, y1, x2, y2 = self._room_rect()

        # Shadow
        c.create_rectangle(x1 - 1, y1 - 1, x2 + 1, y2 + 1,
                           fill="", outline="#1c2840", width=2)
        # Wall
        c.create_rectangle(x1, y1, x2, y2, fill="", outline=_WALL_COLOR, width=2)
        # Corner ticks
        sz = 8
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            sx = 1 if cx == x1 else -1
            sy = 1 if cy == y1 else -1
            c.create_line(cx, cy, cx + sx * sz, cy, fill="#3a4a6a", width=2)
            c.create_line(cx, cy, cx, cy + sy * sz, fill="#3a4a6a", width=2)

        # Room-size label
        c.create_text((x1 + x2) / 2, y2 + 14,
                      text=f"{self._room_w:.1f} m × {self._room_d:.1f} m",
                      fill="#2d3d5a", font=("Consolas", 8), anchor="center")

    def _draw_sound_waves(self):
        c = self._cvs
        for w in self._waves:
            r = w["radius"]
            alpha = w["alpha"]
            x, y  = w["x"], w["y"]
            col   = self._blend(w["color"], _ROOM_FLOOR, alpha)
            thick = 1 if alpha < 0.4 else 2
            c.create_oval(x - r, y - r, x + r, y + r,
                          outline=col, fill="", width=thick)

    def _draw_speakers(self):
        c = self._cvs
        for i, spk in enumerate(self._speakers):
            col = _SPEAKER_COLS[i % len(_SPEAKER_COLS)]
            sr  = self.S_RAD
            sel = (spk.sid == self._selected_sid)

            # Glow when active
            if spk.active and spk.level_rms > 0.01:
                gr  = sr + max(2, int(spk.level_rms * 12))
                gc  = self._blend(col, _ROOM_FLOOR, spk.level_rms * 0.28)
                c.create_oval(spk.x - gr, spk.y - gr,
                              spk.x + gr, spk.y + gr, fill=gc, outline="")

            # Selection ring
            if sel:
                c.create_oval(spk.x - sr - 5, spk.y - sr - 5,
                              spk.x + sr + 5, spk.y + sr + 5,
                              outline="white", fill="", width=1)

            # Orientation arrow
            ang_rad = math.radians(spk.angle_deg)
            adx = math.sin(ang_rad) * self.ARROW_LEN
            ady = -math.cos(ang_rad) * self.ARROW_LEN
            arrow_col = self._blend(col, _ROOM_FLOOR, 0.60)
            c.create_line(spk.x, spk.y,
                          spk.x + adx, spk.y + ady,
                          fill=arrow_col, width=2,
                          arrow=tk.LAST, arrowshape=(7, 9, 3))

            # Body
            body_fill = (self._blend(col, "#ffffff", 0.18)
                         if spk.active else col)
            rim_col   = "white" if sel else self._blend(col, "#ffffff", 0.45)
            c.create_oval(spk.x - sr, spk.y - sr,
                          spk.x + sr, spk.y + sr,
                          fill=body_fill,
                          outline=rim_col,
                          width=2 if sel else 1)

            # Label text
            c.create_text(spk.x, spk.y, text=spk.label,
                          fill="white", font=("Consolas", 7, "bold"),
                          anchor="center")

            # Device assignment (below dot)
            dev_short = (spk.device_label[:16] + "…"
                         if len(spk.device_label) > 17
                         else spk.device_label)
            dev_col = self._blend(col, _ROOM_FLOOR, 0.75)
            c.create_text(spk.x, spk.y + sr + 9, text=dev_short,
                          fill=dev_col, font=("Consolas", 6), anchor="center")

    def _draw_listener(self):
        c = self._cvs
        x, y = self._lx, self._ly
        lr   = self.L_RAD

        # Outer pulse ring
        c.create_oval(x - lr - 6, y - lr - 6,
                      x + lr + 6, y + lr + 6,
                      fill="", outline=self._blend(_LISTENER_COL, _ROOM_BG, 0.20),
                      width=1)
        # Dot
        c.create_oval(x - lr, y - lr, x + lr, y + lr,
                      fill=_LISTENER_COL, outline=_LISTENER_RIM, width=1)
        # Forward-facing indicator triangle
        pts = [x, y - lr - 6, x - 4, y - lr, x + 4, y - lr]
        c.create_polygon(pts, fill=_LISTENER_COL, outline="")
        # Label
        c.create_text(x + lr + 6, y, text="YOU",
                      fill=_LISTENER_COL,
                      font=("Consolas", 7, "bold"), anchor="w")

    def _draw_hud(self):
        """Compass rose + scale bar."""
        c = self._cvs
        x1, y1, x2, y2 = self._room_rect()

        # Compass (top-right inside room)
        cx2, cy2 = x2 - 16, y1 + 26
        c.create_text(cx2, cy2 - 12, text="▲ FRONT",
                      fill="#2a3a58", font=("Consolas", 7), anchor="center")

        # Scale bar (bottom-left)
        ppm = self._ppm()
        bx, by = x1 + 8, y2 - 10
        bl = int(ppm)
        c.create_line(bx, by, bx + bl, by, fill="#2a3a58", width=2)
        c.create_line(bx,      by - 3, bx,      by + 3, fill="#2a3a58", width=2)
        c.create_line(bx + bl, by - 3, bx + bl, by + 3, fill="#2a3a58", width=2)
        c.create_text(bx + bl // 2, by - 6, text="1 m",
                      fill="#2a3a58", font=("Consolas", 7), anchor="center")

    # ------------------------------------------------------------------ #
    # Sound-wave animation
    # ------------------------------------------------------------------ #

    def _animate(self):
        if not self._anim_running:
            return

        dt  = self.WAVE_FPS / 1000.0
        now = time.time()

        # Emit new rings from active speakers
        iv = self.WAVE_IVMS / 1000.0
        for i, spk in enumerate(self._speakers):
            if not spk.active:
                continue
            last = self._last_emit.get(spk.sid, 0.0)
            if now - last >= iv:
                self._last_emit[spk.sid] = now
                col = _SPEAKER_COLS[i % len(_SPEAKER_COLS)]
                self._waves.append({
                    "sid":    spk.sid,
                    "x":      spk.x,
                    "y":      spk.y,
                    "radius": float(self.S_RAD + 2),
                    "alpha":  min(1.0, spk.level_rms * 2.2) * 0.7,
                    "color":  col,
                })

        # Update existing rings
        keep = []
        decay = 0.7 / (self.WAVE_MAX_R / self.WAVE_SPD)  # alpha / s
        for w in self._waves:
            w["radius"] += self.WAVE_SPD * dt
            w["alpha"]  -= decay * dt
            if w["alpha"] > 0.01 and w["radius"] < self.WAVE_MAX_R:
                keep.append(w)
        self._waves = keep

        self._draw_all()

        try:
            self._cvs.after(self.WAVE_FPS, self._animate)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Hit testing
    # ------------------------------------------------------------------ #

    def _hit(self, x: float, y: float):
        """Return 'listener', a speaker sid (int), or None."""
        if math.hypot(x - self._lx, y - self._ly) <= self.L_RAD + self.HIT_PAD:
            return "listener"
        # Last drawn = topmost → test in reverse
        for spk in reversed(self._speakers):
            if math.hypot(x - spk.x, y - spk.y) <= self.S_RAD + self.HIT_PAD:
                return spk.sid
        return None

    # ------------------------------------------------------------------ #
    # Mouse interaction
    # ------------------------------------------------------------------ #

    def _on_press(self, ev):
        target = self._hit(ev.x, ev.y)

        if target == "listener":
            self._drag_target = "listener"
            self._drag_ox = self._lx - ev.x
            self._drag_oy = self._ly - ev.y
            self._selected_sid = None

        elif isinstance(target, int):
            self._drag_target = target
            spk = self._find(target)
            if spk:
                self._drag_ox = spk.x - ev.x
                self._drag_oy = spk.y - ev.y
            prev = self._selected_sid
            self._selected_sid = target
            if prev != target and self._on_speaker_selected:
                self._on_speaker_selected(target)

        else:
            self._drag_target = None
            self._selected_sid = None

        self._draw_all()

    def _on_drag(self, ev):
        if self._drag_target is None:
            return
        nx, ny = (ev.x + self._drag_ox, ev.y + self._drag_oy)
        nx, ny = self._clamp_to_room(nx, ny)

        if self._drag_target == "listener":
            self._lx, self._ly = nx, ny
            if self._on_listener_moved:
                self._on_listener_moved(nx, ny)
        else:
            spk = self._find(self._drag_target)
            if spk:
                spk.x, spk.y = nx, ny

        self._draw_all()

    def _on_release(self, ev):
        if self._drag_target is not None and self._drag_target != "listener":
            spk = self._find(self._drag_target)
            if spk and self._on_speaker_moved:
                az   = self._canvas_to_azimuth(spk.x, spk.y)
                dist = self._canvas_to_distance_m(spk.x, spk.y)
                self._on_speaker_moved(spk.sid, az, dist)
        self._drag_target = None

    def _on_right_click(self, ev):
        target = self._hit(ev.x, ev.y)
        if isinstance(target, int):
            self._selected_sid = target
            self._draw_all()
            if self._on_speaker_selected:
                self._on_speaker_selected(target)
            self._show_context_menu(ev, target)

    def _show_context_menu(self, ev, sid: int):
        spk = self._find(sid)
        if spk is None:
            return
        m = tk.Menu(self._cvs, tearoff=0,
                    bg="#1c2333", fg="#e8edf3",
                    activebackground="#4361ee", activeforeground="white",
                    bd=0, relief=tk.FLAT, font=("Segoe UI", 10))
        m.add_command(
            label=f"Speaker: {spk.label}  |  {spk.angle_deg:.0f}° facing",
            state="disabled")
        m.add_separator()

        presets = [
            (0,   "▲  Face front (0°)"),
            (90,  "▶  Face right (90°)"),
            (180, "▼  Face back (180°)"),
            (270, "◀  Face left (270°)"),
        ]
        for ang, lbl in presets:
            m.add_command(label=lbl,
                          command=lambda a=ang, s=sid: self._set_angle(s, a))
        m.add_separator()
        m.add_command(label="⊙  Face listener",
                      command=lambda s=sid: self._face_listener(s))
        m.add_command(label="⊗  Face away from listener",
                      command=lambda s=sid: self._face_away(s))
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()

    # ------------------------------------------------------------------ #
    # Orientation helpers
    # ------------------------------------------------------------------ #

    def _set_angle(self, sid: int, angle: float):
        spk = self._find(sid)
        if spk:
            spk.angle_deg = angle % 360.0
        self._draw_all()

    def _face_listener(self, sid: int):
        spk = self._find(sid)
        if spk:
            dx = self._lx - spk.x
            dy = self._ly - spk.y
            spk.angle_deg = math.degrees(math.atan2(dx, -dy)) % 360.0
        self._draw_all()

    def _face_away(self, sid: int):
        spk = self._find(sid)
        if spk:
            dx = self._lx - spk.x
            dy = self._ly - spk.y
            spk.angle_deg = (math.degrees(math.atan2(dx, -dy)) + 180.0) % 360.0
        self._draw_all()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _find(self, sid: int) -> Optional[Speaker]:
        return next((s for s in self._speakers if s.sid == sid), None)

    def _on_resize(self, ev):
        self._cw = ev.width
        self._ch = ev.height
        self._lx = ev.width  / 2.0
        self._ly = ev.height * 0.58
        self._draw_all()

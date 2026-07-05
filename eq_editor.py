"""
EQ Editor Dialog — interactive parametric EQ with reference curve overlay.

Opened from the Theater tab (single global EQ) or the Multi-Speaker tab
(per-speaker EQ with a speaker selector dropdown).

Canvas layout
-------------
  X axis : 20 Hz – 20 kHz (logarithmic)
  Y axis : -15 dB – +15 dB (linear)

Interaction
-----------
  Left-drag band node       : freq (left/right) + gain (up/down)
  Mouse-wheel over node      : Q factor
  Right-click node           : context menu (reset band, set gain to 0)
  Double-click canvas (empty): no action
  Esc / close button         : close (changes already applied live)
"""

from __future__ import annotations
import json
import math
import os
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np

try:
    import customtkinter as ctk
except ImportError:
    ctk = None  # type: ignore

from dsp.custom_eq        import ParametricEQ, DEFAULT_BANDS
from dsp.speaker_profiles import get_correction_bands
from dsp.device_analyzer  import DeviceAnalyzer


# ---------------------------------------------------------------------------
# Reference curves
# All curves are lists of (freq_hz, gain_db) points; interpolated on log scale.
# ---------------------------------------------------------------------------

REFERENCE_CURVES: dict[str, list] = {
    "Flat (0 dB)": [
        (20, 0), (20000, 0),
    ],
    "X-curve / ISO 2969": [
        (20, 0), (1000, 0), (2000, 0), (3150, -0.5),
        (5000, -2.0), (8000, -4.5), (12500, -7.0),
        (16000, -9.0), (20000, -11.0),
    ],
    "Harman Headphone (2018)": [
        (20, 4), (60, 4), (120, 2.5), (200, 1.0),
        (500, -1.0), (1000, -2.5), (2000, -3.0),
        (3000, -0.5), (4000, 2.5), (6000, 4.0),
        (8000, 2.5), (10000, 0.5), (12000, -1.5),
        (14000, -3.5), (16000, -5.5), (20000, -8.0),
    ],
    "Harman In-Room Speaker": [
        (20, 0), (100, 0), (200, -0.5), (500, -1.0),
        (1000, -1.5), (2000, -2.5), (4000, -3.5),
        (8000, -5.5), (16000, -8.0), (20000, -10.0),
    ],
    "V-Shape": [
        (20, 3), (60, 5), (100, 4), (200, 2),
        (500, 0), (1000, -2), (2000, -2),
        (4000, 0), (8000, 2), (12000, 4), (20000, 4),
    ],
    "Bass Boost": [
        (20, 1), (60, 4), (100, 5), (200, 3),
        (500, 1), (1000, 0), (5000, 0), (20000, 0),
    ],
    "Vocal Clarity": [
        (20, 0), (200, -1), (500, -1.5), (1000, 0),
        (2500, 3.5), (4000, 3), (8000, 1), (20000, 0),
    ],
}

# Per-reference-curve colors (dark-mode friendly)
CURVE_COLORS: dict[str, str] = {
    "Flat (0 dB)":            "#3a3a44",
    "X-curve / ISO 2969":     "#4a6fa5",
    "Harman Headphone (2018)":"#6a9e6f",
    "Harman In-Room Speaker": "#9e6a6a",
    "V-Shape":                "#9e8e4a",
    "Bass Boost":             "#7a4a9e",
    "Vocal Clarity":          "#4a9e8e",
}


# ---------------------------------------------------------------------------
# Band node appearance
# ---------------------------------------------------------------------------

_NODE_COLORS: dict[str, str] = {
    "highpass":  "#a56aff",
    "lowpass":   "#a56aff",
    "lowshelf":  "#ffa040",
    "highshelf": "#ffa040",
    "peaking":   "#5ad3e0",
}
_NODE_R = 9          # node radius (pixels)
_SEL_R  = 12         # selected node radius


# ---------------------------------------------------------------------------
# EQEditorDialog
# ---------------------------------------------------------------------------

class EQEditorDialog:
    """
    Parametric EQ editor.  Opens as a Toplevel window.

    Parameters
    ----------
    parent          : tk/ctk parent widget
    eq              : ParametricEQ for the "current" speaker/channel
    mode            : "theater" | "multi_speaker"
    speaker_list    : [{"sid": int, "label": str, "device_name": str, "eq": ParametricEQ}, ...]
                      (multi_speaker mode only)
    output_device_name : name of the current output device (theater mode)
    on_close        : optional callback()
    fs              : sample rate (for frequency response calc)
    eq_state_path   : path to JSON file for persistence
    """

    FREQ_MIN = 20.0
    FREQ_MAX = 20000.0
    GAIN_MIN = -15.0
    GAIN_MAX = +15.0

    # Canvas layout constants
    _PAD_L  = 52    # left  (y-axis labels)
    _PAD_R  = 16    # right
    _PAD_T  = 18    # top
    _PAD_B  = 36    # bottom (x-axis labels)

    _CANVAS_W = 620
    _CANVAS_H = 300

    def __init__(
        self,
        parent,
        eq: ParametricEQ,
        mode: str = "theater",
        speaker_list: list | None = None,
        output_device_name: str = "",
        output_device_idx: int | None = None,
        on_close=None,
        fs: int = 48000,
        eq_state_path: str = "",
    ):
        self._parent      = parent
        self._eq          = eq
        self._mode        = mode
        self._spk_list    = speaker_list or []
        self._dev_name    = output_device_name
        self._dev_idx     = output_device_idx   # sounddevice index for Stage 2
        self._on_close    = on_close
        self._fs          = fs
        self._state_path  = eq_state_path

        # DeviceAnalyzer — shared across the dialog lifetime
        self._analyzer = DeviceAnalyzer(fs=fs)

        # Per-speaker active EQ index (multi mode)
        self._active_spk_idx = 0

        # Dragging state
        self._drag_band:  int | None = None
        self._drag_start: tuple | None = None
        self._drag_init:  tuple | None = None
        self._hovered_band: int | None = None

        # Diagnostic background thread (Stage 2)
        self._diag_running = False

        # Active reference curves to overlay
        self._active_curves: set[str] = {"Flat (0 dB)"}

        # Frequency points for response calculation (log-spaced)
        self._viz_freqs = np.geomspace(self.FREQ_MIN, self.FREQ_MAX, 512)

        # Build window
        self._win = tk.Toplevel(parent)
        self._win.title("EQ Editor")
        self._win.configure(bg="#111113")
        self._win.resizable(False, False)
        self._win.protocol("WM_DELETE_WINDOW", self._close)

        self._build_ui()
        self._detect_profile()
        self._redraw()

        # Center on parent
        try:
            px = parent.winfo_rootx() + parent.winfo_width()  // 2
            py = parent.winfo_rooty() + parent.winfo_height() // 2
            ww = self._CANVAS_W + self._PAD_L + 40
            wh = self._CANVAS_H + 260
            self._win.geometry(f"{ww}x{wh}+{px - ww//2}+{py - wh//2}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        win  = self._win
        C_bg = "#111113"
        C_sf = "#1c1c1e"
        C_sf2= "#0d0d0f"
        C_ac = "#0A84FF"
        C_tx = "#F5F5F7"
        C_dm = "#8E8E93"

        # ---- Top bar: speaker selector + profile + auto-detect ----------
        top = tk.Frame(win, bg=C_bg)
        top.pack(fill="x", padx=14, pady=(10, 4))

        if self._mode == "multi_speaker" and self._spk_list:
            tk.Label(top, text="Speaker:", bg=C_bg, fg=C_dm,
                     font=("Segoe UI", 10)).pack(side="left")
            self._spk_var = tk.StringVar()
            spk_names = [f"#{i+1}  {s['label']}" for i, s in enumerate(self._spk_list)]
            self._spk_combo = ttk.Combobox(
                top, textvariable=self._spk_var,
                values=spk_names, state="readonly", width=22,
            )
            self._spk_combo.set(spk_names[0] if spk_names else "")
            self._spk_combo.pack(side="left", padx=(4, 14))
            self._spk_combo.bind("<<ComboboxSelected>>", self._on_speaker_select)

        self._profile_lbl = tk.Label(
            top, text="Starting point: —",
            bg=C_bg, fg=C_dm, font=("Segoe UI", 9),
        )
        self._profile_lbl.pack(side="left", padx=(0, 10))

        self._detect_btn = tk.Button(
            top, text="Run Diagnostic",
            bg=C_sf, fg=C_tx, relief="flat",
            font=("Segoe UI", 9), padx=8, pady=3,
            command=self._detect_and_apply,
        )
        self._detect_btn.pack(side="left", padx=(0, 6))

        # Acoustic sweep — requires a mic input; shown alongside Run Diagnostic
        self._acoustic_btn = tk.Button(
            top, text="Acoustic Sweep…",
            bg=C_sf, fg=C_dm, relief="flat",
            font=("Segoe UI", 9), padx=8, pady=3,
            command=self._open_acoustic_sweep,
        )
        self._acoustic_btn.pack(side="left", padx=(0, 6))

        tk.Button(
            top, text="Reset Flat",
            bg=C_sf, fg=C_dm, relief="flat",
            font=("Segoe UI", 9), padx=8, pady=3,
            command=self._reset_flat,
        ).pack(side="left", padx=(0, 6))

        # ---- Reference curve row ----------------------------------------
        ref_row = tk.Frame(win, bg=C_bg)
        ref_row.pack(fill="x", padx=14, pady=(0, 4))

        tk.Label(ref_row, text="Reference:", bg=C_bg, fg=C_dm,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))

        self._curve_btns: dict[str, tk.Button] = {}
        for cname in REFERENCE_CURVES:
            short = (cname.split("/")[0]
                     .replace(" (0 dB)", "")
                     .replace(" (2018)", "")
                     .replace("Harman ", "")
                     .strip())
            col  = CURVE_COLORS.get(cname, "#4a4a6a")
            active = cname in self._active_curves
            btn = tk.Button(
                ref_row,
                text=short,
                bg=col if active else C_sf,
                fg="white",
                relief="flat",
                font=("Segoe UI", 8),
                padx=6, pady=2,
                command=lambda n=cname: self._toggle_ref_curve(n),
            )
            btn.pack(side="left", padx=2)
            self._curve_btns[cname] = btn

        # ---- Main EQ canvas ---------------------------------------------
        canvas_frame = tk.Frame(win, bg=C_sf2, bd=0)
        canvas_frame.pack(padx=14, pady=(0, 6))

        self._canvas = tk.Canvas(
            canvas_frame,
            width=self._CANVAS_W,
            height=self._CANVAS_H,
            bg=C_sf2, highlightthickness=0, bd=0,
        )
        self._canvas.pack()

        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<MouseWheel>",      self._on_scroll)
        self._canvas.bind("<Motion>",          self._on_hover)
        self._canvas.bind("<Leave>",           lambda _: self._clear_hover())

        # ---- Band info row ----------------------------------------------
        band_row = tk.Frame(win, bg=C_bg)
        band_row.pack(fill="x", padx=14, pady=(0, 4))

        self._band_info_lbl = tk.Label(
            band_row, text="Hover over a band node for details",
            bg=C_bg, fg=C_dm, font=("Segoe UI", 9), anchor="w",
        )
        self._band_info_lbl.pack(side="left", fill="x", expand=True)

        self._q_lbl = tk.Label(
            band_row, text="Scroll wheel to adjust Q",
            bg=C_bg, fg=C_dm, font=("Segoe UI", 9), anchor="e",
        )
        self._q_lbl.pack(side="right")

        # ---- Profile description ----------------------------------------
        self._desc_lbl = tk.Label(
            win, text="",
            bg=C_bg, fg=C_dm, font=("Segoe UI", 9),
            wraplength=self._CANVAS_W + 10, justify="left", anchor="w",
        )
        self._desc_lbl.pack(fill="x", padx=14, pady=(0, 6))

        # ---- Bottom buttons --------------------------------------------
        bot = tk.Frame(win, bg=C_bg)
        bot.pack(fill="x", padx=14, pady=(2, 14))

        tk.Button(
            bot, text="Save & Close",
            bg="#30D158", fg="#0d1117",
            relief="flat", font=("Segoe UI", 11, "bold"),
            padx=16, pady=6,
            command=self._save_close,
        ).pack(side="right", padx=(6, 0))

        tk.Button(
            bot, text="Close",
            bg=C_sf, fg=C_tx,
            relief="flat", font=("Segoe UI", 10),
            padx=12, pady=6,
            command=self._close,
        ).pack(side="right")

        # Gain labels along left edge
        for g_db in (-12, -9, -6, -3, 0, 3, 6, 9, 12):
            y = self._gain_to_y(g_db)
            lbl_text = f"{g_db:+d}" if g_db != 0 else "0"
            tk.Label(
                canvas_frame, text=lbl_text,
                bg=C_sf2, fg="#555566",
                font=("Consolas", 7), width=4, anchor="e",
            ).place(x=0, y=y - 6)

    # ------------------------------------------------------------------
    # Coordinate transforms

    @property
    def _plot_left(self):  return self._PAD_L
    @property
    def _plot_right(self): return self._CANVAS_W - self._PAD_R
    @property
    def _plot_top(self):   return self._PAD_T
    @property
    def _plot_bot(self):   return self._CANVAS_H - self._PAD_B
    @property
    def _plot_w(self):     return self._plot_right - self._plot_left
    @property
    def _plot_h(self):     return self._plot_bot   - self._plot_top

    def _freq_to_x(self, freq: float) -> float:
        frac = math.log10(max(freq, 1.0) / self.FREQ_MIN) / math.log10(self.FREQ_MAX / self.FREQ_MIN)
        return self._plot_left + frac * self._plot_w

    def _x_to_freq(self, x: float) -> float:
        frac = (x - self._plot_left) / self._plot_w
        frac = max(0.0, min(1.0, frac))
        return self.FREQ_MIN * 10 ** (frac * math.log10(self.FREQ_MAX / self.FREQ_MIN))

    def _gain_to_y(self, gain_db: float) -> float:
        frac = 1.0 - (gain_db - self.GAIN_MIN) / (self.GAIN_MAX - self.GAIN_MIN)
        return self._plot_top + frac * self._plot_h

    def _y_to_gain(self, y: float) -> float:
        frac = (y - self._plot_top) / self._plot_h
        return self.GAIN_MAX - frac * (self.GAIN_MAX - self.GAIN_MIN)

    # ------------------------------------------------------------------
    # Drawing

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        self._draw_grid()
        self._draw_ref_curves()
        self._draw_eq_curve()
        self._draw_nodes()

    def _draw_grid(self):
        c = self._canvas
        bg2 = "#1a1a1e"
        dim = "#252530"

        # Fill plot area
        c.create_rectangle(
            self._plot_left, self._plot_top,
            self._plot_right, self._plot_bot,
            fill=bg2, outline="",
        )

        # Frequency gridlines + labels
        freq_marks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for f in freq_marks:
            x = self._freq_to_x(f)
            c.create_line(x, self._plot_top, x, self._plot_bot,
                          fill=dim, width=1)
            label = (f"{f//1000}k" if f >= 1000 else str(f))
            c.create_text(x, self._plot_bot + 10,
                          text=label, fill="#44445a", font=("Consolas", 7))

        # Gain gridlines
        for g_db in (-12, -9, -6, -3, 0, 3, 6, 9, 12):
            y    = self._gain_to_y(g_db)
            col  = "#3a3a4a" if g_db == 0 else dim
            wid  = 1.5 if g_db == 0 else 1
            c.create_line(self._plot_left, y, self._plot_right, y,
                          fill=col, width=wid)

        # Axis labels
        c.create_text(self._plot_left - 4, self._plot_top - 10,
                      text="dB", fill="#44445a", font=("Consolas", 7), anchor="w")
        c.create_text((self._plot_left + self._plot_right) // 2,
                      self._CANVAS_H - 6,
                      text="Frequency (Hz)", fill="#44445a",
                      font=("Consolas", 7))

    def _interp_curve_db(self, points: list) -> np.ndarray:
        """Interpolate a reference curve at self._viz_freqs (log scale)."""
        pts  = sorted(points, key=lambda p: p[0])
        xp   = np.log10([p[0] for p in pts])
        yp   = [p[1] for p in pts]
        xq   = np.log10(self._viz_freqs)
        return np.interp(xq, xp, yp)

    def _curve_to_canvas_points(self, gain_arr: np.ndarray) -> list:
        pts = []
        for freq, g in zip(self._viz_freqs, gain_arr):
            pts.append(self._freq_to_x(float(freq)))
            pts.append(self._gain_to_y(float(g)))
        return pts

    def _draw_ref_curves(self):
        c = self._canvas
        for cname in self._active_curves:
            if cname not in REFERENCE_CURVES or cname == "Flat (0 dB)":
                continue
            pts_def = REFERENCE_CURVES[cname]
            gain_arr = self._interp_curve_db(pts_def)
            pts = self._curve_to_canvas_points(gain_arr)
            if len(pts) >= 4:
                col = CURVE_COLORS.get(cname, "#4a4a6a")
                c.create_line(*pts, fill=col, width=1.5,
                              smooth=True, dash=(4, 3))
                # Legend label at right edge
                last_y = self._gain_to_y(float(gain_arr[-1]))
                short  = cname.split("/")[0].replace(" (0 dB)", "").strip()[:16]
                c.create_text(self._plot_right - 4, last_y,
                              text=short, fill=col,
                              font=("Consolas", 7), anchor="e")

    def _draw_eq_curve(self):
        c    = self._canvas
        eq   = self._eq
        gain_arr = eq.get_frequency_response(self._viz_freqs)
        pts  = self._curve_to_canvas_points(gain_arr)
        if len(pts) >= 4:
            c.create_line(*pts, fill="#e8e8f0", width=2.0, smooth=True)

    def _draw_nodes(self):
        c  = self._canvas
        eq = self._eq
        params = eq.get_band_params()

        for i, p in enumerate(params):
            btype    = p["type"]
            freq     = p["freq"]
            gain_db  = p["gain_db"]
            enabled  = p["enabled"]

            # HP/LP nodes are fixed-gain; show at 0 dB gain position
            show_gain = gain_db if btype not in ("highpass", "lowpass") else 0.0

            x = self._freq_to_x(freq)
            y = self._gain_to_y(show_gain)

            if not (self._plot_left <= x <= self._plot_right
                    and self._plot_top  <= y <= self._plot_bot + 10):
                continue

            col  = _NODE_COLORS.get(btype, "#5ad3e0")
            r    = _SEL_R if i == self._drag_band or i == self._hovered_band else _NODE_R
            alpha_fill = col if enabled else "#333340"
            outline    = "#ffffff" if i == self._drag_band else (
                "#aaaacc" if i == self._hovered_band else "#555566"
            )

            c.create_oval(x - r, y - r, x + r, y + r,
                          fill=alpha_fill, outline=outline, width=1.5,
                          tags=(f"node_{i}",))

            # Tiny type label inside node
            lbl_map = {"highpass": "HP", "lowpass": "LP",
                       "lowshelf": "LS", "highshelf": "HS",
                       "peaking":  f"P{i-1}" if i >= 2 else "LS"}
            lbl_map["peaking"] = f"P{i - 1}"
            short = lbl_map.get(btype, btype[:2].upper())
            c.create_text(x, y, text=short, fill="#0d0d0f",
                          font=("Consolas", 6, "bold"))

    # ------------------------------------------------------------------
    # Hit testing

    def _hit_test(self, cx: float, cy: float) -> int | None:
        """Return band index under (cx, cy), or None."""
        eq = self._eq
        params = eq.get_band_params()
        best_idx  = None
        best_dist = float("inf")
        for i, p in enumerate(params):
            freq    = p["freq"]
            gain_db = p["gain_db"] if p["type"] not in ("highpass", "lowpass") else 0.0
            nx = self._freq_to_x(freq)
            ny = self._gain_to_y(gain_db)
            dist = math.hypot(cx - nx, cy - ny)
            if dist < _SEL_R + 4 and dist < best_dist:
                best_dist = dist
                best_idx  = i
        return best_idx

    # ------------------------------------------------------------------
    # Event handlers

    def _on_press(self, event):
        idx = self._hit_test(event.x, event.y)
        if idx is not None:
            p = self._eq.get_band_params()[idx]
            self._drag_band  = idx
            self._drag_start = (event.x, event.y)
            self._drag_init  = (p["freq"], p["gain_db"])
            self._redraw()

    def _on_drag(self, event):
        if self._drag_band is None:
            return
        init_freq, init_gain = self._drag_init  # type: ignore
        sx, sy = self._drag_start               # type: ignore
        dx = event.x - sx
        dy = event.y - sy

        # Frequency: log-scale drag
        freq_log_range = math.log10(self.FREQ_MAX / self.FREQ_MIN)
        d_log = dx / self._plot_w * freq_log_range
        new_freq  = float(np.clip(
            init_freq * 10.0 ** d_log,
            self.FREQ_MIN, self.FREQ_MAX,
        ))

        # Gain: linear, scaled to plot height
        d_gain = -dy / self._plot_h * (self.GAIN_MAX - self.GAIN_MIN)
        new_gain = float(np.clip(init_gain + d_gain, -12.0, +12.0))

        btype = self._eq.get_band_params()[self._drag_band]["type"]
        if btype in ("highpass", "lowpass"):
            self._eq.update_band(self._drag_band, freq=new_freq)
        else:
            self._eq.update_band(self._drag_band, freq=new_freq, gain_db=new_gain)

        self._redraw()
        self._update_band_info(self._drag_band)

    def _on_release(self, event):
        self._drag_band  = None
        self._drag_start = None
        self._drag_init  = None
        self._redraw()

    def _on_scroll(self, event):
        idx = self._hit_test(event.x, event.y)
        if idx is None:
            idx = self._drag_band
        if idx is None:
            return
        params = self._eq.get_band_params()
        if idx >= len(params):
            return
        p     = params[idx]
        btype = p["type"]
        if btype in ("highpass", "lowpass"):
            return  # Q not user-adjustable for HP/LP

        direction = 1 if event.delta > 0 else -1
        old_q     = p["q"]
        # Scroll up → wider (lower Q), scroll down → narrower (higher Q)
        new_q = float(np.clip(old_q * (1.0 - direction * 0.07), 0.3, 8.0))
        self._eq.update_band(idx, q=new_q)
        self._redraw()
        self._update_band_info(idx)

    def _on_hover(self, event):
        idx = self._hit_test(event.x, event.y)
        if idx != self._hovered_band:
            self._hovered_band = idx
            self._redraw()
        if idx is not None:
            self._update_band_info(idx)

    def _clear_hover(self):
        if self._hovered_band is not None:
            self._hovered_band = None
            self._redraw()
            self._band_info_lbl.configure(
                text="Hover over a band node for details")

    def _update_band_info(self, idx: int):
        params = self._eq.get_band_params()
        if idx >= len(params):
            return
        p     = params[idx]
        btype = p["type"]
        freq  = p["freq"]
        gain  = p["gain_db"]
        q     = p["q"]
        f_str = (f"{freq/1000:.1f} kHz" if freq >= 1000 else f"{freq:.0f} Hz")
        type_names = {"highpass": "High-pass", "lowpass": "Low-pass",
                      "lowshelf": "Low shelf", "highshelf": "High shelf",
                      "peaking": "Peaking"}
        t_str = type_names.get(btype, btype)
        if btype in ("highpass", "lowpass"):
            info = f"Band {idx+1}: {t_str}  |  fc = {f_str}  |  Q = {q:.2f}"
        else:
            info = (f"Band {idx+1}: {t_str}  |  "
                    f"fc = {f_str}  |  "
                    f"Gain = {gain:+.1f} dB  |  "
                    f"Q = {q:.2f}")
        self._band_info_lbl.configure(text=info)

    # ------------------------------------------------------------------
    # Reference curve toggle

    def _toggle_ref_curve(self, cname: str):
        if cname in self._active_curves:
            self._active_curves.discard(cname)
            self._curve_btns[cname].configure(
                bg="#1c1c1e")
        else:
            self._active_curves.add(cname)
            col = CURVE_COLORS.get(cname, "#4a4a6a")
            self._curve_btns[cname].configure(bg=col)
        self._redraw()

    # ------------------------------------------------------------------
    # Speaker profile detection / diagnostic

    def _current_device_name(self) -> str:
        if self._mode == "multi_speaker" and self._spk_list:
            idx = self._active_spk_idx
            if idx < len(self._spk_list):
                return self._spk_list[idx].get("device_name", "")
        return self._dev_name

    def _current_device_idx(self) -> int | None:
        """Return sounddevice index for the currently selected speaker, or None."""
        if self._mode == "multi_speaker" and self._spk_list:
            idx = self._active_spk_idx
            if idx < len(self._spk_list):
                return self._spk_list[idx].get("device_idx")
        # Theater mode: use the stored output device index
        return self._dev_idx

    def _detect_profile(self):
        """Stage 1 only — instant database lookup, no audio played."""
        dev = self._current_device_name()
        if not dev:
            self._profile_lbl.configure(text="No device selected")
            self._desc_lbl.configure(text="")
            return
        _, bands, confidence, display, desc = get_correction_bands(dev)
        conf_str = f"  ({confidence*100:.0f}% match)" if confidence < 0.80 else ""
        self._profile_lbl.configure(
            text=f"Starting point: {display}{conf_str}"
        )
        self._desc_lbl.configure(text=desc)

    def _detect_and_apply(self):
        """
        Run the full diagnostic (Stage 1 + Stage 2 loopback if available)
        and apply the resulting EQ correction.  Stage 2 runs in a background
        thread so the UI stays responsive.
        """
        if self._diag_running:
            return
        dev       = self._current_device_name()
        dev_idx   = self._current_device_idx()
        params    = self._eq.get_band_params()
        band_freqs= [p["freq"] for p in params]

        # Immediately apply Stage 1 (instant name-based lookup)
        _, bands, conf, display, desc = get_correction_bands(dev)
        from dsp.speaker_profiles import _bands_to_gains
        gains_s1 = _bands_to_gains(bands, band_freqs)

        if not dev:
            # No device selected at all
            self._profile_lbl.configure(text="No output device selected.")
            self._desc_lbl.configure(text="Select an output device, then run the diagnostic.")
            return

        for i, g in enumerate(gains_s1):
            self._eq.update_band(i, gain_db=g)
        self._redraw()

        # Show immediate Stage 1 feedback
        conf_str = f"  ({conf*100:.0f}%)" if conf < 0.80 else ""
        if not any(abs(g) > 0.05 for g in gains_s1):
            self._profile_lbl.configure(
                text=f"No correction: {display}{conf_str}  (device not in database)"
            )
            self._desc_lbl.configure(
                text="Device not recognised — no EQ correction applied. "
                     "Adjust nodes manually or try Acoustic Sweep for a measurement-based profile."
            )
        else:
            self._profile_lbl.configure(text=f"Applied: {display}{conf_str}")
            self._desc_lbl.configure(text=desc)

        # Launch Stage 2 in background thread (plays brief -20 dBFS tones for loopback analysis)
        self._diag_running = True
        self._detect_btn.configure(text="Running…", state="disabled")

        def _run():
            try:
                from dsp.device_analyzer import DeviceAnalyzer
                analyzer = DeviceAnalyzer(fs=self._fs)
                result   = analyzer.analyze_device(
                    device_name      = dev,
                    device_idx       = dev_idx,
                    band_freqs       = band_freqs,
                    run_loopback     = True,
                    progress_callback= lambda stage, pct, note="": (
                        self._win.after(0, lambda s=stage, p=pct, n=note:
                                        self._on_diag_progress(s, p, n))
                        if _win_alive() else None
                    ),
                )
            except Exception as exc:
                from dsp.device_analyzer import AnalysisResult
                result = AnalysisResult(
                    display_name = display,
                    description  = desc,
                    confidence   = conf,
                    band_gains   = list(gains_s1),
                    notes        = [f"Stage 2 failed: {exc}"],
                )
            if _win_alive():
                self._win.after(0, lambda r=result: self._on_diag_complete(r))
            else:
                self._diag_running = False

        def _win_alive() -> bool:
            try:
                return bool(self._win.winfo_exists())
            except Exception:
                return False

        threading.Thread(target=_run, daemon=True).start()

    def _on_diag_progress(self, stage: str, pct: float, note: str = ""):
        """Called from the main thread with Stage 2 progress updates."""
        try:
            if note:
                self._profile_lbl.configure(text=note[:70])
        except Exception:
            pass

    def _on_diag_complete(self, result):
        """Called from the main thread when Stage 2 finishes."""
        self._diag_running = False
        try:
            self._detect_btn.configure(text="Run Diagnostic", state="normal")
        except Exception:
            pass

        if not result.loopback_ran:
            # Stage 2 skipped — Stage 1 already applied at the top of _detect_and_apply
            conf_str = f"  ({result.confidence*100:.0f}%)" if result.confidence < 0.80 else ""
            has_correction = any(abs(g) > 0.05 for g in result.band_gains)
            if has_correction:
                self._profile_lbl.configure(text=f"Applied: {result.display_name}{conf_str}")
                self._desc_lbl.configure(text=result.description)
            # else: label already set to "No correction" message above
            return

        # Stage 2 ran — apply the combined (Stage 1 + driver correction) gains
        params     = self._eq.get_band_params()
        band_freqs = [p["freq"] for p in params]
        for i, g in enumerate(result.band_gains):
            if i < len(band_freqs):
                self._eq.update_band(i, gain_db=g)
        self._redraw()

        # Build a summary label
        headroom_str = (
            f"  |  Headroom: {result.headroom_db:+.0f} dBFS"
            if result.headroom_db is not None else ""
        )
        loopback_note = ""
        if result.driver_eq_notes:
            loopback_note = f"\nLoopback: {result.driver_eq_notes}"
        conf_str = f"  ({result.confidence*100:.0f}%)" if result.confidence < 0.80 else ""

        self._profile_lbl.configure(
            text=f"Applied: {result.display_name}{conf_str}{headroom_str}"
        )
        self._desc_lbl.configure(
            text=(result.description + loopback_note).strip()
        )

    def _open_acoustic_sweep(self):
        """Open a small dialog to configure and run the acoustic sweep (Stage 3)."""
        import sounddevice as _sd_check
        try:
            devs = _sd_check.query_devices()
        except Exception:
            tk.messagebox.showerror(
                "Acoustic Sweep",
                "sounddevice not available — cannot run acoustic sweep.",
                parent=self._win,
            )
            return

        # Build input device list
        in_devs = [(i, d["name"]) for i, d in enumerate(devs)
                   if d.get("max_input_channels", 0) > 0]
        if not in_devs:
            tk.messagebox.showerror(
                "Acoustic Sweep",
                "No microphone input devices found.",
                parent=self._win,
            )
            return

        dlg = tk.Toplevel(self._win)
        dlg.title("Acoustic Sweep")
        dlg.configure(bg="#111113")
        dlg.resizable(False, False)
        dlg.grab_set()

        C_bg = "#111113"; C_sf = "#1c1c1e"; C_tx = "#F5F5F7"; C_dm = "#8E8E93"

        tk.Label(dlg, text="Acoustic Sweep Measurement",
                 bg=C_bg, fg=C_tx, font=("Segoe UI", 12, "bold")).pack(padx=16, pady=(12, 4))

        warning = (
            "WARNING: This measurement uses an uncalibrated microphone.\n"
            "Built-in laptop mics have ±5–10 dB coloration — treat the\n"
            "result as a rough starting point only. Place the microphone\n"
            "1–2 m from the speaker for best results."
        )
        tk.Label(dlg, text=warning, bg=C_bg, fg="#FF9F0A",
                 font=("Segoe UI", 9), justify="left").pack(padx=16, pady=(0, 8))

        tk.Label(dlg, text="Microphone input:", bg=C_bg, fg=C_dm,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=16)
        mic_var = tk.StringVar()
        mic_opts = [f"[{i}] {name}" for i, name in in_devs]
        mic_combo = ttk.Combobox(dlg, textvariable=mic_var,
                                  values=mic_opts, state="readonly", width=38)
        mic_combo.set(mic_opts[0] if mic_opts else "")
        mic_combo.pack(padx=16, pady=(2, 8))

        status_lbl = tk.Label(dlg, text="Ready.", bg=C_bg, fg=C_dm,
                               font=("Segoe UI", 9))
        status_lbl.pack(padx=16, pady=(0, 6))

        def _run_sweep():
            sel = mic_var.get()
            try:
                mic_idx = int(sel.split("]")[0].strip("["))
            except Exception:
                return
            dev_idx = self._current_device_idx()
            if dev_idx is None:
                status_lbl.configure(text="No output device selected.", fg="#FF453A")
                return
            status_lbl.configure(text="Playing sweep — please wait…", fg=C_dm)
            run_btn.configure(state="disabled")

            def _bg():
                result = self._analyzer.run_acoustic_sweep(
                    output_device_idx= dev_idx,
                    input_device_idx = mic_idx,
                    band_freqs       = [p["freq"] for p in self._eq.get_band_params()],
                    level_dbfs       = -20.0,
                    sweep_duration   = 1.5,
                    progress_callback= lambda _s, _p, note="": (
                        dlg.after(0, lambda n=note: status_lbl.configure(text=n))
                        if dlg.winfo_exists() else None
                    ),
                )
                def _apply():
                    if not dlg.winfo_exists():
                        return
                    run_btn.configure(state="normal")
                    # Ask user to confirm before applying
                    msg = (
                        f"Acoustic sweep complete.\n"
                        f"Confidence: {result.confidence*100:.0f}%\n\n"
                        f"{result.description}\n\n"
                        "Apply this correction to the EQ?"
                    )
                    if tk.messagebox.askyesno("Apply Acoustic Correction?", msg,
                                               parent=dlg):
                        for i, g in enumerate(result.band_gains):
                            self._eq.update_band(i, gain_db=g)
                        self._redraw()
                        self._profile_lbl.configure(
                            text=f"Applied: {result.display_name}")
                        self._desc_lbl.configure(text=result.description)
                        dlg.destroy()
                    else:
                        status_lbl.configure(text="Correction not applied.", fg=C_dm)
                dlg.after(0, _apply)

            threading.Thread(target=_bg, daemon=True).start()

        run_btn = tk.Button(
            dlg, text="Start Sweep",
            bg="#0A84FF", fg="white", relief="flat",
            font=("Segoe UI", 10, "bold"), padx=12, pady=5,
            command=_run_sweep,
        )
        run_btn.pack(padx=16, pady=(0, 12))

        tk.Button(dlg, text="Cancel", bg=C_sf, fg=C_dm, relief="flat",
                  font=("Segoe UI", 9), padx=10, pady=4,
                  command=dlg.destroy).pack(pady=(0, 12))

    def _reset_flat(self):
        self._eq.reset_to_flat()
        self._profile_lbl.configure(text="Reset to flat")
        self._desc_lbl.configure(text="")
        self._redraw()

    # ------------------------------------------------------------------
    # Multi-speaker selector

    def _on_speaker_select(self, event=None):
        sel = self._spk_combo.get()  # type: ignore
        try:
            idx = int(sel.split("#")[1].split()[0]) - 1
        except Exception:
            idx = 0
        self._active_spk_idx = max(0, idx)
        if idx < len(self._spk_list):
            self._eq = self._spk_list[idx]["eq"]
        self._detect_profile()
        self._redraw()

    # ------------------------------------------------------------------
    # Persistence

    def _save_state(self):
        if not self._state_path:
            return
        try:
            # Load existing
            if os.path.exists(self._state_path):
                with open(self._state_path, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            if self._mode == "theater":
                data["theater_eq"] = self._eq.get_band_params()
            else:
                spk_eqs = data.get("speaker_eqs", {})
                for i, spk in enumerate(self._spk_list):
                    key = spk.get("label", f"speaker_{i}")
                    spk_eqs[key] = spk["eq"].get_band_params()
                data["speaker_eqs"] = spk_eqs

            os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[eq_editor] save failed: {e}")

    # ------------------------------------------------------------------
    # Close

    def _save_close(self):
        self._save_state()
        self._close()

    def _close(self):
        if self._on_close:
            self._on_close()
        try:
            self._win.destroy()
        except Exception:
            pass

    def is_open(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

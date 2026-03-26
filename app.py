#!/usr/bin/env python3
"""
ModAudio - Theater Experience
GUI Application
Run with:  python app.py
"""

import sys, os, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import customtkinter as ctk
except ImportError:
    sys.exit("customtkinter not found. Run:  pip install customtkinter")

import sounddevice as sd

from config    import HEADPHONES_PRESET, SPEAKERS_PRESET, SAMPLE_RATE, BLOCK_SIZE
from audio_io  import find_default_devices
from dsp       import TheaterChain


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ---------------------------------------------------------------------------
# Presets  (merged into the base headphones / speakers preset)
# ---------------------------------------------------------------------------

PRESETS = {
    "Cinema": {
        "rt60": 1.3,  "rt60_hf": 0.65,
        "reverb_predelay_ms": 22.0,  "reverb_mix": 0.25,  "early_ref_mix": 0.45,
        "stereo_width": 2.0,  "surround_level": 0.72,  "lfe_level": 0.85,
        "rear_level": 0.60,
        "bass_boost_db": 6.0,  "sub_bass_db": 4.5,
        "bass_harm_drive": 2.8,  "bass_harm_level": 0.50,
        "air_exciter_level": 0.18,
        "mb_compress_drive": 1.6,  "transient_amount": 0.55,
        "output_gain_db": -1.5,
    },
    "IMAX": {
        "rt60": 1.7,  "rt60_hf": 0.75,
        "reverb_predelay_ms": 28.0,  "reverb_mix": 0.32,  "early_ref_mix": 0.55,
        "stereo_width": 2.3,  "surround_level": 0.85,  "lfe_level": 0.95,
        "rear_level": 0.72,
        "bass_boost_db": 9.0,  "sub_bass_db": 6.0,
        "bass_harm_drive": 3.5,  "bass_harm_level": 0.65,
        "air_exciter_level": 0.22,
        "mb_compress_drive": 1.9,  "transient_amount": 0.75,
        "output_gain_db": -2.0,
    },
    "Dolby": {
        "rt60": 1.0,  "rt60_hf": 0.55,
        "reverb_predelay_ms": 20.0,  "reverb_mix": 0.18,  "early_ref_mix": 0.38,
        "stereo_width": 1.8,  "surround_level": 0.65,  "lfe_level": 0.80,
        "rear_level": 0.55,
        "bass_boost_db": 4.5,  "sub_bass_db": 3.5,
        "bass_harm_drive": 2.2,  "bass_harm_level": 0.38,
        "air_exciter_level": 0.14,
        "mb_compress_drive": 1.3,  "transient_amount": 0.40,
        "output_gain_db": -1.5,
    },
    "Home": {
        "rt60": 0.8,  "rt60_hf": 0.45,
        "reverb_predelay_ms": 15.0,  "reverb_mix": 0.12,  "early_ref_mix": 0.28,
        "stereo_width": 1.5,  "surround_level": 0.55,  "lfe_level": 0.70,
        "rear_level": 0.45,
        "bass_boost_db": 3.0,  "sub_bass_db": 2.5,
        "bass_harm_drive": 1.8,  "bass_harm_level": 0.28,
        "air_exciter_level": 0.10,
        "mb_compress_drive": 1.1,  "transient_amount": 0.25,
        "output_gain_db": -1.5,
    },
}

# Slider definitions: (label, param_key, min, max, format_fn)
SLIDERS = [
    ("Reverb",   "rt60",             0.3,  2.5, lambda v: f"{v:.1f} s"),
    ("Width",    "stereo_width",     1.0,  2.8, lambda v: f"{v:.1f}\u00d7"),
    ("Bass",     "bass_boost_db",    0.0, 12.0, lambda v: f"+{v:.0f} dB"),
    ("Dynamics", "mb_compress_drive",1.0,  2.2, lambda v: f"{v:.1f}"),
]

# Color palette
C = {
    "accent":    "#4361ee",
    "success":   "#06d6a0",
    "danger":    "#ef476f",
    "warn":      "#ffd166",
    "surface":   "#1c2333",
    "surface2":  "#161b27",
    "dim":       "#6e7a8a",
    "text":      "#e8edf3",
}


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class ModAudioApp(ctk.CTk):

    W, H = 520, 800

    def __init__(self):
        super().__init__()

        self.title("ModAudio")
        self.geometry(f"{self.W}x{self.H}")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # -- Audio state
        self._running    = False
        self._stream     = None
        self._chain      = None
        self._rb_timer   = None        # debounce rebuild timer
        self._xruns      = 0
        self._blk_count  = 0
        self._t_start    = 0.0

        # Level meter state
        self._raw_in  = np.zeros(2, dtype=np.float32)
        self._raw_out = np.zeros(2, dtype=np.float32)
        self._dsp_in  = np.zeros(2, dtype=np.float32)  # display-smoothed
        self._dsp_out = np.zeros(2, dtype=np.float32)

        # DSP parameters
        self._mode         = "headphones"
        self._preset_name  = "Cinema"
        self._slider_vals  = {}   # overrides from sliders

        # -- Discover audio devices
        self._devs     = sd.query_devices()
        self._hostapis = sd.query_hostapis()
        self._build_device_lists()

        auto_in, auto_out = find_default_devices()
        # Select best input from our filtered loopback list
        if auto_in is not None and any(i == auto_in for i, *_ in self._in_list):
            self._in_dev_idx = auto_in
        else:
            self._in_dev_idx = self._in_list[0][0] if self._in_list else 0
        # Select output compatible with chosen input
        compat = self._compatible_outputs(self._in_dev_idx)
        if auto_out is not None and any(i == auto_out for i, *_ in compat):
            self._out_dev_idx = auto_out
        else:
            self._out_dev_idx = compat[0][0] if compat else 0
        self._out_list = compat

        # -- Build and start
        self._build_ui()
        self._apply_preset("Cinema", animate=False)
        self._tick_meters()

    # =======================================================================
    # Device discovery helpers
    # =======================================================================

    # Keywords that identify system-audio loopback / capture sources.
    _LOOPBACK_KW = ("stereo mix", "what u hear", "wave out mix",
                    "loopback", "cable output", "vb-audio")

    def _dev_hostapi(self, device_idx: int) -> int:
        """Return the host-API index for a given device index."""
        try:
            return int(self._devs[device_idx]["hostapi"])
        except (IndexError, KeyError):
            return -1

    def _hostapi_label(self, ha_idx: int) -> str:
        """Short host-API tag, e.g. 'WASAPI', 'MME', 'DS'."""
        try:
            name = self._hostapis[ha_idx]["name"]
            if "WASAPI" in name:   return "WASAPI"
            if "MME"   in name:   return "MME"
            if "DirectSound" in name: return "DS"
            if "WDM"   in name:   return "WDM"
            return name[:8]
        except (IndexError, KeyError):
            return "?"

    def _build_device_lists(self):
        """
        Input list  – only loopback / system-capture sources so the user can
                      only capture audio that is already playing on the PC.
        Out list    – rebuilt dynamically to match the input's host API.
        Each entry is (device_idx, display_name).
        """
        capture_devs = []
        for i, d in enumerate(self._devs):
            if d["max_input_channels"] < 1:
                continue
            nl = d["name"].lower()
            if any(kw in nl for kw in self._LOOPBACK_KW):
                tag   = self._hostapi_label(d["hostapi"])
                label = f"{d['name'][:44]}  [{tag}]"
                capture_devs.append((i, label))

        # Fallback: if no loopback devices found, show all input devices
        if not capture_devs:
            for i, d in enumerate(self._devs):
                if d["max_input_channels"] >= 1:
                    tag   = self._hostapi_label(d["hostapi"])
                    label = f"{d['name'][:44]}  [{tag}]"
                    capture_devs.append((i, label))

        self._in_list = capture_devs

    def _compatible_outputs(self, input_dev_idx: int) -> list:
        """
        Return list of (device_idx, display_name) for output devices that
        share the same host API as input_dev_idx.  Falls back to all outputs
        if nothing matches.
        """
        ha = self._dev_hostapi(input_dev_idx)
        same_api = []
        for i, d in enumerate(self._devs):
            if d["max_output_channels"] < 1:
                continue
            if d["hostapi"] == ha:
                tag   = self._hostapi_label(d["hostapi"])
                label = f"{d['name'][:44]}  [{tag}]"
                same_api.append((i, label))

        if same_api:
            return same_api

        # Fallback: all output devices
        out = []
        for i, d in enumerate(self._devs):
            if d["max_output_channels"] >= 1:
                tag   = self._hostapi_label(d["hostapi"])
                label = f"{d['name'][:44]}  [{tag}]"
                out.append((i, label))
        return out

    def _refresh_out_menu(self):
        """Rebuild the output dropdown to only show devices compatible with current input."""
        compat = self._compatible_outputs(self._in_dev_idx)
        self._out_list = compat
        names = [label for _, label in compat]
        if not names:
            names = ["No compatible output found"]

        # Try to keep the previously selected output if it's still valid
        cur_name = next((label for idx, label in compat if idx == self._out_dev_idx), None)
        if cur_name is None:
            self._out_dev_idx = compat[0][0] if compat else 0
            cur_name = names[0]

        self._out_dev_menu.configure(values=names)
        self._out_dev_menu.set(cur_name)

    # =======================================================================
    # UI construction
    # =======================================================================

    def _build_ui(self):
        pad = 20

        # Root scrollable frame (handles small displays gracefully)
        self._root_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=C["surface"],
            scrollbar_button_hover_color=C["dim"],
        )
        self._root_frame.pack(fill="both", expand=True, padx=0, pady=0)

        self._build_header(pad)
        self._build_section("PRESET", pad)
        self._build_presets(pad)
        self._build_section("MODE", pad)
        self._build_mode(pad)
        self._build_section("AUDIO DEVICES", pad)
        self._build_devices(pad)
        self._build_section("LEVELS", pad)
        self._build_meters(pad)
        self._build_section("FINE TUNE", pad)
        self._build_sliders(pad)
        self._build_transport(pad)
        self._build_statusbar(pad)

    # -- Header --------------------------------------------------------------

    def _build_header(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color=C["surface2"], corner_radius=0,
                         height=68)
        f.pack(fill="x", padx=0, pady=(0, 2))
        f.pack_propagate(False)

        ctk.CTkLabel(
            f, text="ModAudio",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=C["text"],
        ).place(x=pad, y=16)

        ctk.CTkLabel(
            f, text="Theater Experience",
            font=ctk.CTkFont(size=12),
            text_color=C["dim"],
        ).place(x=pad + 4, y=44)

        # Status badge (right side)
        self._status_badge = ctk.CTkLabel(
            f, text="  STOPPED",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C["danger"],
            fg_color=C["surface"],
            corner_radius=8,
            padx=10, pady=4,
        )
        self._status_badge.place(relx=1.0, x=-pad, y=20, anchor="ne")

    # -- Section label -------------------------------------------------------

    def _build_section(self, title, pad):
        ctk.CTkLabel(
            self._root_frame,
            text=title,
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=C["dim"],
        ).pack(anchor="w", padx=pad + 2, pady=(12, 2))

    # -- Presets -------------------------------------------------------------

    def _build_presets(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color="transparent")
        f.pack(fill="x", padx=pad, pady=(0, 4))

        self._preset_btns = {}
        names = list(PRESETS.keys())
        for col, name in enumerate(names):
            btn = ctk.CTkButton(
                f,
                text=name,
                font=ctk.CTkFont(size=13),
                fg_color=C["accent"],
                hover_color="#5a73f5",
                text_color="white",
                corner_radius=8,
                height=36,
                command=lambda n=name: self._apply_preset(n),
            )
            btn.grid(row=0, column=col, padx=(0 if col == 0 else 6, 0), sticky="ew")
            f.grid_columnconfigure(col, weight=1)
            self._preset_btns[name] = btn

    # -- Mode ----------------------------------------------------------------

    def _build_mode(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color="transparent")
        f.pack(fill="x", padx=pad, pady=(0, 4))

        self._mode_seg = ctk.CTkSegmentedButton(
            f,
            values=["Headphones", "Speakers", "Surround", "Mono"],
            font=ctk.CTkFont(size=12),
            height=36,
            selected_color=C["accent"],
            selected_hover_color="#5a73f5",
            corner_radius=8,
            command=self._on_mode_change,
        )
        self._mode_seg.set("Headphones")
        self._mode_seg.pack(fill="x")

        # Mode description label
        self._mode_desc = ctk.CTkLabel(
            f,
            text="Binaural 5.1 HRTF — optimised for headphones",
            font=ctk.CTkFont(size=10),
            text_color=C["dim"],
            anchor="w",
        )
        self._mode_desc.pack(fill="x", padx=2, pady=(4, 0))

    # -- Devices -------------------------------------------------------------

    def _build_devices(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color=C["surface"],
                         corner_radius=10)
        f.pack(fill="x", padx=pad, pady=(0, 4))

        rows = [
            ("Capture", self._in_list,  "_in_dev_menu",  "_in_dev_idx"),
            ("Output",  self._out_list, "_out_dev_menu", "_out_dev_idx"),
        ]
        for r, (label, dev_list, attr_menu, attr_idx) in enumerate(rows):
            ctk.CTkLabel(
                f, text=label,
                font=ctk.CTkFont(size=12),
                text_color=C["dim"],
                width=60, anchor="w",
            ).grid(row=r, column=0, padx=(14, 6), pady=8, sticky="w")

            names    = [d[1] for d in dev_list]
            cur_idx  = getattr(self, attr_idx)
            cur_pos  = next((j for j, (i, _) in enumerate(dev_list) if i == cur_idx), 0)
            cur_name = names[cur_pos] if names else "No devices"

            menu = ctk.CTkOptionMenu(
                f,
                values=names if names else ["No devices found"],
                font=ctk.CTkFont(size=11),
                fg_color=C["surface2"],
                button_color=C["accent"],
                button_hover_color="#5a73f5",
                dropdown_fg_color=C["surface2"],
                dropdown_hover_color=C["surface"],
                corner_radius=7,
                height=32,
                command=lambda v, a=attr_menu: self._on_device_change(v, a),
            )
            menu.set(cur_name)
            menu.grid(row=r, column=1, padx=(0, 14), pady=8, sticky="ew")
            setattr(self, attr_menu, menu)

        f.grid_columnconfigure(1, weight=1)

    # -- Level meters --------------------------------------------------------

    def _build_meters(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color=C["surface"], corner_radius=10)
        f.pack(fill="x", padx=pad, pady=(0, 4))
        f.grid_columnconfigure(1, weight=1)

        METER_H = 14
        rows = [
            ("In  L",  "_cvs_inL"),
            ("In  R",  "_cvs_inR"),
            ("Out L",  "_cvs_outL"),
            ("Out R",  "_cvs_outR"),
        ]
        for r, (label, attr) in enumerate(rows):
            ctk.CTkLabel(
                f, text=label,
                font=ctk.CTkFont(size=11, family="Consolas"),
                text_color=C["dim"],
                width=46, anchor="w",
            ).grid(row=r, column=0, padx=(14, 6), pady=(7 if r == 0 else 4, 0))

            cvs = ctk.CTkCanvas(f, height=METER_H, bg=C["surface2"],
                                highlightthickness=0, bd=0)
            cvs.grid(row=r, column=1, padx=(0, 6), pady=(7 if r == 0 else 4, 0),
                     sticky="ew")
            setattr(self, attr, cvs)

            lbl = ctk.CTkLabel(
                f, text="-inf",
                font=ctk.CTkFont(size=10, family="Consolas"),
                text_color=C["dim"],
                width=44, anchor="e",
            )
            lbl.grid(row=r, column=2, padx=(0, 14),
                     pady=(7 if r == 0 else 4, 0))
            setattr(self, attr.replace("_cvs_", "_lbl_"), lbl)

        # Separator row at bottom
        ctk.CTkLabel(f, text="", height=6).grid(row=4, column=0)

    # -- Sliders -------------------------------------------------------------

    def _build_sliders(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color=C["surface"], corner_radius=10)
        f.pack(fill="x", padx=pad, pady=(0, 4))
        f.grid_columnconfigure(1, weight=1)

        self._slider_widgets = {}
        self._slider_var_lbls = {}

        for r, (label, key, lo, hi, fmt) in enumerate(SLIDERS):
            ctk.CTkLabel(
                f, text=label,
                font=ctk.CTkFont(size=12),
                text_color=C["text"],
                width=70, anchor="w",
            ).grid(row=r, column=0, padx=(14, 4),
                   pady=(10 if r == 0 else 6, 0), sticky="w")

            slider = ctk.CTkSlider(
                f, from_=lo, to=hi,
                height=16,
                button_color=C["accent"],
                button_hover_color="#5a73f5",
                progress_color=C["accent"],
                fg_color=C["surface2"],
                corner_radius=4,
                command=lambda v, k=key, fn=fmt: self._on_slider(k, v, fn),
            )
            slider.grid(row=r, column=1, padx=(0, 6),
                        pady=(10 if r == 0 else 6, 0), sticky="ew")
            self._slider_widgets[key] = slider

            val_lbl = ctk.CTkLabel(
                f, text=fmt(lo),
                font=ctk.CTkFont(size=11, family="Consolas"),
                text_color=C["success"],
                width=60, anchor="e",
            )
            val_lbl.grid(row=r, column=2, padx=(0, 14),
                         pady=(10 if r == 0 else 6, 0))
            self._slider_var_lbls[key] = val_lbl

        ctk.CTkLabel(f, text="", height=6).grid(row=len(SLIDERS), column=0)

    # -- Transport -----------------------------------------------------------

    def _build_transport(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color="transparent")
        f.pack(fill="x", padx=pad, pady=(10, 4))

        self._start_btn = ctk.CTkButton(
            f,
            text="   START",
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color=C["success"],
            hover_color="#08f0b0",
            text_color="#0d1117",
            corner_radius=12,
            height=62,
            command=self._toggle,
        )
        self._start_btn.pack(fill="x")

    # -- Status bar ----------------------------------------------------------

    def _build_statusbar(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color="transparent")
        f.pack(fill="x", padx=pad, pady=(6, 16))
        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        f.grid_columnconfigure(2, weight=1)

        self._lbl_latency = ctk.CTkLabel(
            f, text=f"Latency: {1000*BLOCK_SIZE/SAMPLE_RATE:.1f} ms",
            font=ctk.CTkFont(size=10), text_color=C["dim"], anchor="w"
        )
        self._lbl_latency.grid(row=0, column=0, sticky="w")

        self._lbl_cpu = ctk.CTkLabel(
            f, text="CPU: --",
            font=ctk.CTkFont(size=10), text_color=C["dim"], anchor="center"
        )
        self._lbl_cpu.grid(row=0, column=1)

        self._lbl_xruns = ctk.CTkLabel(
            f, text="Xruns: 0",
            font=ctk.CTkFont(size=10), text_color=C["dim"], anchor="e"
        )
        self._lbl_xruns.grid(row=0, column=2, sticky="e")

    # =======================================================================
    # Meter drawing
    # =======================================================================

    def _draw_meter(self, canvas, db):
        try:
            w = canvas.winfo_width()
        except Exception:
            w = 300
        if w < 4:
            w = 300
        h = 14

        norm = max(0.0, min(1.0, (db + 60) / 60))

        # Background segments
        n_seg = 28
        sw = (w - 2) / n_seg

        canvas.delete("all")
        canvas.configure(bg=C["surface2"])

        for i in range(n_seg):
            x1 = 1 + i * sw + 0.5
            x2 = 1 + (i + 1) * sw - 0.5
            frac = i / n_seg
            if frac < norm:
                if frac < 0.70:
                    col = C["success"]
                elif frac < 0.90:
                    col = C["warn"]
                else:
                    col = C["danger"]
            else:
                col = "#1a2030"
            canvas.create_rectangle(x1, 2, x2, h - 2, fill=col, outline="")

    def _tick_meters(self):
        """50 ms GUI timer: smooth + redraw level meters, update status bar."""
        ALPHA_ATK = 0.75
        ALPHA_REL = 0.12

        raw_in  = self._raw_in.copy()
        raw_out = self._raw_out.copy()

        for ch in range(2):
            for disp, raw in [(self._dsp_in, raw_in), (self._dsp_out, raw_out)]:
                if raw[ch] > disp[ch]:
                    disp[ch] = ALPHA_ATK * disp[ch] + (1 - ALPHA_ATK) * raw[ch]
                else:
                    disp[ch] = ALPHA_REL * disp[ch] + (1 - ALPHA_REL) * raw[ch]

        db_in  = [20 * np.log10(max(1e-7, v)) for v in self._dsp_in]
        db_out = [20 * np.log10(max(1e-7, v)) for v in self._dsp_out]

        pairs = [
            (self._cvs_inL,  self._lbl_inL,  db_in[0]),
            (self._cvs_inR,  self._lbl_inR,  db_in[1]),
            (self._cvs_outL, self._lbl_outL, db_out[0]),
            (self._cvs_outR, self._lbl_outR, db_out[1]),
        ]
        for cvs, lbl, db in pairs:
            self._draw_meter(cvs, db)
            txt = f"{db:+.0f}" if db > -60 else "-inf"
            lbl.configure(text=f"{txt} dB")

        # Status bar update
        if self._running and self._t_start:
            elapsed = time.time() - self._t_start
            if elapsed > 0:
                bps = self._blk_count / elapsed
                rt_cpu = bps * BLOCK_SIZE / SAMPLE_RATE * 100
                self._lbl_cpu.configure(
                    text=f"CPU: {rt_cpu:.0f}%",
                    text_color=C["warn"] if rt_cpu > 50 else C["dim"],
                )
        self._lbl_xruns.configure(
            text=f"Xruns: {self._xruns}",
            text_color=C["danger"] if self._xruns > 0 else C["dim"],
        )

        self.after(50, self._tick_meters)

    # =======================================================================
    # Parameter / preset logic
    # =======================================================================

    def _apply_preset(self, name, animate=True):
        self._preset_name = name
        params = PRESETS[name]
        self._slider_vals = dict(params)

        for key, slider in self._slider_widgets.items():
            if key in params:
                slider.set(params[key])
                fmt = next(fn for lbl, k, lo, hi, fn in SLIDERS if k == key)
                self._slider_var_lbls[key].configure(text=fmt(params[key]))

        # Highlight active preset button
        for n, btn in self._preset_btns.items():
            btn.configure(
                fg_color=C["accent"] if n != name else C["success"],
                text_color="white" if n != name else "#0d1117",
            )

        self._schedule_rebuild()

    def _on_slider(self, key, value, fmt):
        self._slider_vals[key] = value
        self._slider_var_lbls[key].configure(text=fmt(value))
        # Deselect preset buttons (custom config)
        for btn in self._preset_btns.values():
            btn.configure(fg_color=C["accent"], text_color="white")
        self._schedule_rebuild()

    # Mode label shown under the segmented button
    _MODE_DESCS = {
        "headphones":   "Binaural 5.1 HRTF — optimised for headphones",
        "speakers":     "Stereo widening + Haas depth — for speaker playback",
        "surround":     "Virtual 7.1 HRTF with elevation — headphones or speakers",
        "surround_mono":"7.1 HRTF collapsed to mono — works with a single speaker",
    }

    def _on_mode_change(self, value):
        mapping = {
            "Headphones": "headphones",
            "Speakers":   "speakers",
            "Surround":   "surround",
            "Mono":       "surround_mono",
        }
        self._mode = mapping.get(value, "headphones")
        if hasattr(self, "_mode_desc"):
            self._mode_desc.configure(text=self._MODE_DESCS.get(self._mode, ""))
        self._schedule_rebuild()

    def _on_device_change(self, display_name, attr_menu):
        """Map display name back to device index; refresh outputs when input changes."""
        lst      = self._in_list if attr_menu == "_in_dev_menu" else self._out_list
        idx_attr = "_in_dev_idx" if attr_menu == "_in_dev_menu" else "_out_dev_idx"
        for dev_idx, dev_name in lst:
            if dev_name == display_name:
                setattr(self, idx_attr, dev_idx)
                break
        # When input changes, rebuild output list to match its host API
        if attr_menu == "_in_dev_menu":
            self._refresh_out_menu()

    def _build_preset(self) -> dict:
        """Merge active preset + slider overrides + mode into a final preset dict."""
        # Surround modes use headphones base (detailed HRTF-oriented settings).
        # Mono surround uses speakers base (physical playback oriented).
        use_hp = self._mode in ("headphones", "surround")
        base = dict(HEADPHONES_PRESET if use_hp else SPEAKERS_PRESET)
        base["mode"] = self._mode
        base.update(self._slider_vals)
        return base

    def _schedule_rebuild(self):
        """Debounce chain rebuild: wait 250 ms after last parameter change."""
        if self._rb_timer:
            self._rb_timer.cancel()
        self._rb_timer = threading.Timer(0.25, self._rebuild_chain)
        self._rb_timer.start()

    def _rebuild_chain(self):
        """Build a new TheaterChain with current parameters (runs in timer thread)."""
        try:
            preset    = self._build_preset()
            new_chain = TheaterChain(fs=SAMPLE_RATE, preset=preset)
            self._chain = new_chain   # Python assignment is atomic under GIL
        except Exception as e:
            print(f"[rebuild] {e}")

    # =======================================================================
    # Transport: start / stop
    # =======================================================================

    def _toggle(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        preset = self._build_preset()
        self._chain  = TheaterChain(fs=SAMPLE_RATE, preset=preset)
        self._xruns  = 0
        self._blk_count = 0
        self._t_start   = time.time()

        in_dev  = self._in_dev_idx
        out_dev = self._out_dev_idx

        try:
            in_info  = sd.query_devices(in_dev,  "input")
            in_ch    = min(in_info["max_input_channels"], 2)
        except Exception:
            in_ch = 2

        try:
            out_info = sd.query_devices(out_dev, "output")
            out_ch   = min(out_info["max_output_channels"], 8)
            out_ch   = max(out_ch, 2)
        except Exception:
            out_ch = 2

        try:
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=(in_dev, out_dev),
                channels=(in_ch, out_ch),
                dtype="float32",
                callback=self._audio_cb,
                latency="low",
            )
            self._stream.start()
            self._running = True
            self._set_running_ui(True)
        except Exception as e:
            self._show_error(f"Could not open audio stream:\n{e}\n\nCheck device selection.")

    def _stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._set_running_ui(False)
        # Decay meters to silence
        self._raw_in[:] = 0
        self._raw_out[:] = 0

    def _set_running_ui(self, running: bool):
        if running:
            self._start_btn.configure(
                text="   STOP",
                fg_color=C["danger"],
                hover_color="#f56070",
                text_color="white",
            )
            self._status_badge.configure(text="  RUNNING", text_color=C["success"])
        else:
            self._start_btn.configure(
                text="   START",
                fg_color=C["success"],
                hover_color="#08f0b0",
                text_color="#0d1117",
            )
            self._status_badge.configure(text="  STOPPED", text_color=C["danger"])
            self._lbl_cpu.configure(text="CPU: --", text_color=C["dim"])

    # =======================================================================
    # Audio callback (runs in sounddevice thread)
    # =======================================================================

    def _audio_cb(self, indata, outdata, frames, time_info, status):
        if status:
            self._xruns += 1

        chain = self._chain   # atomic read (GIL)
        if chain is None:
            outdata[:] = 0.0
            return

        try:
            block = np.ascontiguousarray(
                indata[:, :2] if indata.shape[1] >= 2
                else np.column_stack([indata[:, 0], indata[:, 0]]),
                dtype=np.float32,
            )

            # Measure input RMS
            sq = block * block
            self._raw_in = np.sqrt(np.array([sq[:, 0].mean(), sq[:, 1].mean()],
                                            dtype=np.float32))

            result = chain.process(block)

            # Measure output RMS
            sq2 = result * result
            self._raw_out = np.sqrt(np.array([sq2[:, 0].mean(), sq2[:, 1].mean()],
                                             dtype=np.float32))

            out_ch = min(outdata.shape[1], 2)
            outdata[:, :out_ch] = result[:, :out_ch]
            if outdata.shape[1] > out_ch:
                outdata[:, out_ch:] = 0.0

            self._blk_count += 1

        except Exception as exc:
            outdata[:] = 0.0
            print(f"[audio] {exc}")

    # =======================================================================
    # Error dialog
    # =======================================================================

    def _show_error(self, message: str):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Error")
        dlg.geometry("400x200")
        dlg.resizable(False, False)
        dlg.grab_set()

        ctk.CTkLabel(
            dlg, text=message,
            font=ctk.CTkFont(size=12),
            wraplength=360,
        ).pack(padx=20, pady=30)

        ctk.CTkButton(
            dlg, text="OK", width=100,
            fg_color=C["accent"],
            command=dlg.destroy,
        ).pack()

    # =======================================================================
    # Shutdown
    # =======================================================================

    def _on_close(self):
        if self._rb_timer:
            self._rb_timer.cancel()
        self._stop()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ModAudioApp()
    app.mainloop()

#!/usr/bin/env python3
"""
ModAudio - Theater Experience
GUI Application
Run with:  python app.py
"""

import sys, os, threading, time

_IS_WIN = sys.platform == "win32"
_IS_MAC = sys.platform == "darwin"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import customtkinter as ctk
except ImportError:
    sys.exit("customtkinter not found. Run:  pip install customtkinter")

import sounddevice as sd

from config      import HEADPHONES_PRESET, SPEAKERS_PRESET, SAMPLE_RATE, BLOCK_SIZE
from audio_io    import find_default_devices
from dsp         import TheaterChain
from audio_multi import MultiDeviceStream, MultiSpeakerStreamN, is_bluetooth_device
from room_canvas_3d import Room3DCanvas, SPEAKER_LAYOUTS_3D, CHANNEL_DIRECTIONS, DIRECTION_TO_SPEAKER
try:
    import virtual_device as _vdev
    _HAS_VDEV = True
except ImportError:
    _vdev = None        # type: ignore
    _HAS_VDEV = False


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ---------------------------------------------------------------------------
# Presets  (merged into the base headphones / speakers preset)
# ---------------------------------------------------------------------------

PRESETS = {
    "Cinema": {
        "rt60": 1.3,  "rt60_hf": 0.65,
        "reverb_predelay_ms": 22.0,  "reverb_mix": 0.25,  "early_ref_mix": 0.45,
        "stereo_width": 2.0,  "surround_level": 0.78,  "lfe_level": 0.88,
        "center_level": 0.88,  "rear_level": 0.65,
        "bass_boost_db": 6.0,  "sub_bass_db": 4.5,
        "bass_harm_drive": 2.8,  "bass_harm_level": 0.50,
        "air_exciter_level": 0.18,
        "mb_compress_drive": 1.6,  "transient_amount": 0.55,
        "output_gain_db": 4.5,    # +4.5 dB compensates for VBAP normalisation attenuation
    },
    # IMAX: massive room, deep sub extension, wide dynamic range, punchy transients.
    # Tuned to the characteristic IMAX experience: overwhelming scale and bass impact.
    "IMAX": {
        "rt60": 1.9,  "rt60_hf": 0.80,
        "reverb_predelay_ms": 32.0,  "reverb_mix": 0.36,  "early_ref_mix": 0.60,
        "stereo_width": 2.5,  "surround_level": 0.90,  "lfe_level": 1.00,
        "center_level": 0.85,  "rear_level": 0.78,
        "bass_boost_db": 10.0,  "sub_bass_db": 7.0,
        "bass_harm_drive": 3.8,  "bass_harm_level": 0.72,
        "air_exciter_level": 0.24,
        "mb_compress_drive": 2.0,  "transient_amount": 0.85,
        "output_gain_db": 3.0,    # slightly lower than Cinema — IMAX bass needs headroom
    },
    # Dolby: precision-tuned, tighter room, less aggressive dynamics.
    # Optimises for articulate surround localisation and clean dialog intelligibility.
    # Atmos-style height rendering is enabled; if no physical height speaker is
    # placed in the room canvas, height sends gracefully fall back to ground
    # speakers via the same VBAP engine Dolby's own renderer uses.
    "Dolby": {
        "rt60": 0.95,  "rt60_hf": 0.50,
        "reverb_predelay_ms": 18.0,  "reverb_mix": 0.16,  "early_ref_mix": 0.34,
        "stereo_width": 1.9,  "surround_level": 0.68,  "lfe_level": 0.82,
        "center_level": 0.92,  "rear_level": 0.58,
        "bass_boost_db": 4.0,  "sub_bass_db": 3.0,
        "bass_harm_drive": 2.0,  "bass_harm_level": 0.34,
        "air_exciter_level": 0.12,
        "mb_compress_drive": 1.2,  "transient_amount": 0.38,
        "output_gain_db": 4.5,
        "atmos_mode": True,   "height_level": 0.45,
    },
    "Home": {
        "rt60": 0.8,  "rt60_hf": 0.45,
        "reverb_predelay_ms": 15.0,  "reverb_mix": 0.12,  "early_ref_mix": 0.28,
        "stereo_width": 1.5,  "surround_level": 0.55,  "lfe_level": 0.70,
        "center_level": 0.85,  "rear_level": 0.48,
        "bass_boost_db": 3.0,  "sub_bass_db": 2.5,
        "bass_harm_drive": 1.8,  "bass_harm_level": 0.28,
        "air_exciter_level": 0.10,
        "mb_compress_drive": 1.1,  "transient_amount": 0.25,
        "output_gain_db": 4.5,
    },
}

# Slider definitions: (label, param_key, min, max, format_fn)
SLIDERS = [
    ("Reverb",   "rt60",             0.3,  2.5, lambda v: f"{v:.1f} s"),
    ("Width",    "stereo_width",     1.0,  2.8, lambda v: f"{v:.1f}\u00d7"),
    ("Bass",     "bass_boost_db",    0.0, 12.0, lambda v: f"+{v:.0f} dB"),
    ("Dynamics", "mb_compress_drive",1.0,  2.2, lambda v: f"{v:.1f}"),
]

# Color palette — Apple-inspired dark-mode system
C = {
    "accent":    "#0A84FF",  # system blue (dark mode)
    "success":   "#30D158",  # system green
    "danger":    "#FF453A",  # system red
    "warn":      "#FFD60A",  # system yellow
    "surface":   "#1C1C1E",  # systemGray6 dark
    "surface2":  "#111113",  # deeper surface for window bg
    "dim":       "#8E8E93",  # systemGray
    "text":      "#F5F5F7",  # Apple-website near-white
}

# Corner radius — squircle-adjacent. tkinter can't render true squircles,
# but larger radii read the same at a glance.
RADIUS    = 16
RADIUS_SM = 12
RADIUS_LG = 20

# Font stack. SF Pro isn't redistributable — we fall back through Inter,
# Segoe UI Variable (Windows 11 default), Helvetica Neue, Arial.
_APPLE_FONT_FAMILY = "SF Pro Display"
_APPLE_FONT_FALLBACK = ("SF Pro Text", "Inter", "Segoe UI Variable",
                        "Segoe UI", "Helvetica Neue", "Arial")

def _pick_font_family():
    """Pick the first installed font from the Apple stack."""
    try:
        import tkinter.font as _tkfont
        installed = set(_tkfont.families())
        for fam in (_APPLE_FONT_FAMILY, *_APPLE_FONT_FALLBACK):
            if fam in installed:
                return fam
    except Exception:
        pass
    return "Segoe UI" if _IS_WIN else ("Helvetica Neue" if _IS_MAC else "Arial")

_FONT_FAMILY = None  # resolved lazily after Tk root exists

def apple_font(size=13, weight="normal"):
    """ctk.CTkFont with Apple-style family. Call after root window exists."""
    global _FONT_FAMILY
    if _FONT_FAMILY is None:
        _FONT_FAMILY = _pick_font_family()
    return ctk.CTkFont(family=_FONT_FAMILY, size=size, weight=weight)

# Monkey-patch CTkFont so every widget picks up the Apple family unless
# an explicit family is passed (e.g. "Consolas" for monospaced readouts).
_CTkFont_orig = ctk.CTkFont
def _CTkFont_apple(*args, **kwargs):
    if "family" not in kwargs:
        global _FONT_FAMILY
        if _FONT_FAMILY is None:
            _FONT_FAMILY = _pick_font_family()
        kwargs["family"] = _FONT_FAMILY
    return _CTkFont_orig(*args, **kwargs)
ctk.CTkFont = _CTkFont_apple


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class ModAudioApp(ctk.CTk):

    W, H = 660, 900

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

        # Multi-speaker state
        self._ms_running        = False
        self._ms_stream         = None
        self._ms_front_idx      = 0
        self._ms_rear_idx       = 0
        self._ms_bt_delay       = 150.0
        self._ms_swap_rear_lr   = True   # rear speaker faces listener by default
        self._ms_mode           = "loopback"    # "loopback" | "rear_only" | "dual"
        self._ms_front_gain     = 1.0
        self._ms_rear_gain      = 1.0
        self._ms_bass_priority  = "equal"       # "equal" | "front" | "rear"
        self._master_gain       = 1.0
        self._ms_in_idx         = 0
        self._ms_dsp_front      = np.zeros(2, dtype=np.float32)
        self._ms_dsp_rear       = np.zeros(2, dtype=np.float32)
        self._ms_preset_name    = "Cinema"      # theater mode for multi-speaker
        self._ms_rear_az_deg    = 150.0         # rear speaker azimuth (90–170°)
        self._ms_acoustic_delay = 0.0           # speaker distance delay (ms)

        # N-speaker room canvas state
        # Each entry: {sid, label, device_idx, device_label, azimuth, distance}
        self._ms_speakers: list[dict] = []
        self._room_canvas: "Room3DCanvas | None" = None
        self._ms_selected_sid: "int | None" = None
        # Per-speaker RMS smoothing (sid → np.ndarray([L,R]))
        self._ms_dsp_per_spk: dict = {}
        self._ms_room_w = 6.0   # metres
        self._ms_room_d = 5.0
        self._ms_room_h = 3.0
        # Dynamic output device rows (each: dict with frame/dev_menu/dir_menu/etc.)
        self._ms_output_rows: list[dict] = []

        # -- Discover audio devices
        self._devs     = sd.query_devices()
        self._hostapis = sd.query_hostapis()
        self._build_device_lists()

        # Multi-speaker output device list — prefer the native low-latency API
        # for the current platform (WASAPI on Windows, Core Audio on macOS).
        # MME/DS on Windows and non-preferred APIs introduce buffer mismatches.
        _pref_api = self._preferred_api_label()
        self._all_out_list = []
        for i, d in enumerate(self._devs):
            if d["max_output_channels"] >= 1 and self._hostapi_label(d["hostapi"]) == _pref_api:
                label = f"{d['name'][:48]}  [{_pref_api}]"
                self._all_out_list.append((i, label))
        if not self._all_out_list:
            # Fallback: show all output devices
            for i, d in enumerate(self._devs):
                if d["max_output_channels"] >= 1:
                    tag   = self._hostapi_label(d["hostapi"])
                    label = f"{d['name'][:44]}  [{tag}]"
                    self._all_out_list.append((i, label))
        if not self._all_out_list:
            self._all_out_list = [(0, "No output devices")]

        # Multi-speaker capture input list — preferred API + known loopback devices.
        _loopback_kw = ("stereo mix", "what u hear", "wave out mix",
                        "cable output", "vb-audio", "vb-cable", "loopback",
                        "virtual audio driver", "modaudio surround",
                        "blackhole", "soundflower")
        self._ms_all_in_list = []
        for i, d in enumerate(self._devs):
            if d["max_input_channels"] >= 1 and self._hostapi_label(d["hostapi"]) == _pref_api:
                label = f"{d['name'][:48]}  [{_pref_api}]"
                self._ms_all_in_list.append((i, label))
        # Also include known loopback/virtual capture devices regardless of host API
        for i, d in enumerate(self._devs):
            if d["max_input_channels"] >= 1:
                nl = d["name"].lower()
                if any(k in nl for k in _loopback_kw):
                    tag   = self._hostapi_label(d["hostapi"])
                    label = f"{d['name'][:44]}  [{tag}]"
                    if not any(idx == i for idx, _ in self._ms_all_in_list):
                        self._ms_all_in_list.append((i, label))
        if not self._ms_all_in_list:
            self._ms_all_in_list = [(0, "No input devices")]

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

        # Multi-speaker initial device indices
        if self._all_out_list:
            self._ms_front_idx = (auto_out if auto_out is not None and
                                  any(i == auto_out for i, _ in self._all_out_list)
                                  else self._all_out_list[0][0])
            self._ms_rear_idx  = self._all_out_list[0][0]
        if self._ms_all_in_list:
            self._ms_in_idx = (auto_in if auto_in is not None and
                               any(i == auto_in for i, _ in self._ms_all_in_list)
                               else self._ms_all_in_list[0][0])

        # -- Build and start
        self._build_ui()
        self._apply_preset("Cinema", animate=False)
        self._tick_meters()

    # =======================================================================
    # Device discovery helpers
    # =======================================================================

    # Keywords that identify system-audio loopback / capture sources.
    # Covers Windows (Stereo Mix, VB-Cable) and macOS (BlackHole, Soundflower, VB-Cable).
    _LOOPBACK_KW = ("stereo mix", "what u hear", "wave out mix",
                    "loopback", "cable output", "vb-audio",
                    "blackhole", "soundflower", "vb-cable")

    def _dev_hostapi(self, device_idx: int) -> int:
        """Return the host-API index for a given device index."""
        try:
            return int(self._devs[device_idx]["hostapi"])
        except (IndexError, KeyError):
            return -1

    def _hostapi_label(self, ha_idx: int) -> str:
        """Short host-API tag, e.g. 'WASAPI', 'MME', 'CoreAudio'."""
        try:
            name = self._hostapis[ha_idx]["name"]
            if "WASAPI" in name:      return "WASAPI"
            if "MME"   in name:      return "MME"
            if "DirectSound" in name: return "DS"
            if "WDM"   in name:      return "WDM"
            if "Core Audio" in name:  return "CoreAudio"
            if "ALSA"  in name:      return "ALSA"
            if "PulseAudio" in name:  return "Pulse"
            return name[:8]
        except (IndexError, KeyError):
            return "?"

    def _preferred_api_label(self) -> str:
        """Return the best host-API label for the current platform."""
        if _IS_WIN:
            return "WASAPI"
        if _IS_MAC:
            return "CoreAudio"
        # Linux: prefer PulseAudio, fall back to ALSA
        for api in self._hostapis:
            if "PulseAudio" in api.get("name", ""):
                return "Pulse"
        return "ALSA"

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

        # Global header (above tabs, always visible)
        self._build_header(self, pad)

        # Tab view
        self._tabs = ctk.CTkTabview(
            self,
            fg_color="transparent",
            segmented_button_fg_color=C["surface"],
            segmented_button_selected_color=C["accent"],
            segmented_button_selected_hover_color="#5a73f5",
            segmented_button_unselected_color=C["surface"],
            segmented_button_unselected_hover_color=C["surface2"],
            text_color=C["text"],
        )
        self._tabs.pack(fill="both", expand=True, padx=0, pady=0)
        self._tabs.add("Theater")
        self._tabs.add("Multi-Speaker")

        # ---- Theater tab: scrollable frame ----------------------------------
        self._root_frame = ctk.CTkScrollableFrame(
            self._tabs.tab("Theater"),
            fg_color="transparent",
            scrollbar_button_color=C["surface"],
            scrollbar_button_hover_color=C["dim"],
        )
        self._root_frame.pack(fill="both", expand=True, padx=0, pady=0)

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
        self._build_section("VOLUME", pad)
        self._build_volume(pad)
        self._build_transport(pad)
        self._build_statusbar(pad)

        # ---- Multi-Speaker tab ----------------------------------------------
        self._ms_frame = ctk.CTkScrollableFrame(
            self._tabs.tab("Multi-Speaker"),
            fg_color="transparent",
            scrollbar_button_color=C["surface"],
            scrollbar_button_hover_color=C["dim"],
        )
        self._ms_frame.pack(fill="both", expand=True, padx=0, pady=0)
        self._build_multi_tab(pad)

    # -- Header --------------------------------------------------------------

    def _build_header(self, parent, pad):
        f = ctk.CTkFrame(parent, fg_color=C["surface2"], corner_radius=0,
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
            corner_radius=12,
            padx=10, pady=4,
        )
        self._status_badge.place(relx=1.0, x=-pad, y=20, anchor="ne")

    # -- Section label -------------------------------------------------------

    def _build_section(self, title, pad, parent=None):
        ctk.CTkLabel(
            parent if parent is not None else self._root_frame,
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
                corner_radius=12,
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
            corner_radius=12,
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
                         corner_radius=16)
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
                corner_radius=12,
                height=32,
                command=lambda v, a=attr_menu: self._on_device_change(v, a),
            )
            menu.set(cur_name)
            menu.grid(row=r, column=1, padx=(0, 14), pady=8, sticky="ew")
            setattr(self, attr_menu, menu)

        f.grid_columnconfigure(1, weight=1)

    # -- Level meters --------------------------------------------------------

    def _build_meters(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color=C["surface"], corner_radius=16)
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
        f = ctk.CTkFrame(self._root_frame, fg_color=C["surface"], corner_radius=16)
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

    # -- Volume --------------------------------------------------------------

    def _build_volume(self, pad):
        f = ctk.CTkFrame(self._root_frame, fg_color=C["surface"], corner_radius=16)
        f.pack(fill="x", padx=pad, pady=(0, 4))
        f.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            f, text="Master",
            font=ctk.CTkFont(size=12), text_color=C["text"],
            width=60, anchor="w",
        ).grid(row=0, column=0, padx=(14, 4), pady=(10, 10), sticky="w")

        self._vol_slider = ctk.CTkSlider(
            f, from_=-30.0, to=6.0,
            height=16,
            button_color=C["accent"], button_hover_color="#5a73f5",
            progress_color=C["accent"], fg_color=C["surface2"], corner_radius=4,
            command=self._on_volume_change,
        )
        self._vol_slider.set(0.0)
        self._vol_slider.grid(row=0, column=1, padx=(0, 6), pady=(10, 10), sticky="ew")

        self._vol_lbl = ctk.CTkLabel(
            f, text="0 dB",
            font=ctk.CTkFont(size=11, family="Consolas"),
            text_color=C["success"], width=54, anchor="e",
        )
        self._vol_lbl.grid(row=0, column=2, padx=(0, 14), pady=(10, 10))

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
            corner_radius=18,
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

    # -- Multi-Speaker tab ---------------------------------------------------

    def _build_multi_tab(self, pad):
        """Build Multi-Speaker tab UI into self._ms_frame."""
        ms = self._ms_frame

        # --- VIRTUAL DEVICE SETUP --------------------------------------------
        _vdev_section_title = ("MODAUDIO SURROUND SPEAKER" if _IS_WIN
                               else "AUDIO CAPTURE SETUP")
        self._build_section(_vdev_section_title, pad, parent=ms)
        f_vs = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_vs.pack(fill="x", padx=pad, pady=(0, 4))
        f_vs.grid_columnconfigure(1, weight=1)

        # Status row
        ctk.CTkLabel(
            f_vs, text="Status",
            font=ctk.CTkFont(size=12), text_color=C["dim"],
            width=68, anchor="w",
        ).grid(row=0, column=0, padx=(14, 6), pady=(12, 4), sticky="w")
        self._ms_vdev_status_lbl = ctk.CTkLabel(
            f_vs, text="Checking…",
            font=ctk.CTkFont(size=11), text_color=C["dim"], anchor="w",
        )
        self._ms_vdev_status_lbl.grid(row=0, column=1, padx=(0, 14),
                                      pady=(12, 4), sticky="ew")

        # Hint / instructions
        if _IS_WIN:
            _initial_hint = (
                "Install the free ModAudio Surround virtual speaker so Windows\n"
                "can route all audio through ModAudio.  One-time setup, no reboot.\n"
                "The driver is MIT-licensed and production-signed (no test mode)."
            )
        else:
            _initial_hint = (
                "Install a virtual loopback device to capture system audio:\n"
                "  • VB-Cable for Mac   vb-audio.com/Cable/\n"
                "  • BlackHole (free)   github.com/ExistentialAudio/BlackHole\n"
                "Set it as your system audio output, then select it as Capture below."
            )
        self._ms_vdev_hint = ctk.CTkLabel(
            f_vs,
            text=_initial_hint,
            font=ctk.CTkFont(size=10),
            text_color=C["dim"],
            anchor="w", justify="left", wraplength=460,
        )
        self._ms_vdev_hint.grid(row=1, column=0, columnspan=2,
                                padx=14, pady=(0, 4), sticky="ew")

        # Progress bar (shown during installation)
        self._ms_install_progress = ctk.CTkProgressBar(
            f_vs, height=6, corner_radius=3,
            progress_color=C["accent"], fg_color=C["surface2"],
        )
        self._ms_install_progress.set(0)
        self._ms_install_progress.grid(row=2, column=0, columnspan=2,
                                       padx=14, pady=(0, 4), sticky="ew")
        self._ms_install_progress.grid_remove()

        self._ms_install_status_lbl = ctk.CTkLabel(
            f_vs, text="",
            font=ctk.CTkFont(size=10), text_color=C["dim"], anchor="w",
        )
        self._ms_install_status_lbl.grid(row=3, column=0, columnspan=2,
                                         padx=14, pady=(0, 4), sticky="ew")
        self._ms_install_status_lbl.grid_remove()

        # Main action buttons
        f_vbtn = ctk.CTkFrame(f_vs, fg_color="transparent")
        f_vbtn.grid(row=4, column=0, columnspan=2, padx=14, pady=(0, 8),
                    sticky="ew")
        f_vbtn.grid_columnconfigure((0, 1, 2), weight=1)

        # Primary: Install virtual driver (Windows only)
        self._ms_btn_install_driver = ctk.CTkButton(
            f_vbtn, text="Install Virtual Speaker",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=C["accent"], hover_color="#5a73f5",
            text_color="white", corner_radius=12, height=32,
            command=self._on_ms_install_driver,
        )
        self._ms_btn_install_driver.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        # Set as Windows default output (Windows only)
        self._ms_btn_set_default = ctk.CTkButton(
            f_vbtn, text="Set as Default",
            font=ctk.CTkFont(size=11),
            fg_color=C["surface2"], hover_color=C["surface"],
            border_color=C["dim"], border_width=1,
            text_color=C["dim"], corner_radius=12, height=32,
            command=self._on_ms_set_default_device,
            state="disabled",
        )
        self._ms_btn_set_default.grid(row=0, column=1, padx=4, sticky="ew")

        # Open system audio settings (cross-platform)
        self._ms_btn_sound_settings = ctk.CTkButton(
            f_vbtn, text="Sound Settings",
            font=ctk.CTkFont(size=11),
            fg_color=C["surface2"], hover_color=C["surface"],
            border_color=C["accent"], border_width=1,
            text_color=C["text"], corner_radius=12, height=32,
            command=self._on_ms_open_sound_settings,
        )
        self._ms_btn_sound_settings.grid(row=0, column=2, padx=(4, 0), sticky="ew")

        # On non-Windows: hide the Windows-specific buttons and expand Sound Settings
        if not _IS_WIN:
            self._ms_btn_install_driver.grid_remove()
            self._ms_btn_set_default.grid_remove()
            self._ms_btn_sound_settings.grid_configure(column=0, columnspan=3, padx=0)

        # Full-width: Auto-configure Full Control mode
        self._ms_btn_autoconfigure = ctk.CTkButton(
            f_vs, text="▶  Auto-Configure Full Control (Both Speakers)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=C["success"], hover_color="#08f0b0",
            text_color="#0d1117", corner_radius=12, height=36,
            command=self._on_ms_autoconfigure,
        )
        self._ms_btn_autoconfigure.grid(row=5, column=0, columnspan=2,
                                        padx=14, pady=(0, 12), sticky="ew")
        self._ms_btn_autoconfigure.grid_remove()   # shown after driver installed

        # Populate status asynchronously (avoids blocking startup)
        self.after(200, self._ms_refresh_vdev_status)

        # --- SETUP MODE ------------------------------------------------------
        self._build_section("SETUP MODE", pad, parent=ms)
        f_modesel = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_modesel.pack(fill="x", padx=pad, pady=(0, 4))

        self._ms_mode_seg = ctk.CTkSegmentedButton(
            f_modesel,
            values=["Loopback", "Full Control"],
            font=ctk.CTkFont(size=12),
            height=36,
            selected_color=C["accent"],
            selected_hover_color="#5a73f5",
            corner_radius=12,
            command=self._on_ms_mode_change,
        )
        self._ms_mode_seg.set("Loopback")
        self._ms_mode_seg.pack(fill="x", padx=14, pady=(10, 6))

        self._ms_mode_desc = ctk.CTkLabel(
            f_modesel,
            text=self._MS_MODE_DESCS["loopback"],
            font=ctk.CTkFont(size=10),
            text_color=C["dim"],
            anchor="w", justify="left", wraplength=460,
        )
        self._ms_mode_desc.pack(fill="x", padx=14, pady=(0, 10))

        # --- THEATER MODE ----------------------------------------------------
        # Preset selector specific to the multi-speaker chain.
        self._build_section("THEATER MODE", pad, parent=ms)
        f_tm = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_tm.pack(fill="x", padx=pad, pady=(0, 4))
        f_tm.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._ms_theater_btns = {}
        _theater_modes = [
            ("Cinema",  "Standard multiplex experience.\nBalanced reverb and dynamics."),
            ("IMAX",    "Massive room, extreme bass depth,\nvery wide surround envelope."),
            ("Dolby",   "Precision-tuned, clean dynamics,\nsharp localisation cues."),
            ("Home",    "Intimate scale, gentle reverb,\nsuitable for small rooms."),
        ]
        for col, (nm, tip) in enumerate(_theater_modes):
            is_sel = (nm == self._ms_preset_name)
            btn = ctk.CTkButton(
                f_tm, text=nm,
                font=ctk.CTkFont(size=11, weight="bold" if is_sel else "normal"),
                fg_color=C["accent"] if is_sel else C["surface2"],
                hover_color="#5a73f5",
                border_color=C["accent"],
                border_width=1 if not is_sel else 0,
                text_color="white" if is_sel else C["text"],
                corner_radius=12, height=32,
                command=lambda n=nm: self._on_ms_theater_preset(n),
            )
            btn.grid(row=0, column=col,
                     padx=(4, 4), pady=(10, 4), sticky="ew")
            self._ms_theater_btns[nm] = btn

        self._ms_theater_desc = ctk.CTkLabel(
            f_tm, text=_theater_modes[0][1],
            font=ctk.CTkFont(size=10), text_color=C["dim"],
            anchor="w", justify="left", wraplength=460,
        )
        self._ms_theater_desc.grid(row=1, column=0, columnspan=4,
                                   padx=14, pady=(0, 10), sticky="ew")

        # --- OUTPUT DEVICES --------------------------------------------------
        self._build_section("OUTPUT DEVICES", pad, parent=ms)
        f_dev = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_dev.pack(fill="x", padx=pad, pady=(0, 4))
        f_dev.grid_columnconfigure(1, weight=1)

        # Row 0: Capture device (Full Control mode) OR loopback info (Loopback mode)
        self._ms_cap_row_label = ctk.CTkLabel(
            f_dev, text="Capture",
            font=ctk.CTkFont(size=12), text_color=C["dim"],
            width=68, anchor="w",
        )
        self._ms_cap_row_label.grid(row=0, column=0, padx=(14, 6), pady=8, sticky="w")
        ms_in_names = [label for _, label in self._ms_all_in_list]
        ms_in_cur   = next((l for i, l in self._ms_all_in_list if i == self._ms_in_idx),
                           ms_in_names[0] if ms_in_names else "")
        self._ms_cap_menu = ctk.CTkOptionMenu(
            f_dev,
            values=ms_in_names if ms_in_names else ["No input devices"],
            font=ctk.CTkFont(size=11),
            fg_color=C["surface2"],
            button_color=C["accent"],
            button_hover_color="#5a73f5",
            dropdown_fg_color=C["surface2"],
            dropdown_hover_color=C["surface"],
            corner_radius=12, height=32,
            command=self._on_ms_cap_change,
        )
        self._ms_cap_menu.set(ms_in_cur)
        self._ms_cap_menu.grid(row=0, column=1, columnspan=2, padx=(0, 14),
                               pady=8, sticky="ew")
        # Loopback info label shown in Loopback mode instead of capture dropdown
        self._ms_loopback_info = ctk.CTkLabel(
            f_dev,
            text="Automatic — loopback of Front Out  (no setup required)",
            font=ctk.CTkFont(size=11),
            text_color=C["success"],
            anchor="w",
        )
        self._ms_loopback_info.grid(row=0, column=1, columnspan=2, padx=(0, 14),
                                    pady=8, sticky="ew")

        # Dual-speaker note (shown only in dual mode)
        _dual_note_txt = (
            "Set system default output to VB-Cable Input (or use 'Set as Default' above)."
            if _IS_WIN else
            "Set system audio output to your VB-Cable / BlackHole loopback device."
        )
        self._ms_dual_note = ctk.CTkLabel(
            f_dev,
            text=_dual_note_txt,
            font=ctk.CTkFont(size=10),
            text_color=C["warn"],
            anchor="w", wraplength=420,
        )
        self._ms_dual_note.grid(row=1, column=0, columnspan=2,
                                padx=14, pady=(0, 4), sticky="ew")
        self._ms_dual_note.grid_remove()   # hidden by default (rear_only)

        # Dynamic output device rows
        self._ms_bass_btns = {}   # kept for backward compat
        f_dev.grid_columnconfigure(2, weight=0)

        # Scrollable container for dynamic output rows
        self._ms_out_rows_frame = ctk.CTkFrame(f_dev, fg_color="transparent")
        self._ms_out_rows_frame.grid(row=2, column=0, columnspan=3,
                                     padx=8, pady=(4, 0), sticky="ew")
        self._ms_out_rows_frame.grid_columnconfigure(0, weight=1)

        # "+ Add Device" button row
        f_add_out = ctk.CTkFrame(f_dev, fg_color="transparent")
        f_add_out.grid(row=3, column=0, columnspan=3,
                       padx=8, pady=(2, 8), sticky="w")
        ctk.CTkButton(
            f_add_out, text="+ Add Device",
            font=ctk.CTkFont(size=10, weight="bold"),
            height=26, width=110,
            fg_color=C["accent"], hover_color="#5a73f5",
            text_color="white", corner_radius=6,
            command=self._ms_add_output_row,
        ).pack(side="left")

        # Seed with default Front L/R and Rear L/R rows
        _init_front_name = next((l for i, l in self._all_out_list
                                 if i == self._ms_front_idx), "")
        _init_rear_name  = next((l for i, l in self._all_out_list
                                 if i == self._ms_rear_idx), "")
        self._ms_add_output_row(device_name=_init_front_name, direction="Front L/R")
        self._ms_add_output_row(device_name=_init_rear_name,  direction="Rear L/R")

        # --- BLUETOOTH DELAY -------------------------------------------------
        self._build_section("BLUETOOTH DELAY", pad, parent=ms)
        f_bt = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_bt.pack(fill="x", padx=pad, pady=(0, 4))
        f_bt.grid_columnconfigure(1, weight=1)

        for r, (side, attr_lbl) in enumerate([
            ("Front", "_ms_lbl_front_bt"),
            ("Rear",  "_ms_lbl_rear_bt"),
        ]):
            ctk.CTkLabel(
                f_bt, text=side,
                font=ctk.CTkFont(size=12), text_color=C["dim"],
                width=46, anchor="w",
            ).grid(row=r, column=0, padx=(14, 6), pady=(10 if r == 0 else 4, 4), sticky="w")
            lbl = ctk.CTkLabel(
                f_bt, text="—",
                font=ctk.CTkFont(size=11), text_color=C["dim"], anchor="w",
            )
            lbl.grid(row=r, column=1, padx=(0, 14), pady=(10 if r == 0 else 4, 4), sticky="ew")
            setattr(self, attr_lbl, lbl)

        # Delay slider row
        ctk.CTkLabel(
            f_bt, text="Delay",
            font=ctk.CTkFont(size=12), text_color=C["text"],
            width=46, anchor="w",
        ).grid(row=2, column=0, padx=(14, 6), pady=(4, 6), sticky="w")

        f_sl_row = ctk.CTkFrame(f_bt, fg_color="transparent")
        f_sl_row.grid(row=2, column=1, padx=(0, 14), pady=(4, 6), sticky="ew")
        f_sl_row.grid_columnconfigure(0, weight=1)
        f_sl_row.grid_columnconfigure(1, weight=0)
        f_sl_row.grid_columnconfigure(2, weight=0)

        self._ms_bt_slider = ctk.CTkSlider(
            f_sl_row, from_=0.0, to=300.0, height=16,
            button_color=C["accent"], button_hover_color="#5a73f5",
            progress_color=C["accent"], fg_color=C["surface2"], corner_radius=4,
            command=self._on_ms_bt_slider,
        )
        self._ms_bt_slider.set(self._ms_bt_delay)
        self._ms_bt_slider.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._ms_lbl_bt_delay = ctk.CTkLabel(
            f_sl_row, text=f"{self._ms_bt_delay:.0f} ms",
            font=ctk.CTkFont(size=11, family="Consolas"),
            text_color=C["success"], width=54, anchor="e",
        )
        self._ms_lbl_bt_delay.grid(row=0, column=1, padx=(0, 6))

        self._ms_btn_calibrate = ctk.CTkButton(
            f_sl_row, text="Auto-Calibrate",
            font=ctk.CTkFont(size=11),
            fg_color=C["surface2"],
            hover_color=C["surface"],
            border_color=C["accent"],
            border_width=1,
            text_color=C["text"],
            corner_radius=12, height=28, width=112,
            command=self._on_ms_calibrate,
        )
        self._ms_btn_calibrate.grid(row=0, column=2)

        # Calibration status / hint
        self._ms_lbl_calib_status = ctk.CTkLabel(
            f_bt, text="Press Auto-Calibrate to measure Bluetooth latency automatically",
            font=ctk.CTkFont(size=10), text_color=C["dim"],
            anchor="w", wraplength=460,
        )
        self._ms_lbl_calib_status.grid(row=3, column=0, columnspan=2,
                                       padx=14, pady=(2, 4), sticky="ew")

        # Compensation direction info
        self._ms_lbl_comp_dir = ctk.CTkLabel(
            f_bt, text="Select devices to see delay compensation info",
            font=ctk.CTkFont(size=10), text_color=C["dim"],
            anchor="w", wraplength=460,
        )
        self._ms_lbl_comp_dir.grid(row=4, column=0, columnspan=2,
                                   padx=14, pady=(0, 10), sticky="ew")

        # --- SPEAKER PLACEMENT (room canvas) ---------------------------------
        self._build_section("SPEAKER PLACEMENT", pad, parent=ms)

        # ---- Room canvas ------------------------------------------------
        f_canvas_outer = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_canvas_outer.pack(fill="x", padx=pad, pady=(0, 4))

        canvas_w = self.W - pad * 2 - 4
        canvas_h = 340
        self._room_canvas = Room3DCanvas(
            f_canvas_outer,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            room_width_m=self._ms_room_w,
            room_depth_m=self._ms_room_d,
            room_height_m=self._ms_room_h,
            on_speaker_moved=self._on_ms_speaker_moved,
            on_speaker_selected=self._on_ms_speaker_selected,
            on_speaker_rotated=self._on_ms_speaker_rotated,
            on_change=self._on_ms_canvas_change,
        )
        self._room_canvas.pack(fill="x", padx=2, pady=(4, 2))

        # ---- Room dimension + layout controls ---------------------------
        f_room_ctrl = ctk.CTkFrame(f_canvas_outer, fg_color="transparent")
        f_room_ctrl.pack(fill="x", padx=8, pady=(2, 6))
        for ci in range(9):
            f_room_ctrl.grid_columnconfigure(ci, weight=1 if ci in (1, 3, 5) else 0)

        # Room W
        ctk.CTkLabel(f_room_ctrl, text="Room W:",
                     font=ctk.CTkFont(size=10), text_color=C["dim"]).grid(
            row=0, column=0, padx=(4, 2), pady=0, sticky="e")
        self._ms_room_w_entry = ctk.CTkEntry(
            f_room_ctrl, width=52, height=26,
            fg_color=C["surface2"], border_color=C["dim"],
            font=ctk.CTkFont(size=10, family="Consolas"))
        self._ms_room_w_entry.insert(0, f"{self._ms_room_w:.1f}")
        self._ms_room_w_entry.grid(row=0, column=1, padx=(0, 4), pady=0, sticky="ew")
        self._ms_room_w_entry.bind("<Return>", lambda _: self._on_ms_room_dim_change())
        self._ms_room_w_entry.bind("<FocusOut>", lambda _: self._on_ms_room_dim_change())

        ctk.CTkLabel(f_room_ctrl, text="D:",
                     font=ctk.CTkFont(size=10), text_color=C["dim"]).grid(
            row=0, column=2, padx=(4, 2), pady=0, sticky="e")
        self._ms_room_d_entry = ctk.CTkEntry(
            f_room_ctrl, width=52, height=26,
            fg_color=C["surface2"], border_color=C["dim"],
            font=ctk.CTkFont(size=10, family="Consolas"))
        self._ms_room_d_entry.insert(0, f"{self._ms_room_d:.1f}")
        self._ms_room_d_entry.grid(row=0, column=3, padx=(0, 4), pady=0, sticky="ew")
        self._ms_room_d_entry.bind("<Return>", lambda _: self._on_ms_room_dim_change())
        self._ms_room_d_entry.bind("<FocusOut>", lambda _: self._on_ms_room_dim_change())

        ctk.CTkLabel(f_room_ctrl, text="H:",
                     font=ctk.CTkFont(size=10), text_color=C["dim"]).grid(
            row=0, column=4, padx=(4, 2), pady=0, sticky="e")
        self._ms_room_h_entry = ctk.CTkEntry(
            f_room_ctrl, width=48, height=26,
            fg_color=C["surface2"], border_color=C["dim"],
            font=ctk.CTkFont(size=10, family="Consolas"))
        self._ms_room_h_entry.insert(0, f"{self._ms_room_h:.1f}")
        self._ms_room_h_entry.grid(row=0, column=5, padx=(0, 8), pady=0, sticky="ew")
        self._ms_room_h_entry.bind("<Return>", lambda _: self._on_ms_room_dim_change())
        self._ms_room_h_entry.bind("<FocusOut>", lambda _: self._on_ms_room_dim_change())

        # Layout preset buttons
        _layout_names = list(SPEAKER_LAYOUTS_3D.keys())
        for ci, lname in enumerate(_layout_names):
            short = lname.replace(" Cinema", "").replace(" Dolby", "")
            btn = ctk.CTkButton(
                f_room_ctrl, text=short,
                font=ctk.CTkFont(size=9),
                height=26, width=70,
                fg_color=C["surface2"], hover_color=C["surface"],
                border_color=C["dim"], border_width=1,
                text_color=C["text"], corner_radius=5,
                command=lambda n=lname: self._on_ms_layout_preset(n),
            )
            btn.grid(row=1, column=ci, padx=(2, 0), pady=(2, 0))

        # Add-speaker button (after layout presets)
        ctk.CTkButton(
            f_room_ctrl, text="+ Add",
            font=ctk.CTkFont(size=9, weight="bold"),
            height=26, width=52,
            fg_color=C["accent"], hover_color="#5a73f5",
            text_color="white", corner_radius=5,
            command=self._on_ms_add_speaker,
        ).grid(row=1, column=len(_layout_names), padx=(4, 2), pady=(2, 0))

        # ---- Selected-speaker panel (hidden until a speaker is selected) --
        self._ms_spk_panel = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        self._ms_spk_panel.pack(fill="x", padx=pad, pady=(0, 4))
        self._ms_spk_panel.grid_columnconfigure(1, weight=1)
        self._ms_spk_panel.grid_columnconfigure(3, weight=1)

        # Row 0: speaker id label + device assignment
        self._ms_spk_title = ctk.CTkLabel(
            self._ms_spk_panel, text="No speaker selected",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["dim"], anchor="w",
        )
        self._ms_spk_title.grid(row=0, column=0, columnspan=2,
                                padx=14, pady=(10, 4), sticky="ew")

        # Device dropdown for selected speaker
        ctk.CTkLabel(self._ms_spk_panel, text="Assign device:",
                     font=ctk.CTkFont(size=11), text_color=C["dim"],
                     anchor="w").grid(row=1, column=0, padx=(14, 6),
                                      pady=(0, 4), sticky="w")
        names_all = [label for _, label in self._all_out_list]
        self._ms_spk_dev_menu = ctk.CTkOptionMenu(
            self._ms_spk_panel,
            values=names_all if names_all else ["No output devices"],
            font=ctk.CTkFont(size=10),
            fg_color=C["surface2"],
            button_color=C["accent"], button_hover_color="#5a73f5",
            dropdown_fg_color=C["surface2"], dropdown_hover_color=C["surface"],
            corner_radius=12, height=30,
            command=self._on_ms_spk_dev_change,
        )
        self._ms_spk_dev_menu.grid(row=1, column=1, columnspan=2,
                                   padx=(0, 6), pady=(0, 4), sticky="ew")

        # Remove button
        self._ms_spk_remove_btn = ctk.CTkButton(
            self._ms_spk_panel, text="Remove",
            font=ctk.CTkFont(size=10),
            fg_color=C["danger"], hover_color="#f56070",
            text_color="white", corner_radius=12, height=30, width=80,
            command=self._on_ms_remove_speaker,
        )
        self._ms_spk_remove_btn.grid(row=1, column=3, padx=(0, 14),
                                     pady=(0, 4), sticky="e")

        # Facing azimuth slider
        ctk.CTkLabel(self._ms_spk_panel, text="Face Az:",
                     font=ctk.CTkFont(size=10), text_color=C["dim"],
                     anchor="w").grid(row=2, column=0, padx=(14, 4),
                                      pady=(0, 2), sticky="w")
        self._ms_face_az_slider = ctk.CTkSlider(
            self._ms_spk_panel, from_=0, to=360, number_of_steps=72,
            height=20,
            fg_color=C["surface2"], progress_color=C["accent"],
            button_color=C["accent"], button_hover_color="#5a73f5",
            command=self._on_ms_face_slider_change,
        )
        self._ms_face_az_slider.set(0)
        self._ms_face_az_slider.grid(row=2, column=1, columnspan=2,
                                     padx=(0, 4), pady=(0, 2), sticky="ew")
        self._ms_lbl_face_az = ctk.CTkLabel(
            self._ms_spk_panel, text="0°",
            font=ctk.CTkFont(size=10, family="Consolas"),
            text_color=C["dim"], width=36, anchor="e",
        )
        self._ms_lbl_face_az.grid(row=2, column=3, padx=(0, 14), pady=(0, 2), sticky="e")

        # Facing elevation slider
        ctk.CTkLabel(self._ms_spk_panel, text="Face El:",
                     font=ctk.CTkFont(size=10), text_color=C["dim"],
                     anchor="w").grid(row=3, column=0, padx=(14, 4),
                                      pady=(0, 4), sticky="w")
        self._ms_face_el_slider = ctk.CTkSlider(
            self._ms_spk_panel, from_=-90, to=90, number_of_steps=36,
            height=20,
            fg_color=C["surface2"], progress_color=C["accent"],
            button_color=C["accent"], button_hover_color="#5a73f5",
            command=self._on_ms_face_slider_change,
        )
        self._ms_face_el_slider.set(0)
        self._ms_face_el_slider.grid(row=3, column=1, columnspan=2,
                                     padx=(0, 4), pady=(0, 4), sticky="ew")
        self._ms_lbl_face_el = ctk.CTkLabel(
            self._ms_spk_panel, text="0°",
            font=ctk.CTkFont(size=10, family="Consolas"),
            text_color=C["dim"], width=36, anchor="e",
        )
        self._ms_lbl_face_el.grid(row=3, column=3, padx=(0, 14), pady=(0, 4), sticky="e")

        # Orientation hint
        ctk.CTkLabel(
            self._ms_spk_panel,
            text="Drag dots to reposition  ·  Shift-drag to adjust height  ·  "
                 "Right-click for orientation presets  ·  Use sliders above for precise rotation",
            font=ctk.CTkFont(size=9), text_color=C["dim"],
            anchor="w", wraplength=600,
        ).grid(row=4, column=0, columnspan=4, padx=14, pady=(0, 8), sticky="ew")

        # ---- Backwards-compat stubs (referenced by existing handlers) ----
        # Hidden widgets so old orientation/azimuth handlers don't error
        self._ms_swap_switch = ctk.CTkSwitch(
            f_canvas_outer, text="", fg_color=C["surface2"],
            command=self._on_ms_swap_toggle,
        )
        if self._ms_swap_rear_lr:
            self._ms_swap_switch.select()

        # Initialise canvas with default 2-speaker layout
        self._on_ms_layout_preset("2.0 Stereo")

        # --- CHANNEL ROUTING -------------------------------------------------
        self._build_section("CHANNEL ROUTING", pad, parent=ms)
        f_route = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_route.pack(fill="x", padx=pad, pady=(0, 4))
        f_route.grid_columnconfigure(1, weight=1)

        for r, (side, desc, lbl_attr) in enumerate([
            ("Front speaker:", "System audio output (unprocessed)", "_ms_lbl_front_route"),
            ("Rear speaker:",  "Surround L/R + Rear L/R + sub-bass",  "_ms_lbl_rear_route"),
        ]):
            py = (8 if r == 0 else 4, 8 if r == 1 else 4)
            ctk.CTkLabel(
                f_route, text=side,
                font=ctk.CTkFont(size=11), text_color=C["dim"],
                width=110, anchor="w",
            ).grid(row=r, column=0, padx=(14, 4), pady=py, sticky="w")
            val_lbl = ctk.CTkLabel(
                f_route, text=desc,
                font=ctk.CTkFont(size=11), text_color=C["text"], anchor="w",
            )
            val_lbl.grid(row=r, column=1, padx=(0, 14), pady=py, sticky="ew")
            setattr(self, lbl_attr, val_lbl)

        # --- LEVELS ----------------------------------------------------------
        self._build_section("LEVELS", pad, parent=ms)
        f_met = ctk.CTkFrame(ms, fg_color=C["surface"], corner_radius=16)
        f_met.pack(fill="x", padx=pad, pady=(0, 4))
        f_met.grid_columnconfigure(1, weight=1)

        for r, (label, attr) in enumerate([
            ("Front L", "_ms_cvs_frontL"),
            ("Front R", "_ms_cvs_frontR"),
            ("Rear  L", "_ms_cvs_rearL"),
            ("Rear  R", "_ms_cvs_rearR"),
        ]):
            py = (7 if r == 0 else 4, 0)
            ctk.CTkLabel(
                f_met, text=label,
                font=ctk.CTkFont(size=11, family="Consolas"),
                text_color=C["dim"], width=52, anchor="w",
            ).grid(row=r, column=0, padx=(14, 6), pady=py)

            cvs = ctk.CTkCanvas(f_met, height=14, bg=C["surface2"],
                                highlightthickness=0, bd=0)
            cvs.grid(row=r, column=1, padx=(0, 6), pady=py, sticky="ew")
            setattr(self, attr, cvs)

            lbl_w = ctk.CTkLabel(
                f_met, text="-inf",
                font=ctk.CTkFont(size=10, family="Consolas"),
                text_color=C["dim"], width=44, anchor="e",
            )
            lbl_w.grid(row=r, column=2, padx=(0, 14), pady=py)
            setattr(self, attr.replace("_cvs_", "_lbl_"), lbl_w)

        ctk.CTkLabel(f_met, text="", height=4).grid(row=4, column=0)

        # Volume sliders (Front / Rear)
        for r, (lbl_txt, attr_slider, attr_lbl, cb) in enumerate([
            ("Front Vol", "_ms_front_vol_slider", "_ms_lbl_front_vol",
             lambda v: self._on_ms_front_gain(v)),
            ("Rear  Vol", "_ms_rear_vol_slider",  "_ms_lbl_rear_vol",
             lambda v: self._on_ms_rear_gain(v)),
        ], start=5):
            ctk.CTkLabel(
                f_met, text=lbl_txt,
                font=ctk.CTkFont(size=11, family="Consolas"),
                text_color=C["dim"], width=52, anchor="w",
            ).grid(row=r, column=0, padx=(14, 6), pady=(4, 0))

            sl = ctk.CTkSlider(
                f_met, from_=-30.0, to=6.0,
                height=16,
                button_color=C["accent"], button_hover_color="#5a73f5",
                progress_color=C["accent"], fg_color=C["surface2"], corner_radius=4,
                command=cb,
            )
            sl.set(0.0)
            sl.grid(row=r, column=1, padx=(0, 6), pady=(4, 0), sticky="ew")
            setattr(self, attr_slider, sl)

            lbl_v = ctk.CTkLabel(
                f_met, text="0 dB",
                font=ctk.CTkFont(size=10, family="Consolas"),
                text_color=C["success"], width=44, anchor="e",
            )
            lbl_v.grid(row=r, column=2, padx=(0, 14), pady=(4, 0))
            setattr(self, attr_lbl, lbl_v)

        ctk.CTkLabel(f_met, text="", height=6).grid(row=7, column=0)

        # --- TRANSPORT -------------------------------------------------------
        f_ms_trans = ctk.CTkFrame(ms, fg_color="transparent")
        f_ms_trans.pack(fill="x", padx=pad, pady=(10, 4))

        self._ms_start_btn = ctk.CTkButton(
            f_ms_trans,
            text="   START MULTI-SPEAKER",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=C["success"],
            hover_color="#08f0b0",
            text_color="#0d1117",
            corner_radius=18, height=54,
            command=self._toggle_multi,
        )
        self._ms_start_btn.pack(fill="x")

        # --- STATUS BAR ------------------------------------------------------
        f_ms_stat = ctk.CTkFrame(ms, fg_color="transparent")
        f_ms_stat.pack(fill="x", padx=pad, pady=(6, 16))
        f_ms_stat.grid_columnconfigure(0, weight=1)
        f_ms_stat.grid_columnconfigure(1, weight=1)

        self._ms_lbl_status = ctk.CTkLabel(
            f_ms_stat, text="Stopped",
            font=ctk.CTkFont(size=10), text_color=C["dim"], anchor="w",
        )
        self._ms_lbl_status.grid(row=0, column=0, sticky="w")

        self._ms_lbl_xruns = ctk.CTkLabel(
            f_ms_stat, text="Xruns: 0",
            font=ctk.CTkFont(size=10), text_color=C["dim"], anchor="e",
        )
        self._ms_lbl_xruns.grid(row=0, column=1, sticky="e")

        # Apply initial mode state (loopback default: no capture dropdown)
        self._on_ms_mode_change("Loopback")
        self._update_ms_bt_labels()

    # =======================================================================
    # Room canvas event handlers
    # =======================================================================

    def _on_ms_speaker_moved(self, sid: int, azimuth_deg: float, distance_m: float):
        """Called by Room3DCanvas when a speaker is dragged to a new position."""
        for spk in self._ms_speakers:
            if spk["sid"] == sid:
                # Sync elevation from canvas (canvas is authoritative for 3D position)
                if self._room_canvas:
                    az, el, dist = self._room_canvas.get_speaker_spherical(sid)
                    spk["azimuth"]   = az
                    spk["elevation"] = el
                    spk["distance"]  = dist
                else:
                    spk["azimuth"]  = azimuth_deg
                    spk["distance"] = distance_m
                break
        # Hot-update VBAP matrix if stream is running
        if self._ms_running and self._ms_stream:
            self._ms_hot_update_speakers()

    def _on_ms_speaker_selected(self, sid: int):
        """Called by Room3DCanvas when a speaker dot is clicked."""
        self._ms_selected_sid = sid
        spk = next((s for s in self._ms_speakers if s["sid"] == sid), None)
        if spk is None:
            return
        az   = spk.get("azimuth",   0.0)
        el   = spk.get("elevation", 0.0)
        dist = spk.get("distance",  2.0)
        # Update panel title
        self._ms_spk_title.configure(
            text=f"Speaker: {spk['label']}  "
                 f"(az {az:.0f}°  el {el:.0f}°  "
                 f"dist {dist:.1f} m)",
            text_color=C["text"],
        )
        # Select device in dropdown
        dev_lbl = spk.get("device_label", "Unassigned")
        names = [l for _, l in self._all_out_list]
        if dev_lbl in names:
            self._ms_spk_dev_menu.set(dev_lbl)
        else:
            self._ms_spk_dev_menu.set(names[0] if names else "No devices")
        # Sync facing sliders
        face_az = spk.get("face_az", 0.0)
        face_el = spk.get("face_el", 0.0)
        if hasattr(self, "_ms_face_az_slider"):
            self._ms_face_az_slider.set(face_az)
            self._ms_lbl_face_az.configure(text=f"{face_az:.0f}°")
        if hasattr(self, "_ms_face_el_slider"):
            self._ms_face_el_slider.set(face_el)
            self._ms_lbl_face_el.configure(text=f"{face_el:.0f}°")

    def _on_ms_canvas_change(self):
        """Called by Room3DCanvas on any structural change (add/remove speaker)."""
        if self._room_canvas is None:
            return
        canvas_spks = self._room_canvas.get_speakers()
        # Merge positions — canvas is source of truth for all 3D position/orientation
        for cs in canvas_spks:
            az, el, dist = self._room_canvas.get_speaker_spherical(cs.sid)
            found = next((s for s in self._ms_speakers if s["sid"] == cs.sid), None)
            if found:
                found["azimuth"]      = az
                found["elevation"]    = el
                found["distance"]     = dist
                found["label"]        = cs.label
                found["device_idx"]   = cs.device_idx
                found["device_label"] = cs.device_label
                found["face_az"]      = cs.face_az
                found["face_el"]      = cs.face_el
            else:
                self._ms_speakers.append({
                    "sid":          cs.sid,
                    "label":        cs.label,
                    "azimuth":      az,
                    "elevation":    el,
                    "distance":     dist,
                    "device_idx":   cs.device_idx,
                    "device_label": cs.device_label,
                    "face_az":      cs.face_az,
                    "face_el":      cs.face_el,
                })
        # Remove speakers that no longer exist in canvas
        canvas_sids = {cs.sid for cs in canvas_spks}
        self._ms_speakers = [s for s in self._ms_speakers
                              if s["sid"] in canvas_sids]

    def _on_ms_layout_preset(self, layout_name: str):
        """Load a speaker layout preset into the room canvas."""
        if self._room_canvas is None:
            return
        self._ms_speakers.clear()
        self._room_canvas.clear_speakers()

        layout = SPEAKER_LAYOUTS_3D.get(layout_name, [])
        for label, az, el, dist in layout:
            # Auto-assign speakers based on their direction
            device_idx   = None
            device_label = "Unassigned"
            is_front = label in ("FL", "FR", "C", "FL/FR")
            is_rear  = label in ("SL", "SR", "BL", "BR", "RL", "RR")
            # Use the first matching output row device
            for row in self._ms_output_rows:
                d = row.get("direction", "")
                if is_front and "Front" in d and row.get("dev_idx") is not None:
                    device_idx   = row["dev_idx"]
                    device_label = row["dev_name"][:20]
                    break
                if is_rear and ("Rear" in d or "Surround" in d) and row.get("dev_idx") is not None:
                    device_idx   = row["dev_idx"]
                    device_label = row["dev_name"][:20]
                    break

            sid = self._room_canvas.add_speaker(
                label, az, el, dist,
                device_idx=device_idx,
                device_label=device_label,
            )
            self._ms_speakers.append({
                "sid":          sid,
                "label":        label,
                "azimuth":      az,
                "elevation":    el,
                "distance":     dist,
                "device_idx":   device_idx,
                "device_label": device_label,
                "face_az":      (az + 180.0) % 360.0,
                "face_el":      -el,
            })
        # Initialise per-speaker smoothing buffers
        self._ms_dsp_per_spk = {s["sid"]: np.zeros(2, dtype=np.float32)
                                 for s in self._ms_speakers}
        self._ms_spk_title.configure(
            text="No speaker selected", text_color=C["dim"])

    def _on_ms_add_speaker(self):
        """Add a new speaker at a default position."""
        if self._room_canvas is None:
            return
        n   = len(self._ms_speakers)
        az  = (n * 45.0) % 360.0   # spread around room
        el  = 0.0
        lbl = f"S{n + 1}"
        sid = self._room_canvas.add_speaker(lbl, az, el, 2.5)
        self._ms_speakers.append({
            "sid": sid, "label": lbl,
            "azimuth": az, "elevation": el, "distance": 2.5,
            "device_idx": None, "device_label": "Unassigned",
            "face_az": (az + 180.0) % 360.0, "face_el": 0.0,
        })
        self._ms_dsp_per_spk[sid] = np.zeros(2, dtype=np.float32)
        # Select the new speaker
        self._room_canvas.set_selected_sid(sid)
        self._on_ms_speaker_selected(sid)

    def _on_ms_remove_speaker(self):
        """Remove the currently selected speaker."""
        sid = self._ms_selected_sid
        if sid is None or self._room_canvas is None:
            return
        self._room_canvas.remove_speaker(sid)
        self._ms_speakers = [s for s in self._ms_speakers if s["sid"] != sid]
        self._ms_dsp_per_spk.pop(sid, None)
        self._ms_selected_sid = None
        self._ms_spk_title.configure(
            text="No speaker selected", text_color=C["dim"])

    def _on_ms_spk_dev_change(self, display_name: str):
        """Assign a device to the selected speaker."""
        sid = self._ms_selected_sid
        if sid is None:
            return
        for dev_idx, dev_label in self._all_out_list:
            if dev_label == display_name:
                # Update canvas
                if self._room_canvas:
                    self._room_canvas.set_speaker_device(
                        sid, dev_idx, dev_label[:20])
                # Update state
                for spk in self._ms_speakers:
                    if spk["sid"] == sid:
                        spk["device_idx"]   = dev_idx
                        spk["device_label"] = dev_label[:20]
                        break
                break

    def _on_ms_room_dim_change(self):
        """Room dimension entries changed."""
        try:
            w = float(self._ms_room_w_entry.get())
            d = float(self._ms_room_d_entry.get())
            h = float(self._ms_room_h_entry.get()) if hasattr(self, "_ms_room_h_entry") else self._ms_room_h
            if w > 0 and d > 0 and h > 0:
                self._ms_room_w = w
                self._ms_room_d = d
                self._ms_room_h = h
                if self._room_canvas:
                    self._room_canvas.set_room_size(w, d, h)
        except ValueError:
            pass

    # -----------------------------------------------------------------------
    # New 3D canvas / rotation / output-row handlers
    # -----------------------------------------------------------------------

    def _on_ms_speaker_rotated(self, sid: int, face_az: float, face_el: float):
        """Called by Room3DCanvas when a speaker's facing direction changes."""
        for spk in self._ms_speakers:
            if spk["sid"] == sid:
                spk["face_az"] = face_az
                spk["face_el"] = face_el
                break
        if self._ms_selected_sid == sid:
            if hasattr(self, "_ms_face_az_slider"):
                self._ms_face_az_slider.set(face_az)
                self._ms_lbl_face_az.configure(text=f"{face_az:.0f}°")
            if hasattr(self, "_ms_face_el_slider"):
                self._ms_face_el_slider.set(face_el)
                self._ms_lbl_face_el.configure(text=f"{face_el:.0f}°")
        # Hot-update routing matrix — facing direction affects L/R driver positions
        if self._ms_running and self._ms_stream:
            self._ms_hot_update_speakers()

    def _ms_hot_update_speakers(self):
        """Push current speaker positions and orientations to the running DSP chain."""
        if not self._ms_running or not self._ms_stream:
            return
        azimuths   = [s["azimuth"]                              for s in self._ms_speakers]
        elevations = [s.get("elevation", 0.0)                   for s in self._ms_speakers]
        face_azs   = [s.get("face_az", (s["azimuth"]+180.0)%360.0) for s in self._ms_speakers]
        face_els   = [s.get("face_el", 0.0)                     for s in self._ms_speakers]

        if hasattr(self._ms_stream, "update_speakers"):
            self._ms_stream.update_speakers(azimuths, elevations, face_azs, face_els)
        elif hasattr(self._ms_stream, "update_speaker_azimuths"):
            self._ms_stream.update_speaker_azimuths(azimuths)
        elif hasattr(self._ms_stream, "update_rear_az") and len(azimuths) >= 2:
            # Legacy 2-speaker path — also update full speaker info if available
            stream = self._ms_stream
            if hasattr(stream, "_chain") and hasattr(stream._chain, "update_speaker_info"):
                n = len(self._ms_speakers)
                f  = self._ms_speakers[0] if n >= 1 else {}
                r  = self._ms_speakers[1] if n >= 2 else {}
                stream._chain.update_speaker_info(
                    (f.get("azimuth", 0.0),    f.get("elevation", 0.0),
                     f.get("face_az", 180.0),   f.get("face_el",   0.0)),
                    (r.get("azimuth", 150.0),  r.get("elevation", 0.0),
                     r.get("face_az",   0.0),   r.get("face_el",   0.0)),
                )
            else:
                stream.update_rear_az(max(60.0, min(170.0, abs(azimuths[1]))))

    def _on_ms_face_slider_change(self, _value=None):
        """Facing azimuth/elevation sliders changed."""
        sid = self._ms_selected_sid
        if sid is None or self._room_canvas is None:
            return
        face_az = self._ms_face_az_slider.get()
        face_el = self._ms_face_el_slider.get()
        self._ms_lbl_face_az.configure(text=f"{face_az:.0f}°")
        self._ms_lbl_face_el.configure(text=f"{face_el:.0f}°")
        self._room_canvas.set_speaker_facing(sid, face_az, face_el)
        for spk in self._ms_speakers:
            if spk["sid"] == sid:
                spk["face_az"] = face_az
                spk["face_el"] = face_el
                break
        # Hot-update routing matrix — facing direction changes L/R driver positions
        if self._ms_running and self._ms_stream:
            self._ms_hot_update_speakers()

    def _ms_add_output_row(self, device_name: str = "", direction: str = "Front L/R"):
        """Add a dynamic output device row to the OUTPUT DEVICES panel."""
        names_all = [label for _, label in self._all_out_list]
        cur_name  = device_name if device_name in names_all else (names_all[0] if names_all else "No devices")
        dev_idx   = next((i for i, l in self._all_out_list if l == cur_name), None)

        f_row = ctk.CTkFrame(self._ms_out_rows_frame, fg_color="transparent")
        f_row.pack(fill="x", padx=0, pady=2)
        f_row.grid_columnconfigure(0, weight=1)

        dev_menu = ctk.CTkOptionMenu(
            f_row,
            values=names_all if names_all else ["No output devices"],
            font=ctk.CTkFont(size=11),
            fg_color=C["surface2"],
            button_color=C["accent"], button_hover_color="#5a73f5",
            dropdown_fg_color=C["surface2"], dropdown_hover_color=C["surface"],
            corner_radius=12, height=30,
        )
        dev_menu.set(cur_name)
        dev_menu.grid(row=0, column=0, padx=(0, 4), pady=0, sticky="ew")

        dir_menu = ctk.CTkOptionMenu(
            f_row,
            values=CHANNEL_DIRECTIONS,
            font=ctk.CTkFont(size=10),
            fg_color=C["surface2"],
            button_color=C["accent"], button_hover_color="#5a73f5",
            dropdown_fg_color=C["surface2"], dropdown_hover_color=C["surface"],
            corner_radius=12, height=30, width=148,
        )
        dir_menu.set(direction)
        dir_menu.grid(row=0, column=1, padx=(0, 4), pady=0)

        bass_btn = ctk.CTkButton(
            f_row, text="Bass ♦",
            font=ctk.CTkFont(size=10),
            fg_color=C["surface2"], hover_color=C["surface"],
            border_color=C["dim"], border_width=1,
            text_color=C["dim"],
            corner_radius=12, height=30, width=68,
        )
        bass_btn.grid(row=0, column=2, padx=(0, 4), pady=0)

        del_btn = ctk.CTkButton(
            f_row, text="×",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=C["danger"], hover_color="#f56070",
            text_color="white",
            corner_radius=12, height=30, width=30,
        )
        del_btn.grid(row=0, column=3, pady=0)

        row_data = {
            "frame":     f_row,
            "dev_menu":  dev_menu,
            "dir_menu":  dir_menu,
            "bass_btn":  bass_btn,
            "del_btn":   del_btn,
            "dev_idx":   dev_idx,
            "dev_name":  cur_name,
            "direction": direction,
            "bass_active": False,
        }
        self._ms_output_rows.append(row_data)

        # Wire callbacks (capture row_data by reference)
        dev_menu.configure(command=lambda v, rd=row_data: self._on_ms_out_row_dev_change(rd, v))
        dir_menu.configure(command=lambda v, rd=row_data: self._on_ms_out_row_dir_change(rd, v))
        bass_btn.configure(command=lambda rd=row_data: self._on_ms_bass_priority_row(rd))
        del_btn.configure(command=lambda rd=row_data: self._ms_remove_output_row(rd))

        self._sync_front_rear_from_rows()

    def _ms_remove_output_row(self, row_data: dict):
        """Remove a dynamic output device row."""
        if len(self._ms_output_rows) <= 1:
            return   # keep at least one row
        row_data["frame"].destroy()
        self._ms_output_rows = [r for r in self._ms_output_rows if r is not row_data]
        self._sync_front_rear_from_rows()

    def _on_ms_out_row_dev_change(self, row_data: dict, display_name: str):
        """Device dropdown changed in an output row."""
        for dev_idx, dev_label in self._all_out_list:
            if dev_label == display_name:
                row_data["dev_idx"]  = dev_idx
                row_data["dev_name"] = dev_label
                break
        self._sync_front_rear_from_rows()
        self._update_ms_bt_labels()

    def _on_ms_out_row_dir_change(self, row_data: dict, direction: str):
        """Direction dropdown changed in an output row."""
        row_data["direction"] = direction

    def _on_ms_bass_priority_row(self, row_data: dict):
        """Toggle bass priority for an output row."""
        direction = row_data.get("direction", "")
        side = "front" if "Front" in direction else "rear"
        if self._ms_bass_priority == side:
            self._ms_bass_priority = "equal"
        else:
            self._ms_bass_priority = side
        # Update button appearances across all rows
        for rd in self._ms_output_rows:
            d    = rd.get("direction", "")
            s    = "front" if "Front" in d else "rear"
            active = (self._ms_bass_priority == s)
            rd["bass_btn"].configure(
                fg_color=C["accent"] if active else C["surface2"],
                border_color=C["accent"] if active else C["dim"],
                text_color="white" if active else C["dim"],
            )
        if self._ms_running and self._ms_stream:
            if hasattr(self._ms_stream, "update_bass_priority"):
                self._ms_stream.update_bass_priority(self._ms_bass_priority)

    def _sync_front_rear_from_rows(self):
        """Update _ms_front_idx/_ms_rear_idx from the first front/rear output rows."""
        for rd in self._ms_output_rows:
            d = rd.get("direction", "")
            if "Front" in d and rd.get("dev_idx") is not None:
                self._ms_front_idx = rd["dev_idx"]
                break
        for rd in self._ms_output_rows:
            d = rd.get("direction", "")
            if ("Rear" in d or "Surround" in d) and rd.get("dev_idx") is not None:
                self._ms_rear_idx = rd["dev_idx"]
                break

    # =======================================================================
    # Backward-compat stubs for handlers that referenced old slider widgets
    # =======================================================================

    def _on_ms_orient_change(self, value: str):
        self._ms_swap_rear_lr = (value == "Faces Me")
        if self._ms_swap_switch.get() != self._ms_swap_rear_lr:
            if self._ms_swap_rear_lr:
                self._ms_swap_switch.select()
            else:
                self._ms_swap_switch.deselect()
        if self._ms_running and self._ms_stream:
            if hasattr(self._ms_stream, "update_swap_rear_lr"):
                self._ms_stream.update_swap_rear_lr(self._ms_swap_rear_lr)

    def _on_ms_az_slider(self, value: float):
        self._ms_rear_az_deg = float(value)
        if self._ms_running and self._ms_stream:
            if hasattr(self._ms_stream, "update_rear_az"):
                self._ms_stream.update_rear_az(self._ms_rear_az_deg)

    def _on_ms_spkd_slider(self, value: float):
        self._ms_acoustic_delay = float(value)

    def _update_ms_bt_labels(self):
        """Refresh BT detection labels and compensation direction text."""
        def dev_name(idx):
            try:
                return self._devs[idx]["name"]
            except (IndexError, KeyError):
                return ""

        front_name = dev_name(self._ms_front_idx)
        rear_name  = dev_name(self._ms_rear_idx)
        front_bt   = is_bluetooth_device(front_name)
        rear_bt    = is_bluetooth_device(rear_name)

        def bt_label(name, is_bt):
            short = (name[:36] + "…") if len(name) > 36 else name
            tag   = "  [Bluetooth]" if is_bt else "  [Wired]"
            return (short or "—") + tag

        self._ms_lbl_front_bt.configure(
            text=bt_label(front_name, front_bt),
            text_color=C["warn"] if front_bt else C["dim"],
        )
        self._ms_lbl_rear_bt.configure(
            text=bt_label(rear_name, rear_bt),
            text_color=C["warn"] if rear_bt else C["dim"],
        )

        d = self._ms_bt_delay
        if self._ms_mode in ("loopback", "rear_only"):
            if rear_bt and not front_bt:
                comp = (f"Rear BT lag ≈ {d:.0f} ms — front plays immediately, "
                        f"rear arrives {d:.0f} ms later.  Adjust delay slider to compensate.")
            elif front_bt and not rear_bt:
                comp = (f"Rear (wired) will be delayed {d:.0f} ms to match "
                        "the BT front speaker.")
            else:
                comp = "No Bluetooth compensation needed."
        else:
            # dual mode
            if rear_bt and not front_bt:
                comp = f"Front (wired) will be delayed {d:.0f} ms to sync with rear Bluetooth"
            elif front_bt and not rear_bt:
                comp = f"Rear (wired) will be delayed {d:.0f} ms to sync with front Bluetooth"
            elif front_bt and rear_bt:
                comp = "Both devices are Bluetooth — no compensation applied"
            else:
                comp = "Both devices are wired — no Bluetooth delay compensation"
        self._ms_lbl_comp_dir.configure(text=comp)

    # =======================================================================
    # Virtual device / driver handlers
    # =======================================================================

    def _ms_refresh_vdev_status(self):
        """Query virtual driver / loopback status and update the setup panel."""
        if not _IS_WIN:
            self._ms_refresh_vdev_status_nwin()
            return

        if not _HAS_VDEV:
            self._ms_vdev_status_lbl.configure(
                text="Ready — use Loopback mode (no setup needed)",
                text_color=C["success"],
            )
            return

        import threading as _th
        def _worker():
            try:
                st = _vdev.get_status()
            except Exception as exc:
                def _err():
                    self._ms_vdev_status_lbl.configure(
                        text=f"Status check failed: {exc}", text_color=C["danger"])
                self.after(0, _err)
                return

            def _apply():
                self._ms_vdev_status_cache = st
                is_installed = st.get("driver_installed", False)
                is_modaudio  = st.get("is_modaudio",      False)
                cap          = st.get("best_capture",     {})

                if is_installed:
                    name = st.get("device_name") or "Virtual Audio Driver"
                    if is_modaudio:
                        status_txt   = f"ModAudio Surround  ✓  ready"
                        status_color = C["success"]
                        hint = (
                            "Virtual speaker is installed and named 'ModAudio Surround'.\n"
                            "1. Click 'Set as Default' to route all Windows audio here.\n"
                            "2. Click '▶ Auto-Configure Full Control' to activate."
                        )
                    else:
                        status_txt   = f"Virtual driver installed: {name}"
                        status_color = C["warn"]
                        hint = (
                            "Driver is installed but not yet named 'ModAudio Surround'.\n"
                            "Run ModAudio as Administrator once to apply the rename.\n"
                            "Or use it as-is — it will still work."
                        )

                    self._ms_btn_install_driver.configure(
                        text="Reinstall / Update Driver",
                        fg_color=C["surface2"], hover_color=C["surface"],
                        border_color=C["dim"], border_width=1,
                        text_color=C["dim"],
                    )
                    self._ms_btn_set_default.configure(
                        state="normal",
                        text_color=C["text"], border_color=C["accent"],
                    )
                    self._ms_btn_autoconfigure.grid()

                    # Ensure capture device is in dropdown
                    if cap.get("found") and cap.get("idx") is not None:
                        self._ms_ensure_capture_device(cap["idx"], cap["name"] or "")

                elif cap.get("found") and cap.get("source_type") == "stereo_mix":
                    # Stereo Mix fallback — driver not installed
                    status_txt   = "Stereo Mix detected  (fallback)"
                    status_color = C["warn"]
                    hint = (
                        "Stereo Mix found — Full Control mode will use it.\n"
                        "For the best experience, click 'Install Virtual Speaker'\n"
                        "to create a dedicated 'ModAudio Surround' device."
                    )
                    self._ms_btn_set_default.configure(state="disabled")
                    self._ms_btn_autoconfigure.grid()
                    if cap.get("idx") is not None:
                        self._ms_ensure_capture_device(cap["idx"], cap["name"] or "")

                else:
                    status_txt   = "Virtual speaker not installed"
                    status_color = C["danger"]
                    hint = (
                        "Click 'Install Virtual Speaker' to create a free, open-source\n"
                        "virtual audio device named 'ModAudio Surround'.  This lets\n"
                        "Windows route audio through ModAudio to BOTH speakers at once.\n"
                        "Requires admin rights once.  No reboot needed."
                    )
                    self._ms_btn_set_default.configure(state="disabled")
                    self._ms_btn_autoconfigure.grid_remove()

                self._ms_vdev_status_lbl.configure(
                    text=status_txt, text_color=status_color)
                self._ms_vdev_hint.configure(text=hint, text_color=C["dim"])

            self.after(0, _apply)

        _th.Thread(target=_worker, daemon=True).start()

    def _ms_refresh_vdev_status_nwin(self):
        """Update the setup panel on macOS / Linux (no Windows driver available)."""
        _loopback_kw = ("blackhole", "soundflower", "vb-cable",
                        "cable output", "vb-audio", "loopback")
        found_name = None
        found_idx  = None
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] < 1:
                continue
            nl = d["name"].lower()
            if any(kw in nl for kw in _loopback_kw):
                found_name = d["name"]
                found_idx  = i
                break

        if found_name is not None:
            self._ms_vdev_status_lbl.configure(
                text=f"Loopback device found: {found_name}",
                text_color=C["success"],
            )
            self._ms_vdev_hint.configure(
                text=(
                    f"'{found_name}' is available as a capture source.\n"
                    "In Full Control mode, select it as the Capture device.\n"
                    "Set it as your system audio output to route all audio through ModAudio."
                ),
                text_color=C["dim"],
            )
            # Build a synthetic status cache so _on_ms_autoconfigure works
            self._ms_vdev_status_cache = {
                "driver_installed": False,
                "best_capture": {
                    "found":       True,
                    "idx":         found_idx,
                    "name":        found_name,
                    "source_type": "loopback",
                    "output_idx":  None,
                    "output_name": None,
                },
            }
            self._ms_btn_autoconfigure.grid()
            # Ensure device appears in capture dropdown
            self._ms_ensure_capture_device(found_idx, found_name)
        else:
            if _IS_MAC:
                hint = (
                    "No loopback device found.  Install one to capture system audio:\n"
                    "  • VB-Cable for Mac   (vb-audio.com/Cable/)\n"
                    "  • BlackHole (free)   (github.com/ExistentialAudio/BlackHole)\n"
                    "Set it as your system audio output, then relaunch ModAudio."
                )
            else:
                hint = (
                    "No loopback device found.  Install a virtual audio device\n"
                    "such as VB-Cable to capture system audio for processing."
                )
            self._ms_vdev_status_lbl.configure(
                text="No loopback device found",
                text_color=C["warn"],
            )
            self._ms_vdev_hint.configure(text=hint, text_color=C["dim"])
            self._ms_btn_autoconfigure.grid_remove()

    def _ms_ensure_capture_device(self, dev_idx: int, dev_name: str):
        """Add a capture device to the capture dropdown if not already present."""
        if any(i == dev_idx for i, _ in self._ms_all_in_list):
            return
        try:
            ha = self._hostapi_label(self._devs[dev_idx]["hostapi"])
        except Exception:
            ha = "WDM"
        label = f"{dev_name[:44]}  [{ha}]"
        self._ms_all_in_list.append((dev_idx, label))
        if hasattr(self, "_ms_cap_menu"):
            self._ms_cap_menu.configure(
                values=[lbl for _, lbl in self._ms_all_in_list])

    def _on_ms_install_driver(self):
        """Download and install the free virtual audio driver in a background thread."""
        if not _HAS_VDEV:
            return

        self._ms_btn_install_driver.configure(state="disabled", text="Installing…")
        self._ms_install_progress.set(0)
        self._ms_install_progress.grid()
        self._ms_install_status_lbl.configure(text="Starting…", text_color=C["dim"])
        self._ms_install_status_lbl.grid()

        import threading as _th
        def _worker():
            def _prog(frac, txt):
                def _ui():
                    self._ms_install_progress.set(frac)
                    self._ms_install_status_lbl.configure(text=txt)
                self.after(0, _ui)

            result = _vdev.setup_virtual_device(progress_cb=_prog)

            def _done():
                self._ms_install_progress.grid_remove()
                self._ms_install_status_lbl.grid_remove()
                if result["success"]:
                    self._ms_vdev_status_lbl.configure(
                        text="ModAudio Surround installed  ✓",
                        text_color=C["success"],
                    )
                    self._ms_vdev_hint.configure(
                        text=(
                            "Installation complete!  Click 'Set as Default' to\n"
                            "route Windows audio to ModAudio Surround, then click\n"
                            "'▶ Auto-Configure Full Control' to activate both speakers."
                        ),
                        text_color=C["success"],
                    )
                    # Refresh device lists
                    self.after(500, self._ms_refresh_vdev_status)
                    self.after(500, self._rebuild_ms_device_lists)
                else:
                    self._ms_vdev_status_lbl.configure(
                        text=f"Install failed: {result['message'][:60]}",
                        text_color=C["danger"],
                    )
                    hint_extra = ""
                    if result.get("need_admin"):
                        hint_extra = (
                            "Run ModAudio as Administrator:\n"
                            "Right-click modaudio.py → 'Run as administrator',\n"
                            "or run 'python app.py' from an elevated terminal."
                        )
                    else:
                        hint_extra = result["message"]
                    self._ms_vdev_hint.configure(
                        text=hint_extra, text_color=C["danger"])

                self._ms_btn_install_driver.configure(
                    state="normal",
                    text="Install Virtual Speaker",
                    fg_color=C["accent"], hover_color="#5a73f5",
                    border_width=0, text_color="white",
                )

            self.after(0, _done)

        _th.Thread(target=_worker, daemon=True).start()

    def _rebuild_ms_device_lists(self):
        """Refresh the multi-speaker device lists after a new device appears."""
        try:
            self._devs  = list(sd.query_devices())
            _pref_api   = self._preferred_api_label()
            _loopback_kw = ("stereo mix", "what u hear", "wave out mix",
                            "cable output", "vb-audio", "vb-cable", "loopback",
                            "virtual audio driver", "modaudio surround",
                            "blackhole", "soundflower")
            # Rebuild output list
            new_out = []
            for i, d in enumerate(self._devs):
                if (d["max_output_channels"] >= 1
                        and self._hostapi_label(d["hostapi"]) == _pref_api):
                    label = f"{d['name'][:48]}  [{_pref_api}]"
                    new_out.append((i, label))
            if new_out:
                self._all_out_list = new_out
            # Rebuild input list
            new_in = []
            for i, d in enumerate(self._devs):
                if (d["max_input_channels"] >= 1
                        and self._hostapi_label(d["hostapi"]) == _pref_api):
                    label = f"{d['name'][:48]}  [{_pref_api}]"
                    new_in.append((i, label))
            for i, d in enumerate(self._devs):
                if d["max_input_channels"] >= 1:
                    nl = d["name"].lower()
                    if any(k in nl for k in _loopback_kw):
                        if not any(idx == i for idx, _ in new_in):
                            tag   = self._hostapi_label(d["hostapi"])
                            label = f"{d['name'][:44]}  [{tag}]"
                            new_in.append((i, label))
            if new_in:
                self._ms_all_in_list = new_in
                if hasattr(self, "_ms_cap_menu"):
                    self._ms_cap_menu.configure(
                        values=[lbl for _, lbl in self._ms_all_in_list])
            # Update front/rear menus
            names_all = [lbl for _, lbl in self._all_out_list]
            for attr in ("_ms_front_menu", "_ms_rear_menu"):
                if hasattr(self, attr):
                    getattr(self, attr).configure(values=names_all)
        except Exception as exc:
            print(f"[rebuild_ms_lists] {exc}")

    def _on_ms_open_sound_settings(self):
        """Open the system audio settings panel (cross-platform)."""
        if _HAS_VDEV:
            _vdev.open_sound_settings()
        elif _IS_MAC:
            import subprocess as _sp
            try:
                _sp.Popen(["open", "/System/Library/PreferencePanes/Sound.prefPane"])
            except Exception:
                pass
        else:
            import subprocess as _sp
            try:
                _sp.Popen(["pavucontrol"])
            except Exception:
                pass

    def _on_ms_set_default_device(self):
        """Set the ModAudio Surround / virtual driver as Windows default output."""
        if not _HAS_VDEV:
            return
        st = getattr(self, "_ms_vdev_status_cache", None)
        if st is None:
            return

        device_name = None
        if st.get("driver_installed"):
            device_name = st.get("device_name") or _vdev.MODAUDIO_DEVICE_NAME
        else:
            cap = st.get("best_capture", {})
            out_name = cap.get("output_name")
            if out_name:
                device_name = out_name

        if not device_name:
            self._ms_vdev_status_lbl.configure(
                text="No virtual device found to set as default",
                text_color=C["danger"],
            )
            return

        self._ms_btn_set_default.configure(state="disabled", text="Setting…")

        import threading as _th
        def _worker():
            ok, msg = _vdev.set_default_output_device(device_name)
            def _done():
                if ok:
                    self._ms_vdev_status_lbl.configure(
                        text=f"Windows default → {device_name}  ✓",
                        text_color=C["success"],
                    )
                else:
                    self._ms_vdev_status_lbl.configure(
                        text=f"Could not set default: {msg[:60]}",
                        text_color=C["warn"],
                    )
                    self._ms_vdev_hint.configure(
                        text=(
                            "Automatic default switching requires comtypes.\n"
                            "pip install comtypes   — or set it manually:\n"
                            "Sound Settings → right-click ModAudio Surround → Set as Default."
                        ),
                        text_color=C["dim"],
                    )
                self._ms_btn_set_default.configure(
                    state="normal", text="Set as Default")
            self.after(0, _done)
        _th.Thread(target=_worker, daemon=True).start()

    def _on_ms_autoconfigure(self):
        """Switch to Full Control mode and auto-select the best capture device."""
        st = getattr(self, "_ms_vdev_status_cache", None)
        cap = (st or {}).get("best_capture", {})
        if not cap.get("found"):
            # Refresh and try again
            self.after(100, self._ms_refresh_vdev_status)
            return

        # Switch to Full Control mode
        if hasattr(self, "_ms_mode_seg"):
            self._ms_mode_seg.set("Full Control")
        self._on_ms_mode_change("Full Control")

        # Select capture device
        in_idx  = cap["idx"]
        in_name = cap["name"] or ""
        self._ms_ensure_capture_device(in_idx, in_name)
        self._ms_in_idx = in_idx

        if hasattr(self, "_ms_cap_menu"):
            lbl = next((l for i, l in self._ms_all_in_list if i == in_idx), None)
            if lbl:
                self._ms_cap_menu.set(lbl)

        src = cap.get("source_type", "")
        src_desc = {
            "virtual_driver": "ModAudio Surround virtual speaker",
            "stereo_mix":     "Stereo Mix (system loopback)",
            "loopback":       cap.get("name") or "loopback device",
        }.get(src, src or "loopback device")

        self._ms_vdev_hint.configure(
            text=(
                f"Capture: {src_desc}\n"
                "Select your Front Out and Rear Out speakers below,\n"
                "then press START MULTI-SPEAKER.  Both speakers will play\n"
                "simultaneously with full theater surround processing."
            ),
            text_color=C["success"],
        )
        # Auto-select front/rear devices if not already different
        if (self._ms_front_idx == self._ms_rear_idx
                and len(self._all_out_list) >= 2):
            self._ms_rear_idx = self._all_out_list[1][0]
            if hasattr(self, "_ms_rear_menu"):
                self._ms_rear_menu.set(self._all_out_list[1][1])

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

        # Multi-speaker meters + room canvas level feed
        if self._ms_running and self._ms_stream:
            raw_front = self._ms_stream.raw_out_front.copy()
            raw_rear  = self._ms_stream.raw_out_rear.copy()
            for ch in range(2):
                for disp, raw in [
                    (self._ms_dsp_front, raw_front),
                    (self._ms_dsp_rear,  raw_rear),
                ]:
                    if raw[ch] > disp[ch]:
                        disp[ch] = ALPHA_ATK * disp[ch] + (1 - ALPHA_ATK) * raw[ch]
                    else:
                        disp[ch] = ALPHA_REL * disp[ch] + (1 - ALPHA_REL) * raw[ch]

            db_f = [20 * np.log10(max(1e-7, v)) for v in self._ms_dsp_front]
            db_r = [20 * np.log10(max(1e-7, v)) for v in self._ms_dsp_rear]
            for cvs, lbl, db in [
                (self._ms_cvs_frontL, self._ms_lbl_frontL, db_f[0]),
                (self._ms_cvs_frontR, self._ms_lbl_frontR, db_f[1]),
                (self._ms_cvs_rearL,  self._ms_lbl_rearL,  db_r[0]),
                (self._ms_cvs_rearR,  self._ms_lbl_rearR,  db_r[1]),
            ]:
                self._draw_meter(cvs, db)
                lbl.configure(text=f"{db:+.0f} dB" if db > -60 else "-inf dB")
            self._ms_lbl_xruns.configure(
                text=f"Xruns: {self._ms_stream.xruns}",
                text_color=C["danger"] if self._ms_stream.xruns > 0 else C["dim"],
            )

            # Feed per-speaker stereo levels to room canvas for wave visualisation
            if self._room_canvas and self._ms_speakers:
                # N-speaker stream: raw_out is a list of [left_rms, right_rms]
                if isinstance(getattr(self._ms_stream, "raw_out", None), list):
                    raw_list = self._ms_stream.raw_out
                    for i, spk in enumerate(self._ms_speakers):
                        if i < len(raw_list) and len(raw_list[i]) >= 2:
                            l_raw = float(raw_list[i][0])
                            r_raw = float(raw_list[i][1])
                        elif i < len(raw_list):
                            l_raw = r_raw = float(np.mean(raw_list[i]))
                        else:
                            l_raw = r_raw = 0.0
                        # Smooth L and R independently (buf[0]=L, buf[1]=R)
                        buf = self._ms_dsp_per_spk.get(spk["sid"],
                                                       np.zeros(2, np.float32))
                        for ci, raw in enumerate((l_raw, r_raw)):
                            buf[ci] = (ALPHA_ATK * buf[ci] + (1 - ALPHA_ATK) * raw
                                       if raw > buf[ci]
                                       else ALPHA_REL * buf[ci] + (1 - ALPHA_REL) * raw)
                        self._ms_dsp_per_spk[spk["sid"]] = buf
                        self._room_canvas.set_speaker_stereo_level(
                            spk["sid"], float(buf[0]), float(buf[1]))
                else:
                    # Legacy 2-speaker: pass true per-channel L/R levels so the
                    # 3D visualisation can show which side of each speaker is active.
                    #
                    # raw_out_front/rear are measured AFTER swap_rear_lr, so:
                    #   _ms_dsp_front[0] = front bus ch0 = left-side content
                    #   _ms_dsp_front[1] = front bus ch1 = right-side content
                    #   _ms_dsp_rear[0]  = rear bus ch0  = right-side content
                    #                      (swap put rear_R here → listener's right)
                    #   _ms_dsp_rear[1]  = rear bus ch1  = left-side content
                    #                      (swap put rear_L here → listener's left)
                    for i, spk in enumerate(self._ms_speakers):
                        if i == 0:
                            self._room_canvas.set_speaker_stereo_level(
                                spk["sid"],
                                float(self._ms_dsp_front[0]),
                                float(self._ms_dsp_front[1]))
                        else:
                            # Rear speaker: ch0=right-side, ch1=left-side
                            # (levels match what each physical driver actually plays)
                            self._room_canvas.set_speaker_stereo_level(
                                spk["sid"],
                                float(self._ms_dsp_rear[0]),
                                float(self._ms_dsp_rear[1]))

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

    if _IS_WIN:
        _MS_MODE_DESCS = {
            "loopback": (
                "Captures what your front speaker is already playing via WASAPI loopback.\n"
                "No extra software needed — works on any Windows PC.\n"
                "Front speaker plays normally; rear speaker adds surround + direct blend."
            ),
            "dual": (
                "Full theater DSP applied to BOTH speakers simultaneously.\n"
                "Requires a virtual capture device (VB-Cable recommended, Stereo Mix works).\n"
                "Use 'Auto-Configure Full Control' above to set up automatically."
            ),
        }
    else:
        _MS_MODE_DESCS = {
            "loopback": (
                "Captures audio from a virtual loopback device (VB-Cable or BlackHole).\n"
                "Set the loopback device as your system audio output first.\n"
                "Front speaker plays normally; rear speaker adds surround + direct blend."
            ),
            "dual": (
                "Full theater DSP applied to BOTH speakers simultaneously.\n"
                "Requires a virtual capture device (VB-Cable or BlackHole).\n"
                "Use 'Auto-Configure Full Control' above to set up automatically."
            ),
        }

    def _on_ms_mode_change(self, value: str):
        self._ms_mode = "dual" if value == "Full Control" else "loopback"
        self._ms_mode_desc.configure(text=self._MS_MODE_DESCS[self._ms_mode])

        if self._ms_mode == "loopback":
            # Front device = loopback source; no separate capture device needed
            if hasattr(self, "_ms_front_menu"):
                self._ms_front_menu.configure(state="normal")
            if hasattr(self, "_ms_cap_row_label"):
                self._ms_cap_row_label.grid_remove()
            if hasattr(self, "_ms_cap_menu"):
                self._ms_cap_menu.grid_remove()
            if hasattr(self, "_ms_loopback_info"):
                self._ms_loopback_info.grid()
            self._ms_dual_note.grid_remove()
            if hasattr(self, "_ms_lbl_front_route"):
                self._ms_lbl_front_route.configure(
                    text="System audio (plays normally, loopback captured)")
            if hasattr(self, "_ms_lbl_rear_route"):
                self._ms_lbl_rear_route.configure(
                    text="Surround L/R + Rear L/R + sub-bass")
        else:  # "dual"
            if hasattr(self, "_ms_front_menu"):
                self._ms_front_menu.configure(state="normal")
            if hasattr(self, "_ms_cap_row_label"):
                self._ms_cap_row_label.grid()
            if hasattr(self, "_ms_cap_menu"):
                self._ms_cap_menu.grid()
            if hasattr(self, "_ms_loopback_info"):
                self._ms_loopback_info.grid_remove()
            self._ms_dual_note.grid()
            if hasattr(self, "_ms_lbl_front_route"):
                self._ms_lbl_front_route.configure(
                    text="FL + FR + Center blend + sub-bass")
            if hasattr(self, "_ms_lbl_rear_route"):
                self._ms_lbl_rear_route.configure(
                    text="Surround L/R + Rear L/R + sub-bass")

        self._update_ms_bt_labels()

    def _on_ms_device_change(self, display_name: str, attr_menu: str, attr_idx: str):
        for dev_idx, dev_name in self._all_out_list:
            if dev_name == display_name:
                setattr(self, attr_idx, dev_idx)
                break
        self._update_ms_bt_labels()

    def _on_ms_bt_slider(self, value: float):
        self._ms_bt_delay = value
        self._ms_lbl_bt_delay.configure(text=f"{value:.0f} ms")
        if self._ms_running and self._ms_stream:
            self._ms_stream.update_bt_delay(value)
        self._update_ms_bt_labels()

    def _on_ms_swap_toggle(self):
        self._ms_swap_rear_lr = bool(self._ms_swap_switch.get())
        if self._ms_running and self._ms_stream:
            self._ms_stream.update_swap_rear_lr(self._ms_swap_rear_lr)

    def _on_ms_orient_change(self, value: str):
        """Orientation segment button changed."""
        self._ms_swap_rear_lr = (value == "Faces Me")
        # Sync hidden switch
        if self._ms_swap_rear_lr:
            self._ms_swap_switch.select()
        else:
            self._ms_swap_switch.deselect()
        if self._ms_running and self._ms_stream:
            self._ms_stream.update_swap_rear_lr(self._ms_swap_rear_lr)

    def _on_ms_az_slider(self, value: float):
        """Rear speaker azimuth slider changed."""
        self._ms_rear_az_deg = float(value)
        self._ms_lbl_az.configure(text=f"{value:.0f}°")
        # Color hint: 150° = directly behind (ideal)
        col = C["success"] if abs(value - 150.0) < 15 else C["warn"]
        self._ms_lbl_az.configure(text_color=col)
        if self._ms_running and self._ms_stream:
            self._ms_stream.update_rear_az(self._ms_rear_az_deg)

    def _on_ms_spkd_slider(self, value: float):
        """Speaker acoustic distance delay slider changed."""
        self._ms_acoustic_delay = float(value)
        if value < 0.5:
            self._ms_lbl_spkd.configure(text="0.0 ms", text_color=C["dim"])
        else:
            cm = value * 34.0   # 1 ms ≈ 34 cm
            self._ms_lbl_spkd.configure(
                text=f"{value:.1f} ms",
                text_color=C["success"],
            )
        # Note: acoustic delay only takes effect on next stream start

    def _on_ms_theater_preset(self, name: str):
        """Theater mode preset button clicked."""
        self._ms_preset_name = name
        # Update button styles
        _theater_descs = {
            "Cinema": "Standard multiplex experience.\nBalanced reverb and dynamics.",
            "IMAX":   "Massive room, extreme bass depth,\nvery wide surround envelope.",
            "Dolby":  "Precision-tuned, clean dynamics,\nsharp localisation cues.",
            "Home":   "Intimate scale, gentle reverb,\nsuitable for small rooms.",
        }
        for nm, btn in self._ms_theater_btns.items():
            is_sel = (nm == name)
            btn.configure(
                fg_color=C["accent"] if is_sel else C["surface2"],
                border_width=0 if is_sel else 1,
                text_color="white" if is_sel else C["text"],
                font=ctk.CTkFont(size=11, weight="bold" if is_sel else "normal"),
            )
        if hasattr(self, "_ms_theater_desc"):
            self._ms_theater_desc.configure(
                text=_theater_descs.get(name, ""))
        # Rebuild the chain with the new preset if stream is running
        if self._ms_running and self._ms_stream:
            preset = self._build_ms_preset()
            self._ms_stream.update_chain(preset)

    def _on_ms_calibrate(self):
        """Run auto-calibration in a background thread (non-blocking)."""
        self._ms_btn_calibrate.configure(state="disabled", text="Measuring…")
        self._ms_lbl_calib_status.configure(
            text="Opening test streams — please wait…", text_color=C["warn"])

        def _worker():
            try:
                # Create a temporary MultiDeviceStream just for calibration
                from audio_multi import MultiDeviceStream as _MDS
                tmp = _MDS(
                    in_dev=self._ms_in_idx,
                    front_dev=self._ms_front_idx,
                    rear_dev=self._ms_rear_idx,
                    fs=SAMPLE_RATE,
                    block_size=BLOCK_SIZE,
                    mode=self._ms_mode,
                )
                estimated_ms = tmp.calibrate_bt_delay_ms()
                del tmp

                def _apply():
                    self._ms_bt_delay = estimated_ms
                    self._ms_bt_slider.set(estimated_ms)
                    self._ms_lbl_bt_delay.configure(text=f"{estimated_ms:.0f} ms")
                    if self._ms_running and self._ms_stream:
                        self._ms_stream.update_bt_delay(estimated_ms)
                    self._ms_lbl_calib_status.configure(
                        text=f"Calibrated: {estimated_ms:.0f} ms  "
                             f"(WASAPI latency measurement)",
                        text_color=C["success"],
                    )
                    self._ms_btn_calibrate.configure(
                        state="normal", text="Auto-Calibrate")
                    self._update_ms_bt_labels()

                self.after(0, _apply)

            except Exception as exc:
                def _err():
                    self._ms_lbl_calib_status.configure(
                        text=f"Calibration failed: {exc}",
                        text_color=C["danger"],
                    )
                    self._ms_btn_calibrate.configure(
                        state="normal", text="Auto-Calibrate")
                self.after(0, _err)

        import threading as _th
        _th.Thread(target=_worker, daemon=True).start()

    def _on_volume_change(self, value: float):
        """Theater tab master gain slider callback (dB)."""
        self._master_gain = 10 ** (value / 20.0)
        db_txt = f"{value:+.0f} dB" if value != 0.0 else "0 dB"
        self._vol_lbl.configure(text=db_txt)

    def _on_ms_cap_change(self, display_name: str):
        """Multi-speaker capture device changed."""
        for dev_idx, dev_name in self._ms_all_in_list:
            if dev_name == display_name:
                self._ms_in_idx = dev_idx
                break

    def _on_ms_front_gain(self, value: float):
        """Front volume slider callback (dB)."""
        self._ms_front_gain = 10 ** (value / 20.0)
        db_txt = f"{value:+.0f} dB" if value != 0.0 else "0 dB"
        self._ms_lbl_front_vol.configure(text=db_txt)
        if self._ms_running and self._ms_stream:
            self._ms_stream.update_front_gain(self._ms_front_gain)

    def _on_ms_rear_gain(self, value: float):
        """Rear volume slider callback (dB)."""
        self._ms_rear_gain = 10 ** (value / 20.0)
        db_txt = f"{value:+.0f} dB" if value != 0.0 else "0 dB"
        self._ms_lbl_rear_vol.configure(text=db_txt)
        if self._ms_running and self._ms_stream:
            self._ms_stream.update_rear_gain(self._ms_rear_gain)

    def _on_ms_bass_priority(self, side: str):
        """Toggle bass priority for front or rear speaker.
        Clicking the active speaker resets to 'equal'.
        """
        if self._ms_bass_priority == side:
            # Deactivate — back to equal
            self._ms_bass_priority = "equal"
        else:
            self._ms_bass_priority = side

        # Update button appearance
        for s, btn in self._ms_bass_btns.items():
            if self._ms_bass_priority == s:
                btn.configure(
                    fg_color=C["accent"], border_color=C["accent"],
                    text_color="white")
            else:
                btn.configure(
                    fg_color=C["surface2"], border_color=C["dim"],
                    text_color=C["dim"])

        if self._ms_running and self._ms_stream:
            self._ms_stream.update_bass_priority(self._ms_bass_priority)

    def _build_preset(self) -> dict:
        """Merge active preset + slider overrides + mode into a final preset dict."""
        # Surround modes use headphones base (detailed HRTF-oriented settings).
        # Mono surround uses speakers base (physical playback oriented).
        use_hp = self._mode in ("headphones", "surround")
        base = dict(HEADPHONES_PRESET if use_hp else SPEAKERS_PRESET)
        base["mode"] = self._mode
        base.update(self._slider_vals)
        return base

    def _build_ms_preset(self) -> dict:
        """Build the preset dict for the multi-speaker chain."""
        base = dict(HEADPHONES_PRESET)
        base["mode"] = "speakers"
        # Overlay the selected theater mode preset
        base.update(PRESETS.get(self._ms_preset_name, PRESETS["Cinema"]))
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
        # Interlock: stop multi-speaker if it is running
        if self._ms_running:
            self._stop_multi()

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
    # Multi-speaker transport
    # =======================================================================

    def _toggle_multi(self):
        if self._ms_running:
            self._stop_multi()
        else:
            self._start_multi()

    def _start_multi(self):
        # Interlock: stop theater if running
        if self._running:
            self._stop()

        preset = self._build_ms_preset()
        n_spk  = len(self._ms_speakers)

        # Determine whether to use N-speaker or legacy 2-speaker stream.
        # The legacy MultiDeviceStream path handles 2-speaker setups correctly:
        # it produces proper stereo L/R per bus via the 4-channel VBAP and
        # applies swap_rear_lr to account for a rear speaker that faces the listener.
        # The N-speaker path collapses each speaker to mono (correct for 3+
        # individual mono speakers, wrong for a single stereo 2-speaker setup).
        use_n_speaker = (n_spk > 2)

        if use_n_speaker:
            # ---- N-speaker mode (dual/Full Control only) ----------------
            if self._ms_mode != "dual":
                self._show_error(
                    "3 or more speakers require Full Control mode.\n"
                    "Please switch to 'Full Control' and set up a virtual "
                    "audio cable (VB-Cable / ModAudio Surround) as your "
                    "system output."
                )
                return

            assigned = [s for s in self._ms_speakers if s["device_idx"] is not None]
            speaker_devs = [s["device_idx"] for s in assigned]
            azimuths     = [s["azimuth"]                         for s in assigned]
            elevations   = [s.get("elevation", 0.0)              for s in assigned]
            face_azs     = [s.get("face_az", (s["azimuth"]+180.0)%360.0) for s in assigned]
            face_els     = [s.get("face_el", 0.0)                for s in assigned]

            if not speaker_devs:
                self._show_error(
                    "No output devices assigned to speakers.\n"
                    "Select a speaker dot in the room map and assign a device.")
                return

            gains = ([self._ms_front_gain]
                     + [self._ms_rear_gain] * (len(speaker_devs) - 1))
            try:
                self._ms_stream = MultiSpeakerStreamN(
                    in_dev=self._ms_in_idx,
                    speaker_devs=speaker_devs,
                    speaker_azimuths=azimuths,
                    speaker_elevations=elevations,
                    speaker_face_azs=face_azs,
                    speaker_face_els=face_els,
                    fs=SAMPLE_RATE,
                    block_size=BLOCK_SIZE,
                    preset=preset,
                    bt_delay_ms=self._ms_bt_delay,
                    gains=gains,
                    bass_priority=self._ms_bass_priority,
                )
                self._ms_stream.start()
                self._ms_running = True
                self._set_ms_running_ui(True)
                if self._room_canvas:
                    self._room_canvas.start_animation()
            except Exception as e:
                self._ms_stream = None
                self._show_error(
                    f"Could not start N-speaker stream:\n{e}\n\n"
                    "Check device assignments and ensure all devices are available.")

        else:
            # ---- Legacy 2-speaker stream (loopback or dual) -------------
            # Derive front/rear device from speaker list if possible
            front_dev = self._ms_front_idx
            rear_dev  = self._ms_rear_idx
            rear_az   = self._ms_rear_az_deg

            # Build full speaker-info tuples so the routing matrix uses
            # actual positions and facing directions (not hardcoded ±30°).
            def _spk_info(spk_dict, default_az, default_face_az):
                az    = float(spk_dict.get("azimuth",   default_az))
                el    = float(spk_dict.get("elevation", 0.0))
                faz   = float(spk_dict.get("face_az",   default_face_az))
                fel   = float(spk_dict.get("face_el",   0.0))
                return (az, el, faz, fel)

            front_info = None
            rear_info  = None
            if n_spk >= 1 and self._ms_speakers[0]["device_idx"] is not None:
                front_dev  = self._ms_speakers[0]["device_idx"]
                front_info = _spk_info(self._ms_speakers[0], 0.0,   180.0)
            if n_spk >= 2 and self._ms_speakers[1]["device_idx"] is not None:
                rear_dev   = self._ms_speakers[1]["device_idx"]
                rear_az    = abs(self._ms_speakers[1]["azimuth"])
                rear_az    = max(60.0, min(170.0, rear_az))
                rear_info  = _spk_info(self._ms_speakers[1], rear_az, (rear_az+180.0)%360.0)

            try:
                self._ms_stream = MultiDeviceStream(
                    in_dev=self._ms_in_idx,
                    front_dev=front_dev,
                    rear_dev=rear_dev,
                    fs=SAMPLE_RATE,
                    block_size=BLOCK_SIZE,
                    preset=preset,
                    bt_delay_ms=self._ms_bt_delay,
                    swap_rear_lr=self._ms_swap_rear_lr,
                    mode=self._ms_mode,
                    front_gain=self._ms_front_gain,
                    rear_gain=self._ms_rear_gain,
                    bass_priority=self._ms_bass_priority,
                    rear_az_deg=rear_az,
                    acoustic_delay_ms=self._ms_acoustic_delay,
                    front_info=front_info,
                    rear_info=rear_info,
                )
                self._ms_stream.start()
                self._ms_running = True
                self._set_ms_running_ui(True)
                if self._room_canvas:
                    self._room_canvas.start_animation()
            except Exception as e:
                self._ms_stream = None
                self._show_error(
                    f"Could not start multi-speaker:\n{e}\n\nCheck device selection.")

    def _stop_multi(self):
        self._ms_running = False
        if self._ms_stream:
            self._ms_stream.stop()
            self._ms_stream = None
        self._ms_dsp_front[:] = 0
        self._ms_dsp_rear[:]  = 0
        if self._room_canvas:
            self._room_canvas.stop_animation()
        self._set_ms_running_ui(False)

    def _set_ms_running_ui(self, running: bool):
        if running:
            self._ms_start_btn.configure(
                text="   STOP MULTI-SPEAKER",
                fg_color=C["danger"],
                hover_color="#f56070",
                text_color="white",
            )
            self._ms_lbl_status.configure(text="Running", text_color=C["success"])
            self._status_badge.configure(text="  RUNNING", text_color=C["success"])
        else:
            self._ms_start_btn.configure(
                text="   START MULTI-SPEAKER",
                fg_color=C["success"],
                hover_color="#08f0b0",
                text_color="#0d1117",
            )
            self._ms_lbl_status.configure(text="Stopped", text_color=C["dim"])
            if not self._running:
                self._status_badge.configure(text="  STOPPED", text_color=C["danger"])

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

            result = chain.process(block) * self._master_gain

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
        self._stop_multi()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ModAudioApp()
    app.mainloop()

"""
DeviceAnalyzer — multi-stage device diagnostic for automatic EQ baseline.

Stage 1 (always, ~0 ms):
    Looks up the device name in MEASUREMENT_DB (model-specific corrections derived
    from Rtings/ASR/InnerFidelity data) and falls back to category-level PROFILES.
    Also uses OS metadata (Bluetooth detection, channel count, sample rate) to
    refine the category assignment when no model match exists.

Stage 2 (Windows only, ~600 ms, plays brief tones at -20 dBFS):
    WASAPI loopback digital analysis.  Plays a log sweep and calibration tones
    through the output device, captures them back via WASAPI loopback.
    Detects:
      a) Windows Audio Enhancements (Bass Boost, Loudness Equalization, Room
         Correction, etc.) that alter the digital signal before it reaches the
         device.  Counteraction bands are added only for *wired* devices —
         Bluetooth speaker DSP is post-loopback and not visible here.
      b) Digital headroom: measures the highest level (dBFS) the path sustains
         without clipping or aggressive limiting.  Works for all device types.
    Skipped automatically when WASAPI is unavailable or on non-Windows systems.

Stage 3 (optional, user-triggered, ~3 s, audible, requires microphone):
    Acoustic sweep measurement.  Plays an exponential sine sweep through the
    selected output, records through a microphone input, and extracts a
    frequency response via deconvolution.  Smoothed in 1/3-octave bands.
    Confidence is capped at 0.50 because an uncalibrated laptop mic introduces
    significant coloration (+5–10 dB at 1–4 kHz, steep rolloff below 200 Hz).
    Always requires explicit user confirmation before running.
"""

from __future__ import annotations

import sys
import math
import threading
from dataclasses import dataclass, field

import numpy as np

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:  # pragma: no cover
    sd = None           # type: ignore
    _HAS_SD = False

_IS_WIN = sys.platform == "win32"

# Detect WASAPI support (sounddevice exposes WasapiSettings on Windows builds)
_WASAPI_AVAILABLE = False
if _IS_WIN and _HAS_SD:
    try:
        _WASAPI_SETTINGS_CLS = sd.WasapiSettings      # type: ignore[attr-defined]
        _WASAPI_AVAILABLE = True
    except AttributeError:
        pass

from .custom_eq        import DEFAULT_BANDS
from .speaker_profiles import get_correction_bands, _bands_to_gains, PROFILES


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Complete result from DeviceAnalyzer.analyze_device()."""

    # Core EQ recommendation
    profile_key:   str   = "reference"
    display_name:  str   = "Unknown Device"
    description:   str   = ""
    confidence:    float = 0.0
    band_gains:    list  = field(default_factory=list)

    # Diagnostic metadata
    stages_run:         list  = field(default_factory=list)
    notes:              list  = field(default_factory=list)
    driver_eq_notes:    str   = ""
    headroom_db:        float | None = None
    is_bluetooth:       bool  = False
    loopback_ran:       bool  = False

    def has_correction(self) -> bool:
        return any(abs(g) > 0.1 for g in self.band_gains)

    def summary_label(self) -> str:
        """Short label suitable for the EQ editor profile display."""
        conf_pct = f"  ({self.confidence * 100:.0f}% match)" if self.confidence < 0.80 else ""
        return f"Starting point: {self.display_name}{conf_pct}"


# ---------------------------------------------------------------------------
# DeviceAnalyzer
# ---------------------------------------------------------------------------

class DeviceAnalyzer:
    """
    Multi-stage acoustic diagnostic for automatic EQ baseline generation.

    Parameters
    ----------
    fs : int  — sample rate (must match the ParametricEQ instance, default 48 000 Hz)
    """

    # Stage 2 tuning
    _SWEEP_LEVEL_DBFS   = -20.0   # sweep playback level
    _SWEEP_DURATION     = 0.30    # seconds
    _HEADROOM_LEVELS    = (-18, -12, -6, -3, -1)  # dBFS levels to test

    def __init__(self, fs: int = 48_000):
        self._fs = fs

    # ------------------------------------------------------------------
    # Public API

    def analyze_device(
        self,
        device_name:       str,
        device_idx:        int | None = None,
        band_freqs:        list | None = None,
        run_loopback:      bool = True,
        progress_callback  = None,  # callable(stage, pct, note_str)
    ) -> AnalysisResult:
        """
        Run all available diagnostic stages and return a combined EQ correction.

        Stages run:
          1. Always — database lookup + OS metadata
          2. Windows + wired (optional) — WASAPI loopback
        Stage 3 (acoustic sweep) must be triggered separately via
        run_acoustic_sweep().

        Parameters
        ----------
        device_name       : audio device name string (from sounddevice)
        device_idx        : sounddevice device index, or None
        band_freqs        : list of EQ band centre frequencies
                            (defaults to DEFAULT_BANDS frequencies)
        run_loopback      : set False to skip Stage 2
        progress_callback : callable(stage_name, fraction_0_to_1, note_str)
        """
        if band_freqs is None:
            band_freqs = [b[1] for b in DEFAULT_BANDS]

        result = AnalysisResult()
        result.band_gains = [0.0] * len(band_freqs)

        def _prog(stage: str, pct: float, note: str = ""):
            if progress_callback:
                try:
                    progress_callback(stage, pct, note)
                except Exception:
                    pass

        # ---- Stage 1 --------------------------------------------------
        _prog("stage1", 0.0, "Checking device database…")
        s1 = self._stage1(device_name, device_idx, band_freqs)

        result.profile_key  = s1["profile_key"]
        result.display_name = s1["display_name"]
        result.description  = s1["description"]
        result.confidence   = s1["confidence"]
        result.is_bluetooth = s1["is_bluetooth"]
        result.notes.extend(s1["notes"])
        for i, g in enumerate(s1["band_gains"]):
            result.band_gains[i] += g
        result.stages_run.append("database_lookup")
        _prog("stage1", 1.0, f"Found: {result.display_name}")

        # ---- Stage 2 --------------------------------------------------
        if (run_loopback and _WASAPI_AVAILABLE and _IS_WIN
                and device_idx is not None and _HAS_SD):
            _prog("stage2", 0.0, "Running WASAPI loopback diagnostic…")
            try:
                s2 = self._stage2_loopback(device_idx, band_freqs, _prog)
                if s2 is not None:
                    result.headroom_db     = s2.get("headroom_db")
                    result.loopback_ran    = True
                    result.driver_eq_notes = s2.get("driver_eq_notes", "")
                    result.notes.extend(s2.get("notes", []))
                    # Only apply driver-EQ correction for wired devices.
                    # Bluetooth DSP lives inside the speaker hardware after
                    # the loopback capture point and cannot be measured here.
                    if not result.is_bluetooth:
                        for i, g in enumerate(s2.get("driver_eq_gains", [])):
                            if i < len(result.band_gains):
                                result.band_gains[i] += g
            except Exception as exc:
                result.notes.append(f"Loopback stage skipped: {exc}")
            _prog("stage2", 1.0)

        return result

    def run_acoustic_sweep(
        self,
        output_device_idx: int,
        input_device_idx:  int,
        band_freqs:        list | None = None,
        level_dbfs:        float = -20.0,
        sweep_duration:    float = 1.5,
        progress_callback  = None,
    ) -> AnalysisResult:
        """
        Measure the acoustic frequency response using a sine sweep + microphone.

        !! WARNING !!
        Uncalibrated microphones (especially laptop built-ins) have significant
        frequency coloration (+5–10 dB at 1–4 kHz, steep rolloff below 200 Hz
        and above 8 kHz).  Treat the result as a rough indication only.  Never
        apply it automatically; always require explicit user confirmation.

        Parameters
        ----------
        output_device_idx : output device to play the sweep through
        input_device_idx  : microphone input device to record from
        band_freqs        : EQ band centre frequencies (uses DEFAULT_BANDS if None)
        level_dbfs        : playback level (default −20 dBFS — safe for most devices)
        sweep_duration    : length of the ESS sweep in seconds
        progress_callback : callable(stage, pct, note)

        Returns
        -------
        AnalysisResult with band_gains and a confidence of 0.45 max.
        """
        if band_freqs is None:
            band_freqs = [b[1] for b in DEFAULT_BANDS]

        result = AnalysisResult()
        result.band_gains  = [0.0] * len(band_freqs)
        result.stages_run  = ["acoustic_sweep"]
        result.confidence  = 0.0
        result.display_name = "Acoustic Measurement"

        def _prog(pct: float, note: str = ""):
            if progress_callback:
                try:
                    progress_callback("stage3", pct, note)
                except Exception:
                    pass

        if not _HAS_SD:
            result.notes.append("sounddevice not available — acoustic sweep requires it.")
            return result

        _prog(0.0, "Generating exponential sweep…")
        fs    = self._fs
        level = 10 ** (level_dbfs / 20.0)
        sweep = self._gen_ess(50.0, 18_000.0, sweep_duration, fs)
        pad   = np.zeros(int(fs * 0.6), dtype=np.float32)   # 600 ms capture tail
        play  = np.concatenate([sweep, pad]) * level
        play_stereo = np.column_stack([play, play])

        _prog(0.1, "Playing sweep and recording…")
        try:
            recorded = sd.playrec(
                play_stereo,
                samplerate=fs,
                input_device=input_device_idx,
                output_device=output_device_idx,
                channels=2,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            result.notes.append(f"Acoustic sweep failed: {exc}")
            return result

        _prog(0.55, "Computing impulse response…")
        rec_mono = recorded[:, 0] if recorded.ndim > 1 else recorded

        ir = self._deconvolve(sweep, rec_mono[:len(sweep) + len(pad)], fs)
        if ir is None or np.max(np.abs(ir)) < 1e-8:
            result.notes.append("Deconvolution failed — check microphone and speaker connections.")
            return result

        # Window to early direct sound (0–30 ms) to reduce room-mode influence
        win_n = int(fs * 0.030)
        win_n = min(win_n, len(ir))
        ir_w  = ir[:win_n] * np.hanning(win_n).astype(np.float32)

        _prog(0.75, "Smoothing frequency response…")
        nfft   = 2 ** int(np.ceil(np.log2(win_n * 2)))
        H      = np.fft.rfft(ir_w, n=nfft)
        f_bins = np.fft.rfftfreq(nfft, d=1.0 / fs)
        H_db   = 20.0 * np.log10(np.maximum(np.abs(H), 1e-10))

        # Evaluate at log-spaced measurement frequencies (80 Hz – 16 kHz)
        meas_f = np.geomspace(80.0, 16_000.0, 60)
        H_meas = np.interp(meas_f, f_bins[1:], H_db[1:])

        # 1/3-octave smoothing
        H_sm = _fractional_octave_smooth(H_meas, meas_f, fraction=3)

        # Normalise to 1 kHz reference
        ref_mask = (meas_f >= 800) & (meas_f <= 1_200)
        if np.any(ref_mask):
            H_sm -= float(np.mean(H_sm[ref_mask]))

        # Invert and cap
        corr = np.clip(-H_sm, -10.0, 10.0)
        result.band_gains = _response_to_band_gains(corr, meas_f, band_freqs)

        result.confidence   = 0.45   # capped for uncalibrated mic
        result.display_name = "Acoustic Measurement (uncalibrated mic)"
        result.description  = (
            "Frequency response measured with microphone sweep. "
            "Accuracy is limited by the uncalibrated microphone. "
            "Use as a rough starting point and refine by ear."
        )
        result.notes.append(
            "Built-in laptop mics typically show +5–10 dB at 1–4 kHz and "
            "reduced response below 200 Hz — this affects the correction."
        )

        _prog(1.0, "Acoustic sweep complete.")
        return result

    # ------------------------------------------------------------------
    # Stage 1: database lookup + OS metadata

    def _stage1(
        self, device_name: str, device_idx: int | None, band_freqs: list
    ) -> dict:
        notes      = []
        is_bt      = self._is_bluetooth(device_name, device_idx)
        if is_bt:
            notes.append(
                "Bluetooth device: speaker's built-in DSP cannot be measured "
                "via loopback. Profile correction applied from database."
            )

        key, bands, conf, display, desc = get_correction_bands(device_name)
        gains = _bands_to_gains(bands, band_freqs)

        # Metadata refinement when only a generic/reference category matched
        if conf < 0.65 and device_idx is not None:
            override = self._metadata_refine(device_idx, device_name, key)
            if override:
                notes.append(override["note"])
                new_key = override.get("profile_key")
                if new_key and new_key != key:
                    key     = new_key
                    display = override["display"]
                    desc    = override["desc"]
                    pb      = PROFILES.get(key, PROFILES["reference"])["bands"]
                    gains   = _bands_to_gains(pb, band_freqs)
                    conf    = 0.55

        return {
            "profile_key":  key,
            "display_name": display,
            "description":  desc,
            "confidence":   conf,
            "band_gains":   gains,
            "is_bluetooth": is_bt,
            "notes":        notes,
        }

    def _is_bluetooth(self, device_name: str, device_idx: int | None) -> bool:
        name_lc  = device_name.lower()
        bt_hints = ("bluetooth", " bt ", "-bt-", "handsfree", "a2dp",
                    "avrcp", "btsco", "hands-free")
        if any(h in name_lc for h in bt_hints):
            return True
        if device_idx is not None and _HAS_SD:
            try:
                info = sd.query_devices(device_idx)
                # BT devices typically report high latency (>80 ms via A2DP)
                if info.get("default_high_output_latency", 0.0) > 0.080:
                    return True
            except Exception:
                pass
        return False

    def _metadata_refine(
        self, device_idx: int, device_name: str, current_key: str
    ) -> dict | None:
        """Use OS device metadata to tighten a generic profile assignment."""
        if not _HAS_SD:
            return None
        try:
            info = sd.query_devices(device_idx)
        except Exception:
            return None

        max_ch  = info.get("max_output_channels", 2)
        max_sr  = info.get("default_samplerate", 48_000)
        latency = info.get("default_high_output_latency", 0.02)

        # Mono output → almost certainly a laptop/phone built-in
        if max_ch == 1 and current_key in ("reference", "headphone_consumer", "iem_earbuds"):
            return {
                "note":        "Mono output detected — likely built-in laptop/phone speaker.",
                "profile_key": "laptop_speaker",
                "display":     "Built-in Mono Speaker",
                "desc":        "Mono built-in speaker — limited bass, boosted upper-mids.",
            }

        # Very high sample rate + wired + reference key → studio monitor
        if max_sr >= 96_000 and latency < 0.010:
            return {
                "note":        f"High sample rate ({max_sr/1000:.0f} kHz), low latency → professional device.",
                "profile_key": "reference",
                "display":     "Professional Audio Interface / Monitor",
                "desc":        "High-quality professional device — flat reference EQ applied.",
            }

        return None

    # ------------------------------------------------------------------
    # Stage 2: WASAPI loopback

    def _stage2_loopback(
        self, device_idx: int, band_freqs: list, progress_cb
    ) -> dict | None:
        loopback_idx = self._find_wasapi_loopback(device_idx)
        if loopback_idx is None:
            return None

        notes            = []
        driver_eq_gains  = [0.0] * len(band_freqs)
        headroom_db      = None

        # ---- Sweep test: detect Windows Audio Enhancements -------------
        progress_cb("stage2", 0.10, "Playing sweep, capturing via loopback…")

        fs    = self._fs
        level = 10 ** (self._SWEEP_LEVEL_DBFS / 20.0)
        sweep = self._gen_ess(20.0, 20_000.0, self._SWEEP_DURATION, fs)
        pad   = np.zeros(int(fs * 0.15), dtype=np.float32)
        mono  = np.concatenate([sweep, pad]) * level
        stereo = np.column_stack([mono, mono])

        try:
            recorded = sd.playrec(
                stereo,
                samplerate=fs,
                input_device=loopback_idx,
                output_device=device_idx,
                input_extra_settings=_WASAPI_SETTINGS_CLS(loopback=True),
                channels=2,
                dtype="float32",
            )
            sd.wait()
            rec_mono = recorded[:len(sweep), 0] if recorded is not None else None
        except Exception as exc:
            notes.append(f"Sweep capture failed ({exc}); headroom test will continue.")
            rec_mono = None

        if rec_mono is not None and len(rec_mono) > fs * 0.05:
            h_db = self._transfer_function_db(sweep, rec_mono, fs)
            if h_db is not None:
                deviation = float(np.max(np.abs(h_db)))
                if deviation > 3.0:
                    notes.append(
                        f"Windows Audio Enhancement detected (peak deviation "
                        f"{deviation:.1f} dB). Counteracting in EQ."
                    )
                    meas_f = np.geomspace(30.0, 18_000.0, len(h_db))
                    driver_eq_gains = _response_to_band_gains(
                        -h_db, meas_f, band_freqs   # apply inverse
                    )
                else:
                    notes.append(
                        f"No significant Windows EQ detected "
                        f"(max deviation {deviation:.1f} dB)."
                    )

        # ---- Headroom test -----------------------------------------
        progress_cb("stage2", 0.60, "Measuring digital headroom…")
        headroom_db = self._measure_headroom(device_idx, loopback_idx, fs)
        if headroom_db is not None:
            if headroom_db <= -6:
                notes.append(
                    f"Headroom warning: loopback clips at {headroom_db:+.0f} dBFS. "
                    "Reduce Windows output volume to avoid distortion."
                )
            else:
                notes.append(f"Digital headroom: {headroom_db:+.0f} dBFS (healthy).")

        progress_cb("stage2", 1.00)
        return {
            "driver_eq_gains": driver_eq_gains,
            "driver_eq_notes": " | ".join(notes),
            "headroom_db":     headroom_db,
            "notes":           notes,
        }

    def _find_wasapi_loopback(self, device_idx: int) -> int | None:
        """Return the WASAPI device index usable as a loopback input."""
        if not (_HAS_SD and _WASAPI_AVAILABLE):
            return None
        try:
            wasapi_ha = None
            for i, ha in enumerate(sd.query_hostapis()):
                if "WASAPI" in ha.get("name", ""):
                    wasapi_ha = i
                    break
            if wasapi_ha is None:
                return None

            target      = sd.query_devices(device_idx)
            target_lc   = target["name"].lower()

            # If device is already WASAPI, use it directly as loopback input
            if target.get("hostapi") == wasapi_ha:
                return device_idx

            # Otherwise find the WASAPI output device with a matching name
            for i, d in enumerate(sd.query_devices()):
                if d.get("hostapi") != wasapi_ha:
                    continue
                if d.get("max_output_channels", 0) == 0:
                    continue
                d_lc = d["name"].lower()
                if (target_lc in d_lc or d_lc in target_lc
                        or _name_prefix_match(target_lc, d_lc)):
                    return i
        except Exception:
            pass
        return None

    def _transfer_function_db(
        self, ref: np.ndarray, rec: np.ndarray, fs: int
    ) -> np.ndarray | None:
        """
        Compute H(f) = Y(f)/X(f) in dB at 48 log-spaced measurement frequencies.
        """
        n = min(len(ref), len(rec))
        if n < int(fs * 0.05):
            return None

        nfft  = 2 ** int(np.ceil(np.log2(n)))
        R     = np.fft.rfft(ref[:n].astype(np.float64), n=nfft)
        Y     = np.fft.rfft(rec[:n].astype(np.float64), n=nfft)
        f_bin = np.fft.rfftfreq(nfft, d=1.0 / fs)

        denom = np.abs(R)
        denom[denom < 1e-10] = 1e-10
        H_db  = 20.0 * np.log10(np.maximum(np.abs(Y) / denom, 1e-10))

        # Evaluate at measurement grid and smooth
        meas_f  = np.geomspace(30.0, 18_000.0, 48)
        H_meas  = np.interp(meas_f, f_bin, H_db)
        H_sm    = _fractional_octave_smooth(H_meas, meas_f, fraction=6)

        # Normalise to 300–3000 Hz midband reference
        ref_mask = (meas_f >= 300) & (meas_f <= 3_000)
        if np.any(ref_mask):
            H_sm -= float(np.mean(H_sm[ref_mask]))

        return H_sm

    def _measure_headroom(
        self, output_idx: int, loopback_idx: int, fs: int
    ) -> float | None:
        """
        Play a 1 kHz tone at increasing levels and find the highest dBFS level
        that passes through without clipping in the loopback capture.
        """
        test_f = 997.0   # slightly off 1 kHz to avoid power-line harmonics
        dur    = 0.08    # 80 ms per level
        n      = int(fs * dur)
        t      = np.arange(n, dtype=np.float64) / fs

        last_clean = None
        for level_dbfs in self._HEADROOM_LEVELS:
            amp    = 10 ** (level_dbfs / 20.0)
            tone   = (amp * np.sin(2 * np.pi * test_f * t)).astype(np.float32)
            signal = np.column_stack([tone, tone])
            try:
                rec = sd.playrec(
                    signal,
                    samplerate=fs,
                    input_device=loopback_idx,
                    output_device=output_idx,
                    input_extra_settings=_WASAPI_SETTINGS_CLS(loopback=True),
                    channels=2,
                    dtype="float32",
                )
                sd.wait()
            except Exception:
                return last_clean

            peak = float(np.percentile(np.abs(rec[:, 0]), 99))
            if peak > 0.95:
                # Clipping at this level — previous level was clean
                return last_clean if last_clean is not None else level_dbfs
            last_clean = level_dbfs

        return last_clean   # survived all levels

    # ------------------------------------------------------------------
    # Signal generation & processing helpers

    def _gen_ess(self, f1: float, f2: float, dur: float, fs: int) -> np.ndarray:
        """Generate an exponential sine sweep (Farina 2000)."""
        n   = int(dur * fs)
        t   = np.arange(n, dtype=np.float64) / fs
        L   = dur / np.log(f2 / f1)
        K   = 2.0 * np.pi * f1 * L
        phi = K * (np.exp(t / L) - 1.0)
        x   = np.sin(phi).astype(np.float32)
        # Fade in/out to suppress clicks
        fade = int(fs * 0.010)
        x[:fade]  *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        x[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return x

    def _deconvolve(
        self, ref: np.ndarray, rec: np.ndarray, fs: int
    ) -> np.ndarray | None:
        """Compute impulse response via Wiener deconvolution with regularisation."""
        n    = max(len(ref), len(rec))
        nfft = 2 ** int(np.ceil(np.log2(n * 2)))
        R    = np.fft.rfft(ref.astype(np.float64), n=nfft)
        Y    = np.fft.rfft(rec.astype(np.float64), n=nfft)

        # Tikhonov regularisation
        eps  = float(np.max(np.abs(R) ** 2)) * 1e-4
        H    = (Y * np.conj(R)) / (np.abs(R) ** 2 + eps)
        ir   = np.fft.irfft(H).astype(np.float32)

        # Align to direct-sound peak (search first 500 ms)
        search_n = min(int(fs * 0.5), len(ir))
        peak_idx = int(np.argmax(np.abs(ir[:search_n])))
        if peak_idx > 0:
            ir = np.roll(ir, -peak_idx)

        return ir[:int(fs * 0.1)]   # return first 100 ms


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _name_prefix_match(a: str, b: str, length: int = 14) -> bool:
    """True if two device name strings share a common prefix (after stripping common prefixes)."""
    def _strip(s: str) -> str:
        for pfx in ("speakers (", "headphones (", "microphone (", "line "):
            if s.startswith(pfx):
                s = s[len(pfx):]
        return s
    return _strip(a)[:length] == _strip(b)[:length]


def _fractional_octave_smooth(
    H_db: np.ndarray, freqs: np.ndarray, fraction: int = 6
) -> np.ndarray:
    """
    Smooth an array of dB values by a 1/N-octave moving average.
    fraction=6 → 1/6-octave, fraction=3 → 1/3-octave.
    """
    out = np.empty_like(H_db)
    for i, fc in enumerate(freqs):
        f_lo = fc / 2 ** (1.0 / (2 * fraction))
        f_hi = fc * 2 ** (1.0 / (2 * fraction))
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        out[i] = float(np.mean(H_db[mask])) if np.any(mask) else H_db[i]
    return out


def _response_to_band_gains(
    response_db: np.ndarray,
    response_freqs: np.ndarray,
    band_freqs: list,
) -> list[float]:
    """
    Map a smoothed frequency response to EQ band gains (log-nearest-neighbour).
    Gains are capped to ±8 dB to prevent over-correction.
    """
    gains = []
    for bfreq in band_freqs:
        dists   = np.abs(
            np.log(np.maximum(response_freqs, 1.0))
            - math.log(max(bfreq, 1.0))
        )
        closest = int(np.argmin(dists))
        gains.append(float(np.clip(response_db[closest], -8.0, 8.0)))
    return gains

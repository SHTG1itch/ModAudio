"""
Multi-device audio stream manager — ring-buffer / processing-thread architecture.

How it works
------------
Instead of running DSP inside audio I/O callbacks (which causes xruns when the
input and output clocks drift), audio flows through thread-safe ring buffers:

    Capture source  →  _in_ring  →  [_proc_thread]  →  _rear_ring  →  Rear output
                                                     ↘  _front_ring →  Front output (dual only)

The capture source can be:
  • WASAPI loopback via pyaudiowpatch  (loopback mode — no extra software needed)
  • Stereo Mix or any Windows input    (rear_only mode fallback)
  • An explicit capture device          (dual / Full Control mode)

The output callbacks (run by sounddevice on the hardware clock) simply drain
their ring buffer.  If the ring buffer is momentarily empty they output silence;
the processing thread catches up as soon as possible.  This absorbs clock drift
between different devices (e.g. HDMI monitor vs Bluetooth speaker).

Ring buffer capacity: 200 ms — large enough to absorb BT jitter, small enough
that the listener cannot perceive the buffering delay.

Modes
-----
loopback    WASAPI loopback captures the front device; rear only stream opened.
            Free, no virtual cable needed.  Front device plays Windows audio
            normally.  Rear device gets the derived surround channels.

rear_only   Like loopback but uses an explicit input device (Stereo Mix, VB-Cable)
            instead of pyaudiowpatch.

dual        Full control.  Three streams: capture + front output + rear output.
            Full theater DSP applied to both speakers.  Requires a virtual audio
            cable (e.g. VB-Cable) set as the Windows default output.

Bluetooth delay compensation
-----------------------------
BT codec latency is invisible to the OS.  We add a compensating digital delay to
the FASTER (wired) device.  The mode determines which stream we can delay:
  dual     : we own both output streams → can delay either one.
  loopback : only rear stream exists → can only delay the rear.
"""

from __future__ import annotations
import threading
import time
import numpy as np
import sounddevice as sd

from dsp.multi_speaker import MultiSpeakerChain
from config import SOUND_SPEED_MS

# ---------------------------------------------------------------------------
# Optional WASAPI loopback backend (pyaudiowpatch)
# ---------------------------------------------------------------------------
try:
    import pyaudiowpatch as _pyaw
    _HAS_PYAW = True
except ImportError:
    _pyaw = None          # type: ignore
    _HAS_PYAW = False


def _get_loopback_device_info(pa: "_pyaw.PyAudio", front_dev_idx: int) -> dict | None:
    """
    Return the pyaudiowpatch loopback device info for the given sounddevice
    output device index.

    Strategy
    --------
    1.  pa.get_wasapi_loopback_analogue_by_index() with the sd index directly.
    2.  Iterate all sounddevice WASAPI output indices around front_dev_idx to
        find an index that pyaudiowpatch recognises (handles index skew).
    3.  Name-based fallback — strip loopback suffix and fuzzy-match by prefix.
    """
    # Strategy 1: direct index lookup
    for candidate_idx in range(max(0, front_dev_idx - 4),
                                front_dev_idx + 5):
        try:
            info = pa.get_wasapi_loopback_analogue_by_index(candidate_idx)
            if info and int(info.get("maxInputChannels", 0)) > 0:
                # Verify name matches the requested device (not a random device)
                try:
                    sd_name = sd.query_devices(front_dev_idx)["name"].lower().strip()
                    lb_name = (info["name"].lower()
                               .replace(" [loopback]", "")
                               .replace(" (loopback)", "")
                               .strip())
                    if lb_name[:20] in sd_name or sd_name[:20] in lb_name:
                        return info
                    if candidate_idx == front_dev_idx:
                        return info   # exact index match — trust it regardless
                except Exception:
                    if candidate_idx == front_dev_idx:
                        return info
        except Exception:
            pass

    # Strategy 2: name-based scan of all loopback devices
    try:
        target = sd.query_devices(front_dev_idx)["name"].lower().strip()
    except Exception:
        return None

    best = None
    best_score = 0
    try:
        for lb in pa.get_loopback_device_info_generator():
            if int(lb.get("maxInputChannels", 0)) < 1:
                continue
            lb_name = (lb["name"].lower()
                       .replace(" [loopback]", "")
                       .replace(" (loopback)", "")
                       .strip())
            if lb_name == target:
                return lb
            # Score by common prefix length
            common = min(len(lb_name), len(target))
            score = sum(1 for i in range(common) if lb_name[i] == target[i])
            if score > best_score and score >= min(12, common):
                best_score = score
                best = lb
    except Exception:
        pass
    return best


def _find_stereo_mix_device() -> int | None:
    """Return the sounddevice index of any system-loopback input, or None.

    Searches for both Windows (Stereo Mix, VB-Cable) and macOS
    (BlackHole, Soundflower, VB-Cable for Mac) virtual loopback devices.
    """
    kw = (
        "stereo mix", "what u hear", "wave out mix",  # Windows system capture
        "loopback",                                    # generic
        "cable output", "vb-audio", "vb-cable",       # VB-Cable (Windows & macOS)
        "blackhole",                                   # BlackHole (macOS)
        "soundflower",                                 # Soundflower (macOS, legacy)
    )
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        if any(k in d["name"].lower() for k in kw):
            return i
    return None


# ---------------------------------------------------------------------------
# Bluetooth heuristic
# ---------------------------------------------------------------------------

_BT_KEYWORDS = (
    "bluetooth", " bt ", "bt-", " bt)", "(bt",
    "airpods", "beats", "powerbeats",
    "bose", "soundlink", "quietcomfort",
    "jbl", "charge ", "flip ", "pulse ", "boom ",
    "sony wh", "sony wf", "sony xb",
    "sennheiser", "momentum",
    "jabra", "evolve",
    "earbuds", "headset",
    "wireless", "freedom",
    "soundcore", "anker",
    "megaboom", "hyperboom",
    "marshall", "kilburn",
    "ultimate ears",
)


def is_bluetooth_device(device_name: str) -> bool:
    """Return True if the device name suggests Bluetooth / wireless audio."""
    nl = device_name.lower()
    return any(kw in nl for kw in _BT_KEYWORDS)


# ---------------------------------------------------------------------------
# A2DP codec delay estimation
# ---------------------------------------------------------------------------

_CODEC_PATTERNS: list[tuple[str, float]] = [
    ("aptx low latency", 40.0),
    ("aptx-ll",          40.0),
    ("aptx ll",          40.0),
    ("aptx hd",          80.0),
    ("aptx",             80.0),
    ("ldac",            100.0),
    ("aac",             130.0),
]
_DEFAULT_BT_CODEC_MS = 175.0   # SBC or unrecognised codec


def _estimate_codec_ms(device_name: str) -> float:
    nl = device_name.lower()
    for kw, ms in _CODEC_PATTERNS:
        if kw in nl:
            return ms
    return _DEFAULT_BT_CODEC_MS


# ---------------------------------------------------------------------------
# Thread-safe ring buffer
# ---------------------------------------------------------------------------

_FADE_LEN = 64   # samples (~1.3 ms at 48 kHz) — fade length for under-run edges


class _AudioRingBuffer:
    """
    Thread-safe single-writer / single-reader ring buffer for float32 audio.

    The write() method never blocks — if the buffer is full it silently drops
    the oldest data so the reader always gets recent audio.

    The read() method blocks until enough data is available or the timeout
    expires (returns None on timeout), making it safe to use in a worker
    thread.  The read_nb() variant never blocks.
    """

    def __init__(self, frames: int, channels: int = 2):
        self._cap   = frames
        self._ch    = channels
        self._buf   = np.zeros((frames, channels), dtype=np.float32)
        self._wpos  = 0
        self._rpos  = 0
        self._avail = 0
        self._cond  = threading.Condition(threading.Lock())
        # Output-side fade state (single reader) — see read_out()
        self._flowing  = False
        self._last_out = np.zeros(channels, dtype=np.float32)
        self.underruns = 0
        self.overruns  = 0

    # -- writer side (called from audio callback thread) -------------------

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        if n == 0:
            return
        with self._cond:
            if n > self._cap:
                data = data[-self._cap:]
                n = self._cap
            # Drop oldest if needed.  The reader-side fade state is cleared
            # so the splice point is faded back in rather than clicking.
            if self._avail + n > self._cap:
                drop = self._avail + n - self._cap
                self._rpos  = (self._rpos + drop) % self._cap
                self._avail -= drop
                self.overruns += 1
                self._flowing = False
            s1 = min(n, self._cap - self._wpos)
            self._buf[self._wpos:self._wpos + s1] = data[:s1]
            if s1 < n:
                self._buf[:n - s1] = data[s1:]
            self._wpos  = (self._wpos + n) % self._cap
            self._avail += n
            self._cond.notify_all()

    # -- reader side -------------------------------------------------------

    def _read_locked(self, n: int) -> np.ndarray:
        """Must be called with self._cond held."""
        s1 = min(n, self._cap - self._rpos)
        if s1 == n:
            out = self._buf[self._rpos:self._rpos + n].copy()
        else:
            out = np.empty((n, self._ch), dtype=np.float32)
            out[:s1] = self._buf[self._rpos:]
            out[s1:] = self._buf[:n - s1]
        self._rpos  = (self._rpos + n) % self._cap
        self._avail -= n
        return out

    def read(self, n: int, timeout: float = 0.05) -> np.ndarray | None:
        """Blocking read.  Returns None if data not available within timeout."""
        with self._cond:
            if self._avail < n:
                self._cond.wait(timeout=timeout)
            if self._avail < n:
                return None
            return self._read_locked(n)

    def read_nb(self, n: int) -> np.ndarray | None:
        """Non-blocking read.  Returns None if insufficient data."""
        with self._cond:
            if self._avail < n:
                return None
            return self._read_locked(n)

    def read_out(self, n: int) -> np.ndarray:
        """
        Non-blocking read for output callbacks: always returns exactly n
        frames.  If fewer are buffered, the shortfall is zero-padded and the
        valid→silence and silence→valid transitions are faded over
        _FADE_LEN samples so under-runs never produce a hard click.
        """
        with self._cond:
            take = min(self._avail, n)
            data = self._read_locked(take) if take > 0 else None

        out = np.zeros((n, self._ch), dtype=np.float32)
        fade = _FADE_LEN
        if data is not None:
            out[:take] = data

        if take > 0 and not self._flowing:
            # First audio after silence / an under-run / a writer drop:
            # fade the new audio in.
            f = min(fade, take)
            out[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]

        if take < n:
            # Entering (or continuing) an under-run
            self.underruns += 1
            if take > 0:
                # Fade out the tail of the valid audio into the silence
                f = min(fade, take)
                out[take - f:take] *= np.linspace(1.0, 0.0, f,
                                                  dtype=np.float32)[:, None]
            elif self._flowing:
                # No data at all but we were mid-signal: ramp the previous
                # sample down to zero instead of cutting instantly.
                f = min(fade, n)
                out[:f] = self._last_out[None, :] * np.linspace(
                    1.0, 0.0, f, dtype=np.float32)[:, None]
            self._flowing = False
        else:
            self._flowing = True
            self._last_out = out[-1].copy()
        return out

    @property
    def available(self) -> int:
        with self._cond:
            return self._avail

    def prefill(self, frames: int) -> None:
        """Write `frames` of silence (used to establish the target latency)."""
        self.write(np.zeros((frames, self._ch), dtype=np.float32))

    def reset(self) -> None:
        with self._cond:
            self._buf[:] = 0.0
            self._wpos  = 0
            self._rpos  = 0
            self._avail = 0
        self._flowing  = False
        self._last_out = np.zeros(self._ch, dtype=np.float32)


# ---------------------------------------------------------------------------
# Delay buffer (used for BT compensation)
# ---------------------------------------------------------------------------

class _DelayBuffer:
    """Ring-buffer delay line for stereo float32 blocks.

    Incoming audio is ALWAYS written to the buffer (even at delay 0), so the
    history is valid the moment the delay is raised.  When the requested
    delay changes between blocks, the output crossfades from the old tap to
    the new tap over one block instead of jumping the read pointer (which
    would click / zipper while a delay slider is dragged).
    """

    def __init__(self, max_delay_samples: int, channels: int = 2):
        size = max_delay_samples + 4096
        self._buf  = np.zeros((size, channels), dtype=np.float32)
        self._sz   = size
        self._ptr  = 0
        self._cur_delay = None   # delay used for the previous block

    def _read_tap(self, n: int, delay: int) -> np.ndarray:
        r = np.arange(self._ptr - delay - n, self._ptr - delay,
                      dtype=np.int64) % self._sz
        return self._buf[r]

    def process(self, x: np.ndarray, delay: int) -> np.ndarray:
        n = len(x)
        w = np.arange(self._ptr, self._ptr + n, dtype=np.int64) % self._sz
        self._buf[w] = x
        self._ptr = int((self._ptr + n) % self._sz)

        if self._cur_delay is None:
            self._cur_delay = delay

        if delay == self._cur_delay:
            if delay == 0:
                return x
            return self._read_tap(n, delay).copy()

        # Delay changed — crossfade old tap → new tap across this block
        old = self._read_tap(n, self._cur_delay)
        new = self._read_tap(n, delay)
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        out = old * (1.0 - ramp) + new * ramp
        self._cur_delay = delay
        return out.astype(np.float32)

    def reset(self):
        self._buf[:] = 0.0
        self._ptr = 0
        self._cur_delay = None


# ---------------------------------------------------------------------------
# Varispeed resampler (clock-drift servo)
# ---------------------------------------------------------------------------

class _Varispeed:
    """
    Phase-continuous linear-interpolation resampler used as a clock-drift
    servo.  Each output device runs on its own crystal; without rate
    correction the per-device ring buffers slowly fill (200 ms latency creep
    + drop-out splice) or drain (periodic under-runs).  The proc thread
    nudges each bus's resampling ratio a few hundred ppm at most — far below
    audibility (1000 ppm ≈ 1.7 cents) — to hold the ring fill at its target.
    """

    def __init__(self, channels: int = 2):
        self._ch    = channels
        self._last  = np.zeros((1, channels), dtype=np.float32)
        self._phase = 0.0     # fractional read position into [last | block]

    def process(self, x: np.ndarray, ratio: float) -> np.ndarray:
        """Resample block x by `ratio` (input samples per output sample)."""
        n = len(x)
        if n == 0:
            return x
        if ratio == 1.0 and self._phase == 0.0:
            self._last = x[-1:].copy()
            return x
        buf = np.concatenate([self._last, x])          # positions 0..n
        # Output positions in input-sample units; position 0 = previous
        # block's final sample, position n = x[-1].
        max_k = int(np.floor((n - self._phase) / ratio)) + 1
        pos = self._phase + np.arange(max_k, dtype=np.float64) * ratio
        pos = pos[pos <= n]
        idx  = pos.astype(np.int64)
        frac = (pos - idx).astype(np.float32)[:, None]
        idx1 = np.minimum(idx + 1, n)
        out = buf[idx] * (1.0 - frac) + buf[idx1] * frac
        self._phase = float(pos[-1] + ratio - n) if len(pos) else self._phase - n
        self._last  = x[-1:].copy()
        return np.ascontiguousarray(out, dtype=np.float32)

    def reset(self):
        self._last[:] = 0.0
        self._phase = 0.0


def _boost_thread_priority() -> None:
    """Raise the calling thread to pro-audio priority (Windows; no-op elsewhere)."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetThreadPriority(k32.GetCurrentThread(), 15)   # TIME_CRITICAL
        try:
            task_idx = ctypes.c_ulong(0)
            ctypes.windll.avrt.AvSetMmThreadCharacteristicsW(
                "Pro Audio", ctypes.byref(task_idx))
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Multi-device stream
# ---------------------------------------------------------------------------

_RING_MS   = 200          # ring buffer capacity in milliseconds
_PROC_TIMEOUT = 0.05      # proc thread wait timeout (seconds)
_TARGET_BLOCKS = 3        # output-ring fill target (blocks) — the actual
                          # buffering latency; servo holds fill here
_SERVO_TAU  = 5.0         # servo stiffness (seconds); steady-state error at
                          # 100 ppm device drift ≈ 24 samples
_SERVO_MAX  = 1.0e-3      # ±1000 ppm ratio clamp (≈1.7 cents, inaudible)


class MultiDeviceStream:
    """
    Routes audio to two physical speaker buses via a ring-buffer pipeline.

    Parameters
    ----------
    in_dev          : input device index (used in rear_only / dual modes)
    front_dev       : WASAPI output device index for the front speaker
    rear_dev        : WASAPI output device index for the rear speaker
    fs              : sample rate (Hz)
    block_size      : audio block size (samples, used for output streams)
    preset          : theater preset dict
    bt_delay_ms     : initial Bluetooth latency compensation (ms)
    swap_rear_lr    : swap L/R on rear bus for forward-facing rear speakers
    mode            : "loopback" | "rear_only" | "dual"
    front_gain      : front bus volume multiplier
    rear_gain       : rear bus volume multiplier
    bass_priority   : "equal" | "front" | "rear"
    rear_az_deg     : rear speaker azimuth in degrees (90–170°, default 150°).
                      150° = directly behind.  Lower values move the rear image
                      toward the sides.  Passed through to MultiSpeakerChain.
    acoustic_delay_ms : extra delay applied to the FRONT stream to compensate for
                        the rear speaker being physically farther from the listener
                        (0 = no compensation).  Independent of Bluetooth delay.
    front_dist_m      : physical distance of the front speaker from the listener
                        (metres).  When both distances are given and
                        acoustic_delay_ms is 0, wavefront alignment is computed
                        automatically: the NEARER speaker is delayed by the
                        path-length difference so both wavefronts arrive together
                        (otherwise the precedence effect pulls the image toward
                        the nearer speaker).
    rear_dist_m       : same, for the rear speaker.
    """

    def __init__(
        self,
        in_dev: int,
        front_dev: int,
        rear_dev: int,
        fs: int = 48000,
        block_size: int = 512,
        preset: dict | None = None,
        bt_delay_ms: float = 150.0,
        swap_rear_lr: bool = True,
        mode: str = "loopback",
        front_gain: float = 1.0,
        rear_gain: float = 1.0,
        bass_priority: str = "equal",
        rear_az_deg: float = 150.0,
        acoustic_delay_ms: float = 0.0,
        front_info: tuple | None = None,
        rear_info:  tuple | None = None,
        front_eq=None,
        rear_eq=None,
        front_dist_m: float | None = None,
        rear_dist_m:  float | None = None,
    ):
        self._in_dev    = in_dev
        self._front_dev = front_dev
        self._rear_dev  = rear_dev
        self._fs        = fs
        self._bs        = block_size
        self._mode      = mode   # set FIRST — _apply_bt_delay needs it

        self._chain = MultiSpeakerChain(fs=fs, preset=preset,
                                        bass_priority=bass_priority,
                                        rear_az_deg=rear_az_deg,
                                        front_info=front_info,
                                        rear_info=rear_info,
                                        front_eq=front_eq,
                                        rear_eq=rear_eq)

        # -- Bluetooth detection ------------------------------------------
        devs = sd.query_devices()
        front_name = devs[front_dev]["name"] if front_dev < len(devs) else ""
        rear_name  = devs[rear_dev]["name"]  if rear_dev  < len(devs) else ""
        self.front_is_bt = is_bluetooth_device(front_name)
        self.rear_is_bt  = is_bluetooth_device(rear_name)
        self.front_name  = front_name
        self.rear_name   = rear_name

        # -- Delay compensation -------------------------------------------
        max_d = int(round(500.0 * fs / 1000.0)) + block_size + 256
        self._front_delay_buf = _DelayBuffer(max_d, channels=2)
        self._rear_delay_buf  = _DelayBuffer(max_d, channels=2)
        self._front_comp_delay = 0
        self._rear_comp_delay  = 0
        self._apply_bt_delay(bt_delay_ms)

        # -- Acoustic (distance) delay — aligns wavefront arrival times when
        # the speakers are at different distances from the listener.  Kept
        # separate from Bluetooth compensation so a BT-delay update can never
        # silently wipe the acoustic alignment (and vice versa); the two are
        # summed at processing time.
        #   Manual mode : acoustic_delay_ms > 0 delays the front stream.
        #   Auto mode   : both distances known → delay the NEARER speaker by
        #                 the path-length difference (d/343 m·s⁻¹), otherwise
        #                 the precedence effect drags the image toward it.
        self._front_dist_delay = 0
        self._rear_dist_delay  = 0
        acou_samp = int(round(float(acoustic_delay_ms) * fs / 1000.0))
        if acou_samp > 0:
            self._front_dist_delay = acou_samp
        elif front_dist_m is not None and rear_dist_m is not None:
            diff_samp = int(round((float(rear_dist_m) - float(front_dist_m))
                                  / SOUND_SPEED_MS * fs))
            if diff_samp > 0:
                self._front_dist_delay = min(diff_samp, max_d - block_size - 256)
            elif diff_samp < 0:
                self._rear_dist_delay = min(-diff_samp, max_d - block_size - 256)
        self._acoustic_delay_ms = float(acoustic_delay_ms)
        self._front_dist_m = front_dist_m
        self._rear_dist_m  = rear_dist_m

        # -- Per-bus gains — applied inside the chain, BEFORE the limiter,
        # so a boosted bus is still peak-controlled at the device.
        self._front_gain   = float(front_gain)
        self._rear_gain    = float(rear_gain)
        self._chain.set_bus_gains(self._front_gain, self._rear_gain)
        self._swap_rear_lr = swap_rear_lr

        # -- Ring buffers -------------------------------------------------
        ring_frames = int(round(_RING_MS * fs / 1000.0))
        self._in_ring    = _AudioRingBuffer(ring_frames, channels=2)
        self._front_ring = _AudioRingBuffer(ring_frames, channels=2)
        self._rear_ring  = _AudioRingBuffer(ring_frames, channels=2)

        # -- Clock-drift servo (one varispeed per output bus) --------------
        self._fill_target = _TARGET_BLOCKS * block_size
        self._front_vs    = _Varispeed(2)
        self._rear_vs     = _Varispeed(2)
        self._front_fill  = float(self._fill_target)
        self._rear_fill   = float(self._fill_target)

        # Set when an output stream dies unexpectedly (BT disconnect etc.)
        self.device_error: str | None = None

        # -- Streams / threads --------------------------------------------
        self._lb_pa      = None   # pyaudiowpatch PyAudio instance
        self._lb_stream  = None   # pyaudiowpatch stream
        self._lb_ch      = 2      # channels in loopback capture
        self._sd_in      = None   # sounddevice InputStream (rear_only/dual)
        self._front_sd   = None   # sounddevice OutputStream (dual)
        self._rear_sd    = None   # sounddevice OutputStream (loopback/rear_only/dual)

        self._proc_thread  = None
        self._running      = False
        self._stop_event   = threading.Event()

        # -- Metering -----------------------------------------------------
        self.xruns         = 0
        self.raw_in        = np.zeros(2, dtype=np.float32)
        self.raw_out_front = np.zeros(2, dtype=np.float32)
        self.raw_out_rear  = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Delay management
    # ------------------------------------------------------------------ #

    def _apply_bt_delay(self, bt_delay_ms: float) -> None:
        """
        Set compensating delays.

        dual mode: we own both output streams.
          rear BT, front wired  → delay the wired FRONT
          front BT, rear wired  → delay the wired REAR

        loopback/rear_only: only the rear stream exists.
          front BT, rear wired  → delay the wired REAR to match BT front
          rear BT, front wired  → BT lag is unavoidable (no front stream)
        """
        samples = int(round(bt_delay_ms * self._fs / 1000.0))

        if self._mode == "dual":
            if self.rear_is_bt and not self.front_is_bt:
                self._front_comp_delay = samples
                self._rear_comp_delay  = 0
            elif self.front_is_bt and not self.rear_is_bt:
                self._front_comp_delay = 0
                self._rear_comp_delay  = samples
            else:
                self._front_comp_delay = 0
                self._rear_comp_delay  = 0
        else:
            if self.front_is_bt and not self.rear_is_bt:
                self._front_comp_delay = 0
                self._rear_comp_delay  = samples
            else:
                self._front_comp_delay = 0
                self._rear_comp_delay  = 0

    def update_bt_delay(self, bt_delay_ms: float) -> None:
        self._apply_bt_delay(bt_delay_ms)

    def update_swap_rear_lr(self, swap: bool) -> None:
        self._swap_rear_lr = swap

    def update_front_gain(self, gain: float) -> None:
        self._front_gain = float(gain)
        self._chain.set_bus_gains(self._front_gain, self._rear_gain)

    def update_rear_gain(self, gain: float) -> None:
        self._rear_gain = float(gain)
        self._chain.set_bus_gains(self._front_gain, self._rear_gain)

    def update_bass_priority(self, priority: str) -> None:
        self._chain.set_bass_priority(priority)

    def update_rear_az(self, rear_az_deg: float) -> None:
        """Update rear speaker azimuth and rebuild VBAP routing matrix."""
        self._chain.update_rear_az(rear_az_deg)

    def update_speaker_info(self, front_info: tuple, rear_info: tuple) -> None:
        """Update both speakers' position/orientation and rebuild routing matrix."""
        self._chain.update_speaker_info(front_info, rear_info)

    @property
    def front_delay_ms(self) -> float:
        return (self._front_comp_delay + self._front_dist_delay) * 1000.0 / self._fs

    @property
    def rear_delay_ms(self) -> float:
        return (self._rear_comp_delay + self._rear_dist_delay) * 1000.0 / self._fs

    # ------------------------------------------------------------------ #
    # Auto-calibration
    # ------------------------------------------------------------------ #

    def calibrate_bt_delay_ms(self) -> float:
        """
        Estimate BT compensation delay from WASAPI-reported stream latencies.
        """
        def measure_ms(dev_idx: int) -> float:
            info = sd.query_devices(dev_idx, "output")
            ch   = min(int(info["max_output_channels"]), 2)
            try:
                with sd.OutputStream(
                    samplerate=self._fs,
                    blocksize=self._bs,
                    device=dev_idx,
                    channels=ch,
                    dtype="float32",
                    latency="low",
                ) as st:
                    return float(st.latency) * 1000.0
            except Exception:
                return float(info.get("default_low_output_latency", 0.05)) * 1000.0

        rear_ms = measure_ms(self._rear_dev)

        if self._mode in ("loopback", "rear_only"):
            if self.rear_is_bt:
                wired_front_est = 15.0
                diff = rear_ms - wired_front_est
                if diff < 30.0:
                    diff += _estimate_codec_ms(self.rear_name)
                return max(0.0, diff)
            elif self.front_is_bt:
                front_ms = measure_ms(self._front_dev)
                diff = front_ms - rear_ms
                if diff < 30.0:
                    diff += _estimate_codec_ms(self.front_name)
                return max(0.0, diff)
            else:
                return max(0.0, rear_ms)

        front_ms = measure_ms(self._front_dev)
        if self.rear_is_bt and not self.front_is_bt:
            diff = rear_ms - front_ms
            if diff < 30.0:
                diff += _estimate_codec_ms(self.rear_name)
            return max(0.0, diff)
        elif self.front_is_bt and not self.rear_is_bt:
            diff = front_ms - rear_ms
            if diff < 30.0:
                diff += _estimate_codec_ms(self.front_name)
            return max(0.0, diff)
        return max(0.0, abs(rear_ms - front_ms))

    # ------------------------------------------------------------------ #
    # Chain rebuild
    # ------------------------------------------------------------------ #

    def update_chain(self, preset: dict) -> None:
        old = self._chain
        new = MultiSpeakerChain(
            fs=self._fs, preset=preset,
            bass_priority=old._bass_priority if old is not None else "equal",
            rear_az_deg=old._rear_az_deg     if old is not None else 150.0,
            front_info=old._front_info       if old is not None else None,
            rear_info=old._rear_info         if old is not None else None,
            front_eq=old._front_eq           if old is not None else None,
            rear_eq=old._rear_eq             if old is not None else None,
        )
        new.set_bus_gains(self._front_gain, self._rear_gain)
        self._chain = new

    def update_speaker_eqs(self, front_eq, rear_eq) -> None:
        """Attach custom EQ objects to front/rear buses."""
        if self._chain is not None:
            self._chain.set_bus_eq("front", front_eq)
            self._chain.set_bus_eq("rear",  rear_eq)

    # ------------------------------------------------------------------ #
    # Processing thread
    # ------------------------------------------------------------------ #

    def _proc_loop(self) -> None:
        """
        Dedicated DSP thread: reads from _in_ring, processes, writes to
        _front_ring and _rear_ring.

        Runs continuously until _stop_event is set.  The ring buffer read()
        call blocks for up to _PROC_TIMEOUT seconds waiting for data, so the
        thread exits cleanly within that window after stop() is called.
        """
        _boost_thread_priority()
        bs = self._bs

        while not self._stop_event.is_set():
            block = self._in_ring.read(bs, timeout=_PROC_TIMEOUT)
            if block is None:
                continue   # timeout — loop to check stop_event

            # Meter input
            sq = block * block
            self.raw_in[:] = np.sqrt([sq[:, 0].mean(), sq[:, 1].mean()])

            chain = self._chain   # atomic GIL read
            try:
                front, rear = chain.process(block)
            except Exception as exc:
                print(f"[multi/dsp] {exc}")
                continue

            # Delay compensation: Bluetooth + acoustic distance alignment.
            # (Always run so the delay-line history stays valid and delay
            # changes crossfade cleanly; gains are applied inside the chain
            # before the limiter.)
            f_delay = self._front_comp_delay + self._front_dist_delay
            r_delay = self._rear_comp_delay  + self._rear_dist_delay
            front = self._front_delay_buf.process(front, f_delay)
            rear  = self._rear_delay_buf.process(rear, r_delay)

            # Swap rear L/R if speaker faces listener
            if self._swap_rear_lr:
                rear = rear[:, ::-1].copy()

            # Meter outputs
            sq2 = front * front
            self.raw_out_front[:] = np.sqrt([sq2[:, 0].mean(), sq2[:, 1].mean()])
            sq3 = rear * rear
            self.raw_out_rear[:]  = np.sqrt([sq3[:, 0].mean(), sq3[:, 1].mean()])

            # Clock-drift servo + write to output ring buffers.  Each output
            # device consumes on its own crystal; a varispeed per bus holds
            # the ring fill at the target instead of letting it drain
            # (under-runs) or creep to capacity (latency + drop splices).
            if self._mode == "dual" and self._front_ring is not None:
                front = self._servo_bus(front, self._front_ring,
                                        self._front_vs, "_front_fill")
                self._front_ring.write(front)
            rear = self._servo_bus(rear, self._rear_ring,
                                   self._rear_vs, "_rear_fill")
            self._rear_ring.write(rear)

    def _servo_bus(self, block, ring, vs, fill_attr):
        """EMA the ring fill and varispeed the block toward the fill target."""
        fill = 0.9 * getattr(self, fill_attr) + 0.1 * ring.available
        setattr(self, fill_attr, fill)
        err = (fill - self._fill_target) / (self._fs * _SERVO_TAU)
        ratio = 1.0 + float(np.clip(err, -_SERVO_MAX, _SERVO_MAX))
        # After a hard under-run the fill sits at ~0 — refill to the target
        # in one (faded) step rather than clicking for half a minute.
        if ring.available == 0 and fill < self._bs:
            ring.prefill(self._fill_target)
            setattr(self, fill_attr, float(self._fill_target))
        return vs.process(block, ratio)

    # ------------------------------------------------------------------ #
    # Input callbacks (write to _in_ring)
    # ------------------------------------------------------------------ #

    def _sd_input_cb(self, indata: np.ndarray, frames: int,
                     time_info, status) -> None:
        """sounddevice InputStream callback → _in_ring."""
        if status:
            self.xruns += 1
        ch = indata.shape[1]
        if ch >= 2:
            block = np.ascontiguousarray(indata[:, :2], dtype=np.float32)
        else:
            block = np.column_stack([indata[:, 0], indata[:, 0]]).astype(np.float32)
        self._in_ring.write(block)

    def _pyaw_input_cb(self, in_data: bytes, frame_count: int,
                       time_info, status_flags: int):
        """pyaudiowpatch loopback callback → _in_ring."""
        ch = self._lb_ch
        arr = np.frombuffer(in_data, dtype=np.float32).reshape(frame_count, ch)
        if ch == 1:
            block = np.column_stack([arr[:, 0], arr[:, 0]]).astype(np.float32)
        else:
            block = np.ascontiguousarray(arr[:, :2], dtype=np.float32)
        self._in_ring.write(block)
        return (None, _pyaw.paContinue)

    # ------------------------------------------------------------------ #
    # Output callbacks (drain ring buffers)
    # ------------------------------------------------------------------ #

    def _front_out_cb(self, outdata: np.ndarray, frames: int,
                      time_info, status) -> None:
        """sounddevice front OutputStream callback — drains _front_ring."""
        if status:
            self.xruns += 1
        block = self._front_ring.read_out(frames)
        out_ch = min(outdata.shape[1], 2)
        outdata[:, :out_ch] = block[:, :out_ch]
        if outdata.shape[1] > out_ch:
            outdata[:, out_ch:] = 0.0

    def _rear_out_cb(self, outdata: np.ndarray, frames: int,
                     time_info, status) -> None:
        """sounddevice rear OutputStream callback — drains _rear_ring."""
        if status:
            self.xruns += 1
        block = self._rear_ring.read_out(frames)
        out_ch = min(outdata.shape[1], 2)
        outdata[:, :out_ch] = block[:, :out_ch]
        if outdata.shape[1] > out_ch:
            outdata[:, out_ch:] = 0.0

    # ------------------------------------------------------------------ #
    # Start / Stop
    # ------------------------------------------------------------------ #

    def _make_finished_cb(self, which: str):
        """Flag unexpected stream death (BT disconnect) so the UI can react."""
        def _cb():
            if self._running:
                self.device_error = which
        return _cb

    def start(self) -> None:
        """Open streams and start the processing thread."""
        if self._running:
            return

        self._stop_event.clear()
        self.device_error = None
        self._in_ring.reset()
        self._front_ring.reset()
        self._rear_ring.reset()
        self._front_delay_buf.reset()
        self._rear_delay_buf.reset()
        self._front_vs.reset()
        self._rear_vs.reset()
        # Prefill output rings to the servo target so playback starts at a
        # known, small buffering latency instead of a random fill level.
        self._front_ring.prefill(self._fill_target)
        self._rear_ring.prefill(self._fill_target)
        self._front_fill = float(self._fill_target)
        self._rear_fill  = float(self._fill_target)

        # Query rear device channel count
        rear_info = sd.query_devices(self._rear_dev, "output")
        rear_ch   = min(int(rear_info["max_output_channels"]), 2)
        # WASAPI shared mode: "low" works for Bluetooth endpoints too — the
        # BT stack buffers downstream of WASAPI, so "high" only added latency.
        rear_latency = "low"

        # ── Loopback mode ────────────────────────────────────────────────
        if self._mode == "loopback":
            pyaw_err = None

            if _HAS_PYAW:
                try:
                    self._lb_pa  = _pyaw.PyAudio()
                    lb_info = _get_loopback_device_info(self._lb_pa, self._front_dev)
                    if lb_info is None:
                        self._lb_pa.terminate()
                        self._lb_pa = None
                        dev_name = sd.query_devices(self._front_dev)["name"]
                        raise RuntimeError(
                            f"No WASAPI loopback device found for '{dev_name}'.\n"
                            "Select the [WASAPI] variant of the front device.")

                    self._lb_ch = max(1, min(int(lb_info["maxInputChannels"]), 2))
                    lb_idx = int(lb_info["index"])
                    # Always use our pipeline sample rate so the ring buffer
                    # and output stream are in sync.  WASAPI loopback will
                    # resample internally if the hardware runs at a different
                    # native rate (e.g. 44100 → 48000).
                    lb_fs  = self._fs

                    self._lb_stream = self._lb_pa.open(
                        format=_pyaw.paFloat32,
                        channels=self._lb_ch,
                        rate=lb_fs,
                        input=True,
                        input_device_index=lb_idx,
                        frames_per_buffer=self._bs,
                        stream_callback=self._pyaw_input_cb,
                    )
                    pyaw_err = None

                except Exception as exc:
                    if self._lb_pa:
                        try:
                            self._lb_pa.terminate()
                        except Exception:
                            pass
                    self._lb_pa     = None
                    self._lb_stream = None
                    pyaw_err = str(exc)
            else:
                pyaw_err = "pyaudiowpatch not installed — pip install pyaudiowpatch"

            # Fallback: Stereo Mix or any loopback input
            if self._lb_stream is None:
                sm_idx = _find_stereo_mix_device()
                if sm_idx is not None:
                    sm_info = sd.query_devices(sm_idx, "input")
                    sm_ch   = min(int(sm_info["max_input_channels"]), 2)
                    self._sd_in = sd.InputStream(
                        samplerate=self._fs,
                        blocksize=self._bs,
                        device=sm_idx,
                        channels=sm_ch,
                        dtype="float32",
                        callback=self._sd_input_cb,
                        latency="low",
                    )
                else:
                    raise RuntimeError(
                        "Could not start loopback capture.\n\n"
                        f"pyaudiowpatch: {pyaw_err}\n"
                        "Loopback device: not found on this system.\n\n"
                        "Solutions:\n"
                        "  Windows:\n"
                        "    1. pip install pyaudiowpatch  (recommended)\n"
                        "    2. Enable Stereo Mix in Sound > Recording tab\n"
                        "  macOS:\n"
                        "    1. Install VB-Cable for Mac or BlackHole\n"
                        "    2. Set it as the system audio output\n"
                        "  All platforms:\n"
                        "    Switch to Full Control mode and select the\n"
                        "    loopback device as the Capture source.")

            self._rear_sd = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._rear_dev,
                channels=rear_ch,
                dtype="float32",
                callback=self._rear_out_cb,
                latency=rear_latency,
                finished_callback=self._make_finished_cb(self.rear_name),
            )

        # ── Rear-only mode ───────────────────────────────────────────────
        elif self._mode == "rear_only":
            in_info = sd.query_devices(self._in_dev, "input")
            in_ch   = min(int(in_info["max_input_channels"]), 2)
            self._sd_in = sd.InputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._in_dev,
                channels=in_ch,
                dtype="float32",
                callback=self._sd_input_cb,
                latency="low",
            )
            self._rear_sd = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._rear_dev,
                channels=rear_ch,
                dtype="float32",
                callback=self._rear_out_cb,
                latency=rear_latency,
                finished_callback=self._make_finished_cb(self.rear_name),
            )

        # ── Dual (Full Control) mode ─────────────────────────────────────
        else:
            in_info    = sd.query_devices(self._in_dev,    "input")
            front_info = sd.query_devices(self._front_dev, "output")
            in_ch    = min(int(in_info["max_input_channels"]),    2)
            front_ch = min(int(front_info["max_output_channels"]), 2)

            self._sd_in = sd.InputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._in_dev,
                channels=in_ch,
                dtype="float32",
                callback=self._sd_input_cb,
                latency="low",
            )
            self._front_sd = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._front_dev,
                channels=front_ch,
                dtype="float32",
                callback=self._front_out_cb,
                latency="low",
                finished_callback=self._make_finished_cb(self.front_name),
            )
            self._rear_sd = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._rear_dev,
                channels=rear_ch,
                dtype="float32",
                callback=self._rear_out_cb,
                latency=rear_latency,
                finished_callback=self._make_finished_cb(self.rear_name),
            )

        # ── Start pipeline ───────────────────────────────────────────────
        # Proc thread first so the output rings are being refilled from the
        # moment the output callbacks begin draining the prefilled silence —
        # starting outputs first guaranteed start-up under-runs and left the
        # ring fill (i.e. the buffering latency) at a random level.
        try:
            self._stop_event.clear()
            self._proc_thread = threading.Thread(
                target=self._proc_loop, name="MultiSpeaker-DSP", daemon=True)
            self._proc_thread.start()

            if self._lb_stream is not None:
                self._lb_stream.start_stream()
            if self._sd_in is not None:
                self._sd_in.start()
            if self._front_sd is not None:
                self._front_sd.start()
            self._rear_sd.start()
        except Exception:
            self._running = True   # let stop() tear everything down
            self.stop()
            raise

        self._running = True

    def stop(self) -> None:
        """Stop the processing thread and close all streams."""
        self._running = False
        self._stop_event.set()

        # Wake the proc thread if it's blocked on ring buffer read
        self._in_ring.write(np.zeros((self._bs, 2), dtype=np.float32))

        if self._proc_thread is not None:
            self._proc_thread.join(timeout=1.0)
            self._proc_thread = None

        # Stop pyaudiowpatch loopback
        if self._lb_stream is not None:
            try:
                if self._lb_stream.is_active():
                    self._lb_stream.stop_stream()
                self._lb_stream.close()
            except Exception:
                pass
            self._lb_stream = None
        if self._lb_pa is not None:
            try:
                self._lb_pa.terminate()
            except Exception:
                pass
            self._lb_pa = None

        # Stop sounddevice streams
        for s in (self._sd_in, self._front_sd, self._rear_sd):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._sd_in    = None
        self._front_sd = None
        self._rear_sd  = None

        self._in_ring.reset()
        self._front_ring.reset()
        self._rear_ring.reset()
        self._front_delay_buf.reset()
        self._rear_delay_buf.reset()

    @property
    def running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# N-speaker stream  (Full-Control / dual mode only)
# ---------------------------------------------------------------------------

class MultiSpeakerStreamN:
    """
    Routes captured audio to N physical speaker buses.

    Architecture
    ------------
    Capture device  →  _in_ring  →  [_proc_thread]  →  _out_rings[0..N-1]
                                                      ↓
                                             N sounddevice OutputStreams

    Each output stream drains its own ring buffer independently, so N
    different devices (wired, Bluetooth, HDMI…) can be driven simultaneously
    without clock synchronisation issues.

    Only "dual" (Full Control) mode is supported: a virtual audio cable
    (VB-Cable, ModAudio Surround, etc.) must be set as the system default
    output so that all application audio is captured here.

    Parameters
    ----------
    in_dev             : capture device index (VB-Cable / loopback source)
    speaker_devs       : list of N output device indices, one per speaker
    speaker_azimuths   : list of N azimuth angles in degrees
    speaker_elevations : list of N elevation angles in degrees (optional, default 0)
    speaker_face_azs   : list of N facing azimuths in degrees (optional,
                         default: each speaker facing toward the listener)
    speaker_face_els   : list of N facing elevations in degrees (optional, default 0)
    speaker_distances  : list of N distances from the listener in metres
                         (optional).  When given, automatic time + level
                         alignment is applied per speaker: nearer speakers are
                         digitally delayed so all wavefronts arrive at the
                         listener simultaneously (otherwise the precedence
                         effect pulls the entire image toward the nearest
                         speaker), and attenuated by the inverse-distance law
                         so all speakers are equally loud at the listening
                         position.
    fs                 : sample rate (Hz)
    block_size         : audio block size (samples)
    preset             : theater preset dict
    bt_delay_ms        : global Bluetooth compensation (ms)
    gains              : list of N per-speaker volume multipliers (default 1.0)
    bass_priority      : "equal" | "front" | "rear"
    """

    def __init__(
        self,
        in_dev: int,
        speaker_devs: list,
        speaker_azimuths: list,
        speaker_elevations: list | None = None,
        speaker_face_azs:   list | None = None,
        speaker_face_els:   list | None = None,
        speaker_distances:  list | None = None,
        fs: int = 48000,
        block_size: int = 512,
        preset: dict | None = None,
        bt_delay_ms: float = 150.0,
        gains: list | None = None,
        bass_priority: str = "equal",
        speaker_eqs: list | None = None,
    ):
        if len(speaker_devs) < 1:
            raise ValueError("speaker_devs must have at least one device")
        if len(speaker_azimuths) != len(speaker_devs):
            raise ValueError("speaker_devs and speaker_azimuths must have the same length")

        from dsp.multi_speaker import MultiSpeakerChainN
        self._in_dev       = in_dev
        self._speaker_devs = list(speaker_devs)
        self._N            = len(speaker_devs)
        self._fs           = fs
        self._bs           = block_size

        # Gains per speaker
        if gains is None:
            gains = [1.0] * self._N
        self._gains = [float(g) for g in gains]

        # DSP chain — pass full position+orientation info
        self._chain = MultiSpeakerChainN(
            fs=fs,
            preset=preset,
            speaker_azimuths=speaker_azimuths,
            speaker_elevations=speaker_elevations,
            speaker_face_azs=speaker_face_azs,
            speaker_face_els=speaker_face_els,
            bass_priority=bass_priority,
            speaker_eqs=speaker_eqs,
        )

        # Bluetooth delay — applied as a digital delay to the faster speakers
        devs = sd.query_devices()
        self._is_bt  = []
        self._names  = []
        for d in speaker_devs:
            nm = devs[d]["name"] if d < len(devs) else ""
            self._names.append(nm)
            self._is_bt.append(is_bluetooth_device(nm))

        max_d = int(round(500.0 * fs / 1000.0)) + block_size + 256
        self._max_delay  = max_d - block_size - 256
        self._delay_bufs = [_DelayBuffer(max_d, channels=2)
                            for _ in range(self._N)]
        self._delay_samp = [0] * self._N
        self._apply_bt_delay(bt_delay_ms)

        # Per-speaker distance compensation (time + level alignment)
        self._dist_delay_samp = [0] * self._N
        self._dist_gain       = [1.0] * self._N
        self._apply_distances(speaker_distances)

        # Ring buffers  (one per output device)
        ring_frames = int(round(_RING_MS * fs / 1000.0))
        self._in_ring   = _AudioRingBuffer(ring_frames, channels=2)
        self._out_rings = [_AudioRingBuffer(ring_frames, channels=2)
                           for _ in range(self._N)]

        # Clock-drift servo (one varispeed per output device)
        self._fill_target = _TARGET_BLOCKS * block_size
        self._vs    = [_Varispeed(2) for _ in range(self._N)]
        self._fills = [float(self._fill_target)] * self._N

        # Set when an output stream dies unexpectedly (BT disconnect etc.)
        self.device_error: str | None = None

        # Streams / threads
        self._sd_in      = None
        self._sd_outs    = [None] * self._N
        self._proc_thread = None
        self._running     = False
        self._stop_event  = threading.Event()

        # Metering  (backward-compat aliases + per-speaker list)
        self.xruns          = 0
        self.raw_in         = np.zeros(2, dtype=np.float32)
        self.raw_out        = [np.zeros(2, dtype=np.float32)
                               for _ in range(self._N)]
        # Keep front/rear aliases for code that uses MultiDeviceStream API
        self.raw_out_front  = self.raw_out[0]
        self.raw_out_rear   = self.raw_out[min(1, self._N - 1)]

    # ------------------------------------------------------------------ #
    # Delay management
    # ------------------------------------------------------------------ #

    def _apply_bt_delay(self, bt_delay_ms: float) -> None:
        samples = int(round(bt_delay_ms * self._fs / 1000.0))
        # Delay all wired speakers to match the slowest Bluetooth speaker
        any_bt = any(self._is_bt)
        for i in range(self._N):
            if any_bt and not self._is_bt[i]:
                self._delay_samp[i] = samples
            else:
                self._delay_samp[i] = 0

    def update_bt_delay(self, bt_delay_ms: float) -> None:
        self._apply_bt_delay(bt_delay_ms)

    def _apply_distances(self, distances: list | None) -> None:
        """Compute per-speaker time + level alignment from listener distances.

        Nearer speakers are delayed by the path-length difference to the
        farthest speaker (wavefront alignment) and attenuated by d_i/d_max
        (inverse-distance level matching at the listening position).
        """
        if not distances or len(distances) != self._N:
            self._dist_delay_samp = [0] * self._N
            self._dist_gain       = [1.0] * self._N
            self._push_gains()
            return
        ds = [max(0.1, float(d)) for d in distances]
        d_max = max(ds)
        self._dist_delay_samp = [
            min(self._max_delay,
                int(round((d_max - d) / SOUND_SPEED_MS * self._fs)))
            for d in ds
        ]
        # Clamp so a speaker very close to the listener isn't muted entirely
        self._dist_gain = [max(0.25, d / d_max) for d in ds]
        self._push_gains()

    def update_speakers(
        self,
        azimuths:   list,
        elevations: list | None = None,
        face_azs:   list | None = None,
        face_els:   list | None = None,
        distances:  list | None = None,
    ) -> None:
        """Update speaker positions/orientations and rebuild VBAP routing matrix."""
        self._chain.update_speakers(azimuths, elevations, face_azs, face_els)
        if distances is not None:
            self._apply_distances(distances)

    def update_speaker_azimuths(self, azimuths: list) -> None:
        """Backward-compatible alias — updates azimuths only."""
        self._chain.update_speakers(azimuths)

    def update_chain(self, preset: dict) -> None:
        """Rebuild DSP chain with a new theater preset, preserving speaker layout."""
        from dsp.multi_speaker import MultiSpeakerChainN
        old = self._chain
        azimuths   = list(old._azimuths)
        spk_info   = list(old._speaker_info)   # [(az, el, face_az, face_el), ...]
        elevations = [si[1] for si in spk_info]
        face_azs   = [si[2] for si in spk_info]
        face_els   = [si[3] for si in spk_info]
        self._chain = MultiSpeakerChainN(
            fs=self._fs,
            preset=preset,
            speaker_azimuths=azimuths,
            speaker_elevations=elevations,
            speaker_face_azs=face_azs,
            speaker_face_els=face_els,
            bass_priority=old._bass_priority,
            speaker_eqs=list(old._speaker_eqs) if old._speaker_eqs else None,
        )
        self._push_gains()

    def update_speaker_eqs(self, eqs: list) -> None:
        """Attach per-speaker custom EQ objects to the running chain."""
        if self._chain is not None:
            self._chain.set_all_speaker_eqs(eqs)

    def _push_gains(self) -> None:
        """Push user × distance gains into the chain (applied pre-limiter)."""
        combined = [
            (self._gains[i] if i < len(self._gains) else 1.0)
            * (self._dist_gain[i] if i < len(self._dist_gain) else 1.0)
            for i in range(self._N)
        ]
        self._chain.set_output_gains(combined)

    def update_gains(self, gains: list) -> None:
        self._gains = [float(g) for g in gains]
        self._push_gains()

    def update_bass_priority(self, priority: str) -> None:
        self._chain.set_bass_priority(priority)

    def update_front_gain(self, gain: float) -> None:
        if self._gains:
            self._gains[0] = float(gain)
        self._push_gains()

    def update_rear_gain(self, gain: float) -> None:
        for i in range(1, len(self._gains)):
            self._gains[i] = float(gain)
        self._push_gains()

    # ------------------------------------------------------------------ #
    # Start / stop
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._running:
            return
        try:
            self._start_inner()
        except Exception:
            self._running = True   # let stop() tear down whatever started
            self.stop()
            raise

    def _start_inner(self) -> None:
        self._stop_event.clear()
        self.device_error = None
        self._in_ring.reset()
        for ring, vs in zip(self._out_rings, self._vs):
            ring.reset()
            vs.reset()
            # Prefill to the servo target: playback starts at a known,
            # small buffering latency instead of a random fill level.
            ring.prefill(self._fill_target)
        self._fills = [float(self._fill_target)] * self._N

        # ---- Capture stream (explicit input device) ----------------------
        try:
            in_info = sd.query_devices(self._in_dev, "input")
            in_ch   = min(int(in_info["max_input_channels"]), 2)
        except Exception:
            in_ch = 2

        def _in_cb(indata, frames, time_info, status):
            if status:
                self.xruns += 1
            sq = indata * indata
            # In-place write: raw_in is aliased by UI meter code
            self.raw_in[:] = np.sqrt(
                [sq[:, 0].mean(), sq[:, 1 if sq.shape[1] > 1 else 0].mean()])
            block = (indata[:, :2] if indata.shape[1] >= 2
                     else np.column_stack([indata[:, 0]] * 2)).astype(np.float32)
            self._in_ring.write(block)

        self._sd_in = sd.InputStream(
            samplerate=self._fs,
            blocksize=self._bs,
            device=self._in_dev,
            channels=in_ch,
            dtype="float32",
            callback=_in_cb,
            latency="low",
        )

        # ---- Processing thread (before outputs: rings refill from the
        # moment the callbacks start draining the prefilled silence) -------
        self._running     = True
        self._proc_thread = threading.Thread(
            target=self._proc_loop, daemon=True, name="ModAudio-ProcN")
        self._proc_thread.start()

        self._sd_in.start()

        # ---- N output streams -------------------------------------------
        for i in range(self._N):
            dev  = self._speaker_devs[i]
            ring = self._out_rings[i]
            try:
                info  = sd.query_devices(dev, "output")
                out_ch = min(int(info["max_output_channels"]), 2)
            except Exception:
                out_ch = 2

            def _make_out_cb(r, idx):
                def _cb(outdata, frames, time_info, status):
                    if status:
                        self.xruns += 1
                    data = r.read_out(frames)
                    out_ch = min(outdata.shape[1], 2)
                    outdata[:, :out_ch] = data[:, :out_ch]
                    if outdata.shape[1] > out_ch:
                        outdata[:, out_ch:] = 0.0
                return _cb

            def _make_finished_cb(idx):
                def _cb():
                    if self._running:
                        self.device_error = self._names[idx]
                return _cb

            out_sd = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=dev,
                channels=out_ch,
                dtype="float32",
                callback=_make_out_cb(ring, i),
                latency="low",
                finished_callback=_make_finished_cb(i),
            )
            out_sd.start()
            self._sd_outs[i] = out_sd

    def _proc_loop(self) -> None:
        _boost_thread_priority()
        while not self._stop_event.is_set():
            block = self._in_ring.read(self._bs, timeout=_PROC_TIMEOUT)
            if block is None:
                continue
            try:
                buses = self._chain.process(block)
            except Exception as exc:
                print(f"[MultiSpeakerStreamN] DSP error: {exc}")
                for ring in self._out_rings:
                    ring.write(np.zeros((self._bs, 2), dtype=np.float32))
                continue

            for i, stereo in enumerate(buses):
                # Meter here (proc thread) rather than in the RT callback;
                # in-place write keeps the raw_out_front/rear aliases valid.
                sq = stereo * stereo
                self.raw_out[i][:] = np.sqrt(
                    [sq[:, 0].mean(), sq[:, 1].mean()])
                # BT compensation + wavefront (distance) alignment delay
                delay = self._delay_samp[i]
                if i < len(self._dist_delay_samp):
                    delay += self._dist_delay_samp[i]
                stereo = self._delay_bufs[i].process(stereo, delay)

                # Clock-drift servo: hold each device's ring fill at target
                ring = self._out_rings[i]
                fill = 0.9 * self._fills[i] + 0.1 * ring.available
                self._fills[i] = fill
                err = (fill - self._fill_target) / (self._fs * _SERVO_TAU)
                ratio = 1.0 + float(np.clip(err, -_SERVO_MAX, _SERVO_MAX))
                if ring.available == 0 and fill < self._bs:
                    ring.prefill(self._fill_target)
                    self._fills[i] = float(self._fill_target)
                stereo = self._vs[i].process(stereo, ratio)
                ring.write(stereo)

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()

        if self._proc_thread:
            self._proc_thread.join(timeout=2.0)
            self._proc_thread = None

        if self._sd_in:
            try:
                self._sd_in.stop()
                self._sd_in.close()
            except Exception:
                pass
            self._sd_in = None

        for i, out in enumerate(self._sd_outs):
            if out is not None:
                try:
                    out.stop()
                    out.close()
                except Exception:
                    pass
                self._sd_outs[i] = None

        # Reset all buffers
        self._in_ring.reset()
        for ring in self._out_rings:
            ring.reset()
        for buf in self._delay_bufs:
            buf.reset()

    @property
    def running(self) -> bool:
        return self._running

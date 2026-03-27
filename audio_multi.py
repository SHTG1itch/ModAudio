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
    """Return the sounddevice index of a Stereo Mix / loopback input, or None."""
    kw = ("stereo mix", "what u hear", "wave out mix", "loopback",
          "cable output", "vb-audio")
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

    # -- writer side (called from audio callback thread) -------------------

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        if n == 0:
            return
        with self._cond:
            if n > self._cap:
                data = data[-self._cap:]
                n = self._cap
            # Drop oldest if needed
            if self._avail + n > self._cap:
                drop = self._avail + n - self._cap
                self._rpos  = (self._rpos + drop) % self._cap
                self._avail -= drop
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

    @property
    def available(self) -> int:
        with self._cond:
            return self._avail

    def reset(self) -> None:
        with self._cond:
            self._buf[:] = 0.0
            self._wpos  = 0
            self._rpos  = 0
            self._avail = 0


# ---------------------------------------------------------------------------
# Delay buffer (used for BT compensation)
# ---------------------------------------------------------------------------

class _DelayBuffer:
    """Ring-buffer delay line for stereo float32 blocks."""

    def __init__(self, max_delay_samples: int, channels: int = 2):
        size = max_delay_samples + 4096
        self._buf  = np.zeros((size, channels), dtype=np.float32)
        self._sz   = size
        self._ptr  = 0

    def process(self, x: np.ndarray, delay: int) -> np.ndarray:
        if delay == 0:
            return x
        n = len(x)
        w = np.arange(self._ptr, self._ptr + n, dtype=np.int64) % self._sz
        self._buf[w] = x
        r = np.arange(self._ptr - delay - n, self._ptr - delay,
                      dtype=np.int64) % self._sz
        out = self._buf[r].copy()
        self._ptr = int((self._ptr + n) % self._sz)
        return out

    def reset(self):
        self._buf[:] = 0.0
        self._ptr = 0


# ---------------------------------------------------------------------------
# Multi-device stream
# ---------------------------------------------------------------------------

_RING_MS   = 200          # ring buffer depth in milliseconds
_PROC_TIMEOUT = 0.05      # proc thread wait timeout (seconds)


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
    ):
        self._in_dev    = in_dev
        self._front_dev = front_dev
        self._rear_dev  = rear_dev
        self._fs        = fs
        self._bs        = block_size
        self._mode      = mode   # set FIRST — _apply_bt_delay needs it

        self._chain = MultiSpeakerChain(fs=fs, preset=preset,
                                        bass_priority=bass_priority,
                                        rear_az_deg=rear_az_deg)

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

        # -- Acoustic (distance) delay —  delays FRONT to align wavefronts
        # when the rear speaker is physically farther from the listener.
        # Applied on top of Bluetooth compensation, but only to front stream.
        acou_samp = int(round(float(acoustic_delay_ms) * fs / 1000.0))
        if acou_samp > 0 and self._front_comp_delay == 0:
            # Only apply if BT hasn't already delayed the front
            self._front_comp_delay = acou_samp
        self._acoustic_delay_ms = float(acoustic_delay_ms)

        # -- Per-bus gains ------------------------------------------------
        self._front_gain   = float(front_gain)
        self._rear_gain    = float(rear_gain)
        self._swap_rear_lr = swap_rear_lr

        # -- Ring buffers -------------------------------------------------
        ring_frames = int(round(_RING_MS * fs / 1000.0))
        self._in_ring    = _AudioRingBuffer(ring_frames, channels=2)
        self._front_ring = _AudioRingBuffer(ring_frames, channels=2)
        self._rear_ring  = _AudioRingBuffer(ring_frames, channels=2)

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

    def update_rear_gain(self, gain: float) -> None:
        self._rear_gain = float(gain)

    def update_bass_priority(self, priority: str) -> None:
        self._chain.set_bass_priority(priority)

    def update_rear_az(self, rear_az_deg: float) -> None:
        """Update rear speaker azimuth and rebuild VBAP routing matrix."""
        self._chain.update_rear_az(rear_az_deg)

    @property
    def front_delay_ms(self) -> float:
        return self._front_comp_delay * 1000.0 / self._fs

    @property
    def rear_delay_ms(self) -> float:
        return self._rear_comp_delay * 1000.0 / self._fs

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
        self._chain = MultiSpeakerChain(
            fs=self._fs, preset=preset,
            bass_priority=old._bass_priority if old is not None else "equal",
            rear_az_deg=old._rear_az_deg     if old is not None else 150.0,
        )

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

            # BT delay compensation
            if self._front_comp_delay > 0:
                front = self._front_delay_buf.process(front, self._front_comp_delay)
            if self._rear_comp_delay > 0:
                rear = self._rear_delay_buf.process(rear, self._rear_comp_delay)

            # Apply gains
            front = front * self._front_gain
            rear  = rear  * self._rear_gain

            # Swap rear L/R if speaker faces listener
            if self._swap_rear_lr:
                rear = rear[:, ::-1].copy()

            # Meter outputs
            sq2 = front * front
            self.raw_out_front[:] = np.sqrt([sq2[:, 0].mean(), sq2[:, 1].mean()])
            sq3 = rear * rear
            self.raw_out_rear[:]  = np.sqrt([sq3[:, 0].mean(), sq3[:, 1].mean()])

            # Write to output ring buffers
            if self._mode == "dual" and self._front_ring is not None:
                self._front_ring.write(front)
            self._rear_ring.write(rear)

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
        block = self._front_ring.read_nb(frames)
        if block is not None:
            out_ch = min(outdata.shape[1], 2)
            outdata[:, :out_ch] = block[:, :out_ch]
            if outdata.shape[1] > out_ch:
                outdata[:, out_ch:] = 0.0
        else:
            outdata[:] = 0.0

    def _rear_out_cb(self, outdata: np.ndarray, frames: int,
                     time_info, status) -> None:
        """sounddevice rear OutputStream callback — drains _rear_ring."""
        if status:
            self.xruns += 1
        block = self._rear_ring.read_nb(frames)
        if block is not None:
            out_ch = min(outdata.shape[1], 2)
            outdata[:, :out_ch] = block[:, :out_ch]
            if outdata.shape[1] > out_ch:
                outdata[:, out_ch:] = 0.0
        else:
            outdata[:] = 0.0

    # ------------------------------------------------------------------ #
    # Start / Stop
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Open streams and start the processing thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._in_ring.reset()
        self._front_ring.reset()
        self._rear_ring.reset()
        self._front_delay_buf.reset()
        self._rear_delay_buf.reset()

        # Query rear device channel count
        rear_info = sd.query_devices(self._rear_dev, "output")
        rear_ch   = min(int(rear_info["max_output_channels"]), 2)
        # Use appropriate latency: Bluetooth devices need a higher buffer
        rear_latency = "high" if self.rear_is_bt else "low"

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
                        "Stereo Mix: not found on this system.\n\n"
                        "Solutions:\n"
                        "  1. pip install pyaudiowpatch   (recommended, free)\n"
                        "  2. Enable Stereo Mix in Windows Sound > Recording tab\n"
                        "  3. Switch to Full Control mode (requires VB-Cable)")

            self._rear_sd = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._rear_dev,
                channels=rear_ch,
                dtype="float32",
                callback=self._rear_out_cb,
                latency=rear_latency,
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
            )

        # ── Dual (Full Control) mode ─────────────────────────────────────
        else:
            in_info    = sd.query_devices(self._in_dev,    "input")
            front_info = sd.query_devices(self._front_dev, "output")
            in_ch    = min(int(in_info["max_input_channels"]),    2)
            front_ch = min(int(front_info["max_output_channels"]), 2)
            front_latency = "high" if self.front_is_bt else "low"

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
                latency=front_latency,
            )
            self._rear_sd = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._bs,
                device=self._rear_dev,
                channels=rear_ch,
                dtype="float32",
                callback=self._rear_out_cb,
                latency=rear_latency,
            )

        # ── Start streams ────────────────────────────────────────────────
        if self._lb_stream is not None:
            self._lb_stream.start_stream()
        if self._sd_in is not None:
            self._sd_in.start()
        if self._front_sd is not None:
            self._front_sd.start()
        self._rear_sd.start()

        # ── Start processing thread ──────────────────────────────────────
        self._proc_thread = threading.Thread(
            target=self._proc_loop, name="MultiSpeaker-DSP", daemon=True)
        self._proc_thread.start()

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

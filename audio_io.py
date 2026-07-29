"""
Real-time audio I/O using sounddevice (PortAudio backend).

How audio routing works
-----------------------
Option A - VB-Cable (recommended, zero latency overhead):
  Windows: Install VB-Audio Virtual Cable, set "CABLE Input" as default
           playback, and ModAudio reads from "CABLE Output".
  macOS:   Install VB-Audio Cable for Mac; same principle applies.

Option B - BlackHole (macOS, free alternative to VB-Cable):
  Install BlackHole (https://github.com/ExistentialAudio/BlackHole),
  set it as the system output, and select it as the ModAudio capture.

Option C - Stereo Mix / WASAPI Loopback (Windows only, no extra software):
  Enable "Stereo Mix" in Windows Sound settings (Control Panel ->
  Sound -> Recording tab -> right-click -> Show Disabled Devices).

All loopback sources are auto-detected by this module.  Run with
--list-devices to see all available devices and their indices.
"""

from __future__ import annotations
import threading
import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE, BLOCK_SIZE, CHANNELS


def _write_output(outdata: np.ndarray, result: np.ndarray) -> None:
    """Copy stereo DSP output to a mono, stereo, or multichannel device buffer."""
    if outdata.shape[1] == 1:
        outdata[:, 0] = result.mean(axis=1)
        return
    outdata[:, :2] = result[:, :2]
    if outdata.shape[1] > 2:
        outdata[:, 2:] = 0.0


# -- Device discovery ----------------------------------------------------------

def list_devices() -> None:
    """Print all audio devices with their indices."""
    devices = sd.query_devices()
    print("\n  idx  ch_in  ch_out  name")
    print("  " + "-" * 60)
    for i, d in enumerate(devices):
        marker = ""
        nl = d["name"].lower()
        if any(kw in nl for kw in ("cable output", "stereo mix",
                                    "blackhole", "soundflower")):
            marker = " <- good input (loopback)"
        elif "cable input" in nl:
            marker = " <- set as system default output"
        print(f"  {i:3d}  {d['max_input_channels']:5d}  {d['max_output_channels']:6d}  "
              f"{d['name']}{marker}")
    print()


def find_default_devices() -> tuple[int | None, int | None]:
    """
    Auto-detect the best input/output device pair.

    Input priority:
      1. WASAPI VB-Cable Output  (best quality, lowest latency on Windows)
      2. WASAPI BlackHole / Soundflower  (macOS equivalent)
      3. Any VB-Cable / BlackHole / Soundflower on any API
      4. Stereo Mix / What U Hear  (Windows fallback, no extra software)
      5. Any device with "loopback" in the name
      6. System default input

    The WASAPI variant is preferred over MME/DS because a WASAPI loopback
    source paired with a WASAPI speaker opens as a clean single duplex
    stream (no cross-API bridging required).
    """
    devices  = sd.query_devices()
    hostapis = sd.query_hostapis()
    input_id  = None
    output_id = None

    # Determine preferred host-API label for this platform
    import sys as _sys
    if _sys.platform == "win32":
        _pref_api = "WASAPI"
    elif _sys.platform == "darwin":
        _pref_api = "CoreAudio"
    else:
        _pref_api = None   # prefer any

    def _api_label(ha_idx: int) -> str:
        try:
            name = hostapis[ha_idx]["name"]
            if "WASAPI"      in name: return "WASAPI"
            if "MME"         in name: return "MME"
            if "DirectSound" in name: return "DS"
            if "Core Audio"  in name: return "CoreAudio"
            return name[:8]
        except (IndexError, KeyError):
            return "?"

    # High-priority loopback keywords
    _high = ("cable output", "blackhole", "soundflower", "vb-cable")
    _med  = ("stereo mix", "what u hear", "wave out mix")
    _low  = ("loopback",)

    # Pass 1: preferred-API high-priority loopback
    if _pref_api:
        for i, d in enumerate(devices):
            if d["max_input_channels"] < 1:
                continue
            nl = d["name"].lower()
            if any(kw in nl for kw in _high) and _api_label(d["hostapi"]) == _pref_api:
                input_id = i
                break

    # Pass 2: any-API high-priority loopback
    if input_id is None:
        for i, d in enumerate(devices):
            if d["max_input_channels"] < 1:
                continue
            nl = d["name"].lower()
            if any(kw in nl for kw in _high):
                input_id = i
                break

    # Pass 3: medium-priority (Stereo Mix etc.)
    if input_id is None:
        for i, d in enumerate(devices):
            if d["max_input_channels"] < 1:
                continue
            nl = d["name"].lower()
            if any(kw in nl for kw in _med):
                input_id = i
                break

    # Pass 4: low-priority (any "loopback" device)
    if input_id is None:
        for i, d in enumerate(devices):
            if d["max_input_channels"] < 1:
                continue
            if any(kw in d["name"].lower() for kw in _low):
                input_id = i
                break

    # Pass 5: system default
    if input_id is None:
        try:
            input_id = sd.default.device[0]
        except Exception:
            input_id = None

    # Output: system default (usually the preferred API speaker on modern OSes)
    try:
        output_id = sd.default.device[1]
    except Exception:
        output_id = None

    return input_id, output_id


# -- Stream wrapper ------------------------------------------------------------

class AudioStream:
    """
    Wraps sounddevice's duplex stream for real-time block processing.

    The provided `processor` callable receives a (N, 2) float32 block
    and must return a (N, 2) float32 block.
    """

    def __init__(
        self,
        processor,
        input_device:  int | None = None,
        output_device: int | None = None,
        fs:            int = SAMPLE_RATE,
        block_size:    int = BLOCK_SIZE,
        channels:      int = CHANNELS,
    ):
        self._processor    = processor
        self._input_device  = input_device
        self._output_device = output_device
        self._fs            = fs
        self._block_size    = block_size
        self._channels      = channels
        self._stream        = None
        self._lock          = threading.Lock()

        # Statistics
        self._xruns = 0
        self._blocks_processed = 0

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            self._xruns += 1

        with self._lock:
            try:
                # Ensure float32, shape (N, 2)
                block = np.ascontiguousarray(indata[:, :self._channels], dtype=np.float32)
                if block.shape[1] == 1:
                    block = np.concatenate([block, block], axis=1)

                result = self._processor(block)

                _write_output(outdata, result)

                self._blocks_processed += 1
            except Exception as exc:
                # Never raise inside sounddevice callback - just mute
                outdata[:] = 0.0
                print(f"\n[audio] callback error: {exc}")

    def start(self) -> None:
        """Open and start the audio stream."""
        in_dev  = self._input_device
        out_dev = self._output_device

        # Determine channel counts for devices
        try:
            in_info  = sd.query_devices(in_dev,  "input")
            in_ch    = min(in_info["max_input_channels"], 2)
        except Exception:
            in_ch = 2

        try:
            out_info = sd.query_devices(out_dev, "output")
            out_ch   = min(out_info["max_output_channels"], 2)
        except Exception:
            out_ch = 2

        self._stream = sd.Stream(
            samplerate=self._fs,
            blocksize=self._block_size,
            device=(in_dev, out_dev),
            channels=(in_ch, out_ch),
            dtype="float32",
            callback=self._callback,
            latency="low",
        )
        self._stream.start()

        in_name  = sd.query_devices(in_dev)["name"]  if in_dev  is not None else "default"
        out_name = sd.query_devices(out_dev)["name"] if out_dev is not None else "default"
        print(f"  Input  : [{in_dev}] {in_name}")
        print(f"  Output : [{out_dev}] {out_name}")
        print(f"  Rate   : {self._fs} Hz,  Block: {self._block_size} samples "
              f"({1000 * self._block_size / self._fs:.1f} ms)")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def xruns(self) -> int:
        return self._xruns

    @property
    def blocks_processed(self) -> int:
        return self._blocks_processed

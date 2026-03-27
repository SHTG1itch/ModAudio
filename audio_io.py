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

    Priority for input:
      1. VB-Cable Output   (Windows: "CABLE Output")
      2. BlackHole         (macOS virtual loopback)
      3. Soundflower       (macOS, legacy)
      4. VB-Cable          (macOS variant — no "output" suffix)
      5. Stereo Mix / What U Hear  (Windows)
      6. Any device with "loopback" in the name
      7. System default input
    """
    devices = sd.query_devices()
    input_id  = None
    output_id = None

    # High-priority loopback sources (first match wins)
    _high = ("cable output", "blackhole", "soundflower", "vb-cable")
    # Medium-priority (Windows system capture)
    _med  = ("stereo mix", "what u hear", "wave out mix")
    # Low-priority fallback
    _low  = ("loopback",)

    for i, d in enumerate(devices):
        if d["max_input_channels"] < 1:
            continue
        nl = d["name"].lower()
        if any(kw in nl for kw in _high):
            input_id = i
            break

    if input_id is None:
        for i, d in enumerate(devices):
            if d["max_input_channels"] < 1:
                continue
            nl = d["name"].lower()
            if any(kw in nl for kw in _med) and input_id is None:
                input_id = i
            if any(kw in nl for kw in _low) and input_id is None:
                input_id = i

    # Default input if nothing found
    if input_id is None:
        try:
            input_id = sd.default.device[0]
        except Exception:
            input_id = None

    # Default output
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

                # Write output
                out_ch = min(outdata.shape[1], 2)
                outdata[:, :out_ch] = result[:, :out_ch]
                if outdata.shape[1] > out_ch:
                    outdata[:, out_ch:] = 0.0

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
            out_ch   = min(out_info["max_output_channels"], 8)
            out_ch   = max(out_ch, 2)
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

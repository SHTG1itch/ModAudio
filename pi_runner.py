#!/usr/bin/env python3
"""
ModAudio — Raspberry Pi / TV add-on.
====================================

Standalone headless runner for Raspberry Pi (or any Linux box driving a TV or
AV setup). Loads speaker layout from a YAML/JSON config file, instantiates the
multi-speaker theater chain, and streams processed audio to each physical
output device.

This module is intentionally isolated from the GUI (no customtkinter import)
so it runs on Pi OS Lite without a display server. Import graph:

    pi_runner.py
       ├── config.py
       ├── audio_multi.py   (MultiSpeakerStreamN)
       └── dsp/             (MultiSpeakerChainN)

Quick start
-----------

1.  Write a config file, e.g. ``pi_config.yaml``:

        # ModAudio Pi speaker layout
        sample_rate: 48000
        block_size: 512
        preset: Cinema          # Cinema | IMAX | Dolby | Home
        input_device: 0         # sounddevice index; use `--list-devices`
        bass_priority: equal    # equal | front | rear

        speakers:
          - name: Front-Left
            device: 2
            az:  -30
            el:   0
            face_az: 150
          - name: Front-Right
            device: 3
            az:   30
            face_az: 210
          - name: Rear-Left
            device: 4
            az:  -110
            face_az:  70
          - name: Rear-Right
            device: 5
            az:   110
            face_az: 290

    YAML is preferred (``pip install pyyaml``) but JSON works too.

2.  Test with ``python pi_runner.py --list-devices`` to discover indices.

3.  Run:  ``python pi_runner.py -c pi_config.yaml``

4.  (Optional) install as a systemd service — see the template at the bottom
    of this file, or ``pi_runner.py --print-systemd``.

Audio routing on the Pi
-----------------------
The Pi captures audio from the TV either via (a) an HDMI-ARC capture device,
(b) a USB audio input, (c) a Bluetooth A2DP sink, or (d) PulseAudio monitor.
Output goes to N physical speakers — HDMI, 3.5 mm jack, USB DACs, etc.

We keep the existing Windows/Mac code path untouched; this file is additive.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# Ensure package imports resolve when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sounddevice as sd

from config import HEADPHONES_PRESET, SPEAKERS_PRESET, SAMPLE_RATE, BLOCK_SIZE
from audio_multi import MultiSpeakerStreamN


# Built-in presets (mirrors app.PRESETS without importing the GUI module).
PRESETS: dict[str, dict] = {
    "Cinema": {
        "rt60": 1.3,  "rt60_hf": 0.65,
        "reverb_predelay_ms": 22.0,  "reverb_mix": 0.25,  "early_ref_mix": 0.45,
        "stereo_width": 2.0,  "surround_level": 0.78,  "lfe_level": 0.88,
        "center_level": 0.88,  "rear_level": 0.65,
        "bass_boost_db": 6.0,  "sub_bass_db": 4.5,
        "bass_harm_drive": 2.8,  "bass_harm_level": 0.50,
        "air_exciter_level": 0.18,
        "mb_compress_drive": 1.6,  "transient_amount": 0.55,
        "output_gain_db": 4.5,
    },
    "IMAX": {
        "rt60": 1.9,  "rt60_hf": 0.80,
        "reverb_predelay_ms": 32.0,  "reverb_mix": 0.36,  "early_ref_mix": 0.60,
        "stereo_width": 2.5,  "surround_level": 0.90,  "lfe_level": 1.00,
        "center_level": 0.85,  "rear_level": 0.78,
        "bass_boost_db": 10.0,  "sub_bass_db": 7.0,
        "bass_harm_drive": 3.8,  "bass_harm_level": 0.72,
        "air_exciter_level": 0.24,
        "mb_compress_drive": 2.0,  "transient_amount": 0.85,
        "output_gain_db": 3.0,
    },
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
        "atmos_mode": True,  "height_level": 0.45,
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


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """Load YAML or JSON config. YAML requires `pip install pyyaml`."""
    text = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            sys.exit("PyYAML not installed. Run:  pip install pyyaml  "
                     "(or use a JSON config)")
        return yaml.safe_load(text)
    if ext == ".json":
        return json.loads(text)
    # Try YAML first, fall back to JSON
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def validate_config(cfg: dict) -> dict:
    """Validate and fill defaults. Raises SystemExit on bad input."""
    if not isinstance(cfg, dict):
        sys.exit("Config root must be a mapping/object.")
    speakers = cfg.get("speakers")
    if not speakers or not isinstance(speakers, list):
        sys.exit("Config must contain a non-empty `speakers` list.")
    if len(speakers) < 2:
        sys.exit(f"Need at least 2 speakers, got {len(speakers)}.")
    for i, spk in enumerate(speakers):
        if not isinstance(spk, dict):
            sys.exit(f"speakers[{i}] must be a mapping.")
        if "device" not in spk or "az" not in spk:
            sys.exit(f"speakers[{i}] must have `device` and `az`.")
    cfg.setdefault("sample_rate",   SAMPLE_RATE)
    cfg.setdefault("block_size",    BLOCK_SIZE)
    cfg.setdefault("preset",        "Cinema")
    cfg.setdefault("bass_priority", "equal")
    cfg.setdefault("bt_delay_ms",   0.0)
    if cfg["preset"] not in PRESETS:
        sys.exit(f"Unknown preset '{cfg['preset']}'. "
                 f"Choose from: {', '.join(PRESETS)}")
    if "input_device" not in cfg:
        sys.exit("Config must set `input_device` (use --list-devices to find it).")
    return cfg


def build_preset(name: str, mode: str = "speakers") -> dict:
    """Merge a named preset onto SPEAKERS_PRESET (theater mode on Pi)."""
    base = dict(SPEAKERS_PRESET if mode == "speakers" else HEADPHONES_PRESET)
    base.update(PRESETS[name])
    base["mode"] = mode
    return base


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def run(cfg: dict) -> int:
    """Spin up the multi-speaker stream and block until Ctrl+C / SIGTERM."""
    preset = build_preset(cfg["preset"])
    speakers = cfg["speakers"]

    speaker_devs  = [int(s["device"])        for s in speakers]
    speaker_az    = [float(s["az"])          for s in speakers]
    speaker_el    = [float(s.get("el", 0.0)) for s in speakers]
    speaker_faces = [s.get("face_az")        for s in speakers]
    speaker_face_azs = [
        float(fa) if fa is not None else (az + 180.0) % 360.0
        for az, fa in zip(speaker_az, speaker_faces)
    ]
    speaker_face_els = [float(s.get("face_el", 0.0)) for s in speakers]
    gains            = [float(s.get("gain", 1.0))    for s in speakers]

    names = [s.get("name", f"spk{i}") for i, s in enumerate(speakers)]

    print("ModAudio Pi runner")
    print(f"  preset     : {cfg['preset']}")
    print(f"  sample rate: {cfg['sample_rate']} Hz   block: {cfg['block_size']}")
    print(f"  input dev  : {cfg['input_device']}")
    print(f"  speakers   :")
    for nm, dev, az, el in zip(names, speaker_devs, speaker_az, speaker_el):
        print(f"    - {nm:16s} dev={dev:<3d} az={az:+6.1f} deg  el={el:+5.1f} deg")

    stream = MultiSpeakerStreamN(
        in_dev=int(cfg["input_device"]),
        speaker_devs=speaker_devs,
        speaker_azimuths=speaker_az,
        speaker_elevations=speaker_el,
        speaker_face_azs=speaker_face_azs,
        speaker_face_els=speaker_face_els,
        fs=int(cfg["sample_rate"]),
        block_size=int(cfg["block_size"]),
        preset=preset,
        bt_delay_ms=float(cfg.get("bt_delay_ms", 0.0)),
        gains=gains,
        bass_priority=cfg["bass_priority"],
    )

    stop = [False]
    def _stop(*_): stop[0] = True
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        stream.start()
    except Exception as exc:
        print(f"\n  ERROR: Could not start stream: {exc}", file=sys.stderr)
        return 1

    print("\n  Running. Ctrl+C to stop.")
    t0 = time.time()
    try:
        while not stop[0]:
            time.sleep(5.0)
            elapsed = time.time() - t0
            print(f"\r  {elapsed:7.0f} s  |  xruns: {getattr(stream, 'xruns', 0)}",
                  end="", flush=True)
    finally:
        print("\n  Stopping...")
        try:
            stream.stop()
        except Exception:
            pass
        print("  Done.")
    return 0


# ---------------------------------------------------------------------------
# Systemd unit template (for `pi_runner.py --print-systemd`)
# ---------------------------------------------------------------------------

SYSTEMD_UNIT = """[Unit]
Description=ModAudio Pi — cinema surround for TV
After=sound.target network.target

[Service]
Type=simple
User=pi
WorkingDirectory={workdir}
ExecStart={python} {script} -c {config}
Restart=on-failure
RestartSec=5
Nice=-10

[Install]
WantedBy=multi-user.target
"""


def _print_systemd(config_path: str | None):
    cfg = os.path.abspath(config_path) if config_path else "/home/pi/modaudio/pi_config.yaml"
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.abspath(__file__)
    print(SYSTEMD_UNIT.format(
        workdir=here, python=sys.executable, script=script, config=cfg))
    print("# Install:")
    print("#   sudo tee /etc/systemd/system/modaudio.service < modaudio.service")
    print("#   sudo systemctl daemon-reload")
    print("#   sudo systemctl enable --now modaudio.service")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ModAudio Pi runner — headless multi-speaker theater audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--config", type=Path,
                   help="Path to pi_config.yaml (or .json)")
    p.add_argument("--list-devices", action="store_true",
                   help="Print all audio devices and exit")
    p.add_argument("--print-systemd", action="store_true",
                   help="Print a systemd unit template and exit")
    p.add_argument("--check", action="store_true",
                   help="Validate config and exit without streaming")
    return p.parse_args()


def list_devices():
    print(f"{'idx':>4}  {'in':>3}  {'out':>3}  name")
    print("-" * 62)
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:>4}  {d['max_input_channels']:>3}  {d['max_output_channels']:>3}  "
              f"{d['name']}")


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return 0

    if args.print_systemd:
        _print_systemd(str(args.config) if args.config else None)
        return 0

    if not args.config:
        print("usage: pi_runner.py -c pi_config.yaml", file=sys.stderr)
        print("       pi_runner.py --list-devices", file=sys.stderr)
        return 2

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    cfg = validate_config(load_config(args.config))
    if args.check:
        print("Config OK.")
        return 0
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())

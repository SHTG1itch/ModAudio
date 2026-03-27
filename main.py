#!/usr/bin/env python3
"""
ModAudio - Theater Experience
==============================
Recreates the cinema soundstage in real-time using psychoacoustic processing.
Works with headphones (full binaural HRTF) and speakers (stereo widening).

Quick start
-----------
  1. Install dependencies:  pip install -r requirements.txt

  2. Route audio:
     * VB-Cable (recommended)  https://vb-audio.com/Cable/
       Set "CABLE Input" as Windows default playback, run:
         python main.py -i <CABLE Output idx> -o <speakers idx>
     * Stereo Mix (no extra software)
       Enable in Windows Sound -> Recording -> Show Disabled Devices
       Run: python main.py   (auto-detects Stereo Mix)

  3. Run:
       python main.py                      # auto-detect, headphones mode
       python main.py --mode speakers      # speaker mode (no HRTF)
       python main.py --list-devices       # show all device indices
       python main.py -i 22 -o 15         # specific devices

Tweakable parameters (all optional)
-------------------------------------
  --mode       headphones | speakers   (default: headphones)
  --rt60       room decay in seconds   (default: 1.3)
  --reverb-mix reverb wet level 0-1    (default: 0.25)
  --width      stereo width mult.      (default: 2.0)
  --gain       output gain in dB       (default: -1.5)
  --drive      dynamics drive 1.0-2.0  (default: 1.6)
"""

import argparse
import signal
import sys
import time

import numpy as np

from config   import HEADPHONES_PRESET, SPEAKERS_PRESET, SAMPLE_RATE, BLOCK_SIZE
from audio_io import AudioStream, list_devices, find_default_devices
from dsp      import TheaterChain


# -- Banner --------------------------------------------------------------------

BANNER = r"""
  __  __           _    _             _ _
 |  \/  |         | |  / \  _   _  __| (_) ___
 | |\/| | ___   __| | / _ \| | | |/ _` | |/ _ \
 | |  | |/ _ \ / _  |/ ___ \ |_| | (_| | | (_) |
 |_|  |_|\___/ \__,_/_/   \_\__,_|\__,_|_|\___/

  Theater Experience - by ModAudio
"""


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ModAudio: real-time theater audio processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input",  type=int, default=None, metavar="DEV",
                   help="Input device index")
    p.add_argument("-o", "--output", type=int, default=None, metavar="DEV",
                   help="Output device index")
    p.add_argument("--list-devices", action="store_true",
                   help="Print all audio devices and exit")
    p.add_argument("--mode", choices=["headphones", "speakers"],
                   default="headphones",
                   help="headphones=binaural HRTF, speakers=stereo widening")
    p.add_argument("--fs",         type=int,   default=SAMPLE_RATE)
    p.add_argument("--block-size", type=int,   default=BLOCK_SIZE)
    p.add_argument("--rt60",       type=float, default=None)
    p.add_argument("--reverb-mix", type=float, default=None)
    p.add_argument("--width",      type=float, default=None)
    p.add_argument("--gain",       type=float, default=None)
    p.add_argument("--drive",      type=float, default=None,
                   help="Dynamics drive 1.0-2.0 (default: 1.6)")
    return p.parse_args()


def build_preset(args) -> dict:
    base = HEADPHONES_PRESET if args.mode == "headphones" else SPEAKERS_PRESET
    preset = dict(base)
    preset["mode"] = args.mode
    if args.rt60        is not None: preset["rt60"]              = args.rt60
    if args.reverb_mix  is not None: preset["reverb_mix"]        = args.reverb_mix
    if args.width       is not None: preset["stereo_width"]      = args.width
    if args.gain        is not None: preset["output_gain_db"]    = args.gain
    if args.drive       is not None: preset["mb_compress_drive"] = args.drive
    return preset


# -- Main ----------------------------------------------------------------------

def main():
    args   = parse_args()

    if args.list_devices:
        list_devices()
        return

    print(BANNER)

    preset = build_preset(args)
    fs     = args.fs

    print("  Initialising theater DSP chain  [mode: %s] ..." % args.mode)
    chain = TheaterChain(fs=fs, preset=preset)

    auto_in, auto_out = find_default_devices()
    in_dev  = args.input  if args.input  is not None else auto_in
    out_dev = args.output if args.output is not None else auto_out

    if in_dev is None:
        print("\n  ERROR: No suitable input device found.")
        if sys.platform == "win32":
            print("  Install VB-Cable or enable Stereo Mix, then re-run.")
        elif sys.platform == "darwin":
            print("  Install VB-Cable for Mac or BlackHole, then re-run.")
        else:
            print("  Install a virtual loopback device (e.g. VB-Cable), then re-run.")
        print("  Use --list-devices to see all available devices.\n")
        sys.exit(1)

    print("\n  Audio devices:")
    stream = AudioStream(
        processor=chain.process,
        input_device=in_dev,
        output_device=out_dev,
        fs=fs,
        block_size=args.block_size,
    )

    stop_event = [False]
    signal.signal(signal.SIGINT,  lambda s, f: stop_event.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda s, f: stop_event.__setitem__(0, True))

    print("\n  Preset summary:")
    print("    Mode           : %s" % args.mode)
    print("    RT60           : %.2f s" % preset["rt60"])
    print("    Pre-delay      : %.0f ms" % preset.get("reverb_predelay_ms", 22))
    print("    Reverb mix     : %.0f%%" % (preset["reverb_mix"] * 100))
    print("    Stereo width   : %.1fx" % preset["stereo_width"])
    print("    Bass boost     : +%.0f dB @ %d Hz" % (preset["bass_boost_db"], preset["bass_boost_hz"]))
    print("    Harmonic bass  : drive=%.1f  level=%.0f%%" % (
        preset["bass_harm_drive"], preset["bass_harm_level"] * 100))
    print("    Dynamics drive : %.1f" % preset["mb_compress_drive"])
    print("    Output gain    : %+.1f dB" % preset["output_gain_db"])
    print()

    try:
        stream.start()
    except Exception as exc:
        print("\n  ERROR: Could not open audio stream: %s" % exc)
        print("  Use --list-devices to pick different devices.\n")
        sys.exit(1)

    print("\n  Running  (Ctrl+C to stop)\n")
    t0 = time.time()
    try:
        while not stop_event[0]:
            time.sleep(2.0)
            elapsed = time.time() - t0
            bps = stream.blocks_processed / elapsed if elapsed > 0 else 0
            print("\r  %6.0f s  |  %.0f blk/s  |  xruns: %d" % (
                elapsed, bps, stream.xruns), end="", flush=True)
    finally:
        print("\n\n  Stopping...")
        stream.stop()
        print("  Done.")


if __name__ == "__main__":
    main()

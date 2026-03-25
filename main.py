#!/usr/bin/env python3
"""
ModAudio - Theater Experience
==============================
Recreates the cinema soundstage on stereo headphones in real-time.

Quick start
-----------
  1. Install dependencies:  pip install -r requirements.txt

  2. Route audio through one of:
     * VB-Cable (recommended - free from https://vb-audio.com/Cable/)
       Set "CABLE Input" as your Windows default playback device.
     * Stereo Mix  (enable in Windows Sound -> Recording settings)

  3. Run:
       python main.py                  # auto-detect devices
       python main.py --list-devices   # show device list
       python main.py -i 3 -o 5       # pick specific devices
"""

import argparse
import signal
import sys
import time

import numpy as np

from config    import SAMPLE_RATE, BLOCK_SIZE, THEATER_PRESET
from audio_io  import AudioStream, list_devices, find_default_devices
from dsp       import TheaterChain


# -- Banner ---------------------------------------------------------------------

BANNER = r"""
  __  __           _    _             _ _
 |  \/  |         | |  / \  _   _  __| (_) ___
 | |\/| | ___   __| | / _ \| | | |/ _` | |/ _ \
 | |  | |/ _ \ / _  |/ ___ \ |_| | (_| | | (_) |
 |_|  |_|\___/ \__,_/_/   \_\__,_|\__,_|_|\___/

  Theater Experience - by ModAudio
"""


# -- Entry point ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ModAudio: real-time theater audio processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input",  type=int, default=None,
                   metavar="DEV", help="Input device index (override auto-detect)")
    p.add_argument("-o", "--output", type=int, default=None,
                   metavar="DEV", help="Output device index (override auto-detect)")
    p.add_argument("--list-devices", action="store_true",
                   help="Print available audio devices and exit")
    p.add_argument("--fs",         type=int,   default=SAMPLE_RATE,
                   help=f"Sample rate (default: {SAMPLE_RATE})")
    p.add_argument("--block-size", type=int,   default=BLOCK_SIZE,
                   help=f"Block size in samples (default: {BLOCK_SIZE})")
    p.add_argument("--rt60",       type=float, default=None,
                   help="Override reverb RT60 in seconds (default: 1.2)")
    p.add_argument("--reverb-mix", type=float, default=None,
                   help="Override reverb wet mix 0-1 (default: 0.22)")
    p.add_argument("--width",      type=float, default=None,
                   help="Override stereo width multiplier (default: 1.9)")
    p.add_argument("--gain",       type=float, default=None,
                   help="Override output gain in dB (default: -2.0)")
    return p.parse_args()


def build_preset(args) -> dict:
    preset = dict(THEATER_PRESET)
    if args.rt60        is not None: preset["rt60"]           = args.rt60
    if args.reverb_mix  is not None: preset["reverb_mix"]     = args.reverb_mix
    if args.width       is not None: preset["stereo_width"]   = args.width
    if args.gain        is not None: preset["output_gain_db"] = args.gain
    return preset


def main():
    args   = parse_args()

    if args.list_devices:
        list_devices()
        return

    print(BANNER)

    preset = build_preset(args)
    fs     = args.fs

    # Build DSP chain
    print("  Initialising theater DSP chain...")
    chain  = TheaterChain(fs=fs, preset=preset)

    # Auto-detect or use specified devices
    auto_in, auto_out = find_default_devices()
    in_dev  = args.input  if args.input  is not None else auto_in
    out_dev = args.output if args.output is not None else auto_out

    if in_dev is None:
        print("\n  ERROR: No suitable input device found.")
        print("  Please install VB-Cable or enable Stereo Mix, then re-run.")
        print("  Use --list-devices to see all devices.\n")
        sys.exit(1)

    print("\n  Audio devices:")
    stream = AudioStream(
        processor=chain.process,
        input_device=in_dev,
        output_device=out_dev,
        fs=fs,
        block_size=args.block_size,
    )

    # Graceful shutdown on Ctrl+C
    stop_event = [False]

    def _shutdown(sig, frame):
        stop_event[0] = True

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("\n  Starting theater audio... (Ctrl+C to stop)\n")
    print(f"  Preset:")
    print(f"    RT60              : {preset['rt60']:.2f} s")
    print(f"    Reverb mix        : {preset['reverb_mix']:.0%}")
    print(f"    Stereo width      : {preset['stereo_width']:.1f}x")
    print(f"    Bass boost        : +{preset['bass_boost_db']:.0f} dB @ {preset['bass_boost_hz']} Hz")
    print(f"    Output gain       : {preset['output_gain_db']:+.1f} dB")
    print()

    try:
        stream.start()
    except Exception as exc:
        print(f"\n  ERROR: Could not open audio stream: {exc}")
        print("  Check that your input/output devices support the selected sample rate.")
        print("  Try --list-devices to pick different devices.\n")
        sys.exit(1)

    # Status loop
    t0 = time.time()
    try:
        while not stop_event[0]:
            time.sleep(2.0)
            elapsed = time.time() - t0
            bps     = stream.blocks_processed / elapsed if elapsed > 0 else 0
            print(f"\r  Running {elapsed:6.0f}s  |  {bps:6.1f} blocks/s  |"
                  f"  xruns: {stream.xruns}", end="", flush=True)
    finally:
        print("\n\n  Stopping...")
        stream.stop()
        print("  Done.")


if __name__ == "__main__":
    main()

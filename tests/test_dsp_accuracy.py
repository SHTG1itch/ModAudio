"""Verification tests for the DSP accuracy fixes (LR4 crossovers, VBAP
bracketing, linked limiter, distance compensation, exciter anti-aliasing).

Run directly:  python tests/test_dsp_accuracy.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math

ok = True
def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {detail}")


# ---- 1. LR4 crossover sums flat -------------------------------------------
from dsp.filters import make_lr4_lowpass, make_lr4_highpass

fs = 48000
n  = fs * 2
rng = np.random.default_rng(0)
noise = rng.standard_normal((n, 2)).astype(np.float64) * 0.1

lp = make_lr4_lowpass(120.0, fs=fs, ch=2)
hp = make_lr4_highpass(120.0, fs=fs, ch=2)
recomb = lp.process(noise.copy()) + hp.process(noise.copy())

# Spectral magnitude near 120 Hz vs input
win = np.hanning(n)
f   = np.fft.rfftfreq(n, 1/fs)
H_in  = np.abs(np.fft.rfft(noise[:, 0] * win))
H_out = np.abs(np.fft.rfft(recomb[:, 0] * win))
band = (f > 90) & (f < 160)
ratio_db = 20*np.log10(np.mean(H_out[band]) / np.mean(H_in[band]))
check("LR4 sub/mid recombines flat at 120 Hz", abs(ratio_db) < 1.0, f"({ratio_db:+.2f} dB)")

# Old-style butterworth pair for comparison
from dsp.filters import make_lowpass, make_highpass
lp2 = make_lowpass(120.0, q=0.707, fs=fs, ch=2)
hp2 = make_highpass(120.0, q=0.707, fs=fs, ch=2)
old = lp2.process(noise.copy()) + hp2.process(noise.copy())
H_old = np.abs(np.fft.rfft(old[:, 0] * win))
old_db = 20*np.log10(np.mean(H_old[band]) / np.mean(H_in[band]))
print(f"       (old Butterworth pair for reference: {old_db:+.2f} dB at crossover)")

# ---- 2. VBAP bracketing ----------------------------------------------------
from dsp.multi_speaker import _vbap_sphere, _vbap_circle

# Speakers at 0, 20, 180 deg; source at 40 deg must use the (20, 180) pair,
# not the same-side (0, 20) pair.
g = _vbap_sphere(40.0, 0.0, [(0.0, 0.0), (20.0, 0.0), (180.0, 0.0)])
check("VBAP pair brackets source", g[0] < 1e-6 and g[1] > 0 and g[2] > 0,
      f"(gains={np.round(g, 3)})")
check("VBAP constant power", abs(float(np.sum(g**2)) - 1.0) < 1e-3)

# Source exactly at a speaker
g2 = _vbap_sphere(20.0, 0.0, [(0.0, 0.0), (20.0, 0.0), (180.0, 0.0)])
check("VBAP source at speaker", g2[1] > 0.99, f"(gains={np.round(g2, 3)})")

# Elevated source routes to elevated speaker
g3 = _vbap_sphere(0.0, 45.0, [(0.0, 0.0), (0.0, 60.0), (180.0, 0.0)])
check("VBAP height routing", g3[1] > g3[0] and g3[1] > g3[2], f"(gains={np.round(g3, 3)})")

# ---- 3. Vectorised limiter matches reference loop --------------------------
from dsp.dynamics import PeakLimiter

x = (rng.standard_normal((2048, 2)) * 0.8).astype(np.float32)
x[500:520] *= 4.0   # transient burst

lim = PeakLimiter(threshold=0.93, release_ms=80.0, fs=fs)
out_vec = np.concatenate([lim.process(x[:1000]), lim.process(x[1000:])])

# Reference: original per-sample loop
thr, rel, env = 0.93, math.exp(-1.0 / (80.0 * 1e-3 * fs)), 0.0
x64 = x.astype(np.float64)
ref = np.empty_like(x64)
peak = np.abs(x64).max(axis=1)
for i in range(len(peak)):
    env = max(peak[i], env * rel)
    ref[i] = x64[i] * thr / max(env, thr)
err = np.max(np.abs(out_vec.astype(np.float64) - ref))
check("Vectorised limiter == reference loop", err < 1e-6, f"(max err {err:.2e})")
check("Limiter ceiling respected", float(np.abs(out_vec).max()) <= 0.9301)

# Linked limiting: relative bus balance preserved during limiting
lim2 = PeakLimiter(threshold=0.93, release_ms=80.0, fs=fs)
a = np.ones((256, 2), dtype=np.float32) * 0.5
b = np.ones((256, 2), dtype=np.float32) * 2.0   # this bus clips
la, lb = lim2.process_linked([a, b])
ratio = la / np.maximum(lb, 1e-9)
check("Linked limiter preserves balance", np.allclose(ratio, 0.25, atol=1e-6),
      f"(ratio {ratio[0,0]:.4f}, expected 0.2500)")

# ---- 4. Chains run end-to-end ----------------------------------------------
from dsp.multi_speaker import MultiSpeakerChain, MultiSpeakerChainN
from dsp.theater_chain import TheaterChain
from config import HEADPHONES_PRESET, SPEAKERS_PRESET

blk = (rng.standard_normal((512, 2)) * 0.25).astype(np.float32)

chain2 = MultiSpeakerChain(fs=fs, preset=dict(HEADPHONES_PRESET))
for _ in range(20):
    fr, re_ = chain2.process(blk)
check("MultiSpeakerChain runs", fr.shape == (512, 2) and re_.shape == (512, 2)
      and np.all(np.isfinite(fr)) and np.all(np.isfinite(re_)))

chainN = MultiSpeakerChainN(fs=fs, preset=dict(HEADPHONES_PRESET),
                            speaker_azimuths=[-30.0, 30.0, -110.0, 110.0, 180.0])
for _ in range(20):
    outs = chainN.process(blk)
check("MultiSpeakerChainN runs (5 spk)", len(outs) == 5
      and all(o.shape == (512, 2) and np.all(np.isfinite(o)) for o in outs))

chainN.update_speakers([-30.0, 30.0, 150.0])
outs = chainN.process(blk)
check("ChainN speaker-count hot change", len(outs) == 3)

p = dict(HEADPHONES_PRESET); p["atmos_mode"] = True; p["height_level"] = 0.5
chainH = MultiSpeakerChainN(fs=fs, preset=p,
                            speaker_azimuths=[-30.0, 30.0, 180.0],
                            speaker_elevations=[0.0, 0.0, 45.0])
outs = chainH.process(blk)
check("ChainN atmos heights run", len(outs) == 3 and all(np.all(np.isfinite(o)) for o in outs))

tc = TheaterChain(fs=fs, preset=dict(HEADPHONES_PRESET))
y = tc.process(blk)
check("TheaterChain (headphones) runs", y.shape == (512, 2) and np.all(np.isfinite(y)))

tc2 = TheaterChain(fs=fs, preset=dict(SPEAKERS_PRESET))
y2 = tc2.process(blk)
check("TheaterChain (speakers) runs", y2.shape == (512, 2) and np.all(np.isfinite(y2)))

from dsp.surround_engine import VirtualSurroundBinaural, VirtualSurroundMono
vb = VirtualSurroundBinaural(fs=fs, preset=dict(HEADPHONES_PRESET))
yb = vb.process(blk)
check("VirtualSurroundBinaural runs", yb.shape == (512, 2) and np.all(np.isfinite(yb)))
vm = VirtualSurroundMono(fs=fs, preset=dict(HEADPHONES_PRESET))
ym = vm.process(blk)
check("VirtualSurroundMono runs", ym.shape == (512, 2) and np.all(np.isfinite(ym)))

# ---- 5. update_chain preserves speaker placement ----------------------------
class _FakeStream:
    pass
from audio_multi import MultiDeviceStream

fi = (10.0, 0.0, 190.0, 0.0)
ri = (160.0, 5.0, 340.0, 0.0)
chain = MultiSpeakerChain(fs=fs, preset=dict(HEADPHONES_PRESET),
                          front_info=fi, rear_info=ri)
fake = _FakeStream()
fake._chain = chain
fake._fs = fs
fake._front_gain = 1.0
fake._rear_gain  = 1.0
out = MultiDeviceStream.update_chain(fake, dict(SPEAKERS_PRESET))
check("update_chain preserves placement",
      fake._chain._front_info == fi and fake._chain._rear_info == ri,
      f"(front={fake._chain._front_info}, rear={fake._chain._rear_info})")

# ---- 6. Distance compensation math ------------------------------------------
from audio_multi import MultiSpeakerStreamN

class _FakeN:
    _N = 3
    _fs = fs
    _max_delay = 24000
    _gains = [1.0, 1.0, 1.0]
    _push_gains = MultiSpeakerStreamN._push_gains
    class _chain:                       # noqa: N801 — stub
        @staticmethod
        def set_output_gains(gains):
            pass
fakeN = _FakeN()
MultiSpeakerStreamN._apply_distances(fakeN, [2.0, 3.0, 4.0])
d = fakeN._dist_delay_samp
g = fakeN._dist_gain
exp0 = round((4.0 - 2.0) / 343.0 * fs)
check("Distance delays (nearest gets most)", d[0] == exp0 and d[2] == 0 and d[0] > d[1] > d[2],
      f"(delays={d} samples, expected first={exp0})")
check("Distance gains (nearest attenuated)", abs(g[0] - 0.5) < 1e-9 and g[2] == 1.0,
      f"(gains={np.round(g, 3)})")

# ---- 7. Exciter aliasing reduced --------------------------------------------
from dsp.enhancer import AirBandExciter

t = np.arange(fs) / fs
tone = (0.5 * np.sin(2 * np.pi * 15000 * t)).astype(np.float64)   # 15 kHz: 2nd harm = 30k -> alias 18k
sig = np.stack([tone, tone], axis=1)
exc = AirBandExciter(cutoff=8000.0, level=0.18, fs=fs)
out = exc.process(sig)
S = np.abs(np.fft.rfft(out[:, 0] * np.hanning(fs)))
fbins = np.fft.rfftfreq(fs, 1/fs)
alias_bin = np.argmin(np.abs(fbins - 18000))
fund_bin  = np.argmin(np.abs(fbins - 15000))
alias_db = 20*np.log10(S[alias_bin] / S[fund_bin] + 1e-12)
check("Exciter alias suppressed", alias_db < -60.0, f"(18 kHz alias at {alias_db:.1f} dBc)")

# ---- 7. Ring buffer: faded under-runs, exact reads ---------------------------
from audio_multi import _AudioRingBuffer, _DelayBuffer, _Varispeed

rb = _AudioRingBuffer(4800, 2)
rb.write(np.ones((256, 2), np.float32))
o1 = rb.read_out(512)
check("Ring read_out pads and fades",
      o1.shape == (512, 2) and abs(o1[0, 0]) < 0.05
      and np.all(o1[256:] == 0) and rb.underruns == 1)
rb.write(np.ones((512, 2), np.float32))
o2 = rb.read_out(512)
check("Ring resume fades back in",
      o2[0, 0] < 0.1 and abs(o2[-1, 0] - 1.0) < 1e-6)

# ---- 8. Delay buffer: exact tap + crossfaded changes -------------------------
db = _DelayBuffer(1000, 2)
xd = np.arange(512, dtype=np.float32).reshape(-1, 1).repeat(2, 1)
check("Delay 0 = identity", np.allclose(db.process(xd, 0), xd))
db.process(xd + 512, 100)                        # crossfade block 0 -> 100
y3 = db.process(xd + 1024, 100)                  # steady state
check("Delay tap is exact (no hidden block offset)",
      np.allclose(y3[:, 0], np.arange(1024 - 100, 1536 - 100)))
y4 = db.process(xd + 1536, 200)                  # change: must be smooth
check("Delay change crossfades (no click)",
      np.abs(np.diff(y4[:, 0])).max() < 6.0,
      f"(max step {np.abs(np.diff(y4[:, 0])).max():.2f})")

# ---- 9. Varispeed servo resampler --------------------------------------------
vs = _Varispeed(1)
tot = sum(len(vs.process(np.zeros((512, 1), np.float32), 1.001))
          for _ in range(100))
check("Varispeed rate correct", abs(tot - 51200 / 1.001) < 6, f"(n={tot})")
vs2 = _Varispeed(1)
t0, chunks = 0, []
for r in [1.0, 1.001, 0.999, 1.0005, 1.0] * 20:
    t = (t0 + np.arange(512)) / fs
    t0 += 512
    chunks.append(vs2.process(
        np.sin(2 * np.pi * 1000 * t).astype(np.float32).reshape(-1, 1), r))
yv = np.concatenate(chunks)[:, 0]
check("Varispeed is phase-continuous",
      np.abs(np.diff(yv)).max() < 0.14,
      f"(max step {np.abs(np.diff(yv)).max():.3f})")

# ---- 10. Upmix: centre extraction + pan gating --------------------------------
from dsp.surround_engine import _AdaptiveUpmix71

up = _AdaptiveUpmix71(fs)
mono_sig = rng.standard_normal(512).astype(np.float64) * 0.1
res = up.process(np.stack([mono_sig, mono_sig], axis=1).astype(np.float32))
# FL should be L - 0.5*C = 0.5*L for mono input
check("Upmix subtracts centre from fronts",
      np.allclose(res["FL"], mono_sig * 0.5, atol=1e-6)
      and np.allclose(res["C"], mono_sig, atol=1e-6))

up2 = _AdaptiveUpmix71(fs)
hard_left = np.stack([mono_sig, np.zeros_like(mono_sig)], axis=1).astype(np.float32)
for _ in range(30):                       # let pan/coh smoothing settle
    r2 = up2.process(hard_left)
ls_rms = float(np.sqrt(np.mean(r2["LS"] ** 2)))
fl_rms = float(np.sqrt(np.mean(r2["FL"] ** 2)))
check("Pan-gated coherence keeps hard-panned dry source out of surrounds",
      ls_rms < fl_rms * 1.2, f"(LS {ls_rms:.4f} vs FL {fl_rms:.4f})")

# ---- 11. Mono renderer: direct clean, S rescued -------------------------------
from dsp.surround_engine import VirtualSurroundMono
from config import SINGLE_SPEAKER_PRESET

vmn = VirtualSurroundMono(fs=fs, preset=dict(SINGLE_SPEAKER_PRESET))
xm = np.tile(rng.standard_normal((4800, 1)).astype(np.float32) * 0.1, (1, 2))
ym_ = np.concatenate([vmn.process(xm[i:i+512]) for i in range(0, 4096, 512)])
r_mono = float(np.sqrt((ym_ ** 2).mean()) / np.sqrt((xm ** 2).mean()))
check("Mono renderer ~unity on correlated input (x0.58 level match)",
      0.45 < r_mono < 0.75, f"(ratio {r_mono:.3f})")

vmn.reset()
sd_ = rng.standard_normal(4800).astype(np.float32) * 0.1
xs_ = np.stack([sd_, -sd_], axis=1)
ys_ = np.concatenate([vmn.process(xs_[i:i+512]) for i in range(0, 4096, 512)])
r_side = float(np.sqrt((ys_ ** 2).mean()) / np.sqrt((xs_ ** 2).mean()))
check("Mono renderer rescues side content (plain downmix = 0)",
      r_side > 0.08, f"(ratio {r_side:.3f})")

# ---- 12. Bass enhancer: consecutive harmonics ---------------------------------
from dsp.enhancer import HarmonicBassEnhancer

enh = HarmonicBassEnhancer(cutoff=120.0, drive=2.8, level=0.5, fs=fs)
tt = np.arange(fs * 2) / fs
x40 = np.stack([0.5 * np.sin(2 * np.pi * 40 * tt)] * 2, axis=1).astype(np.float32)
yh_ = np.concatenate([enh.process(x40[i:i+512])
                      for i in range(0, fs * 2 - 512, 512)])
Yh = np.abs(np.fft.rfft(yh_[fs:, 0] * np.hanning(len(yh_) - fs)))
fh = np.fft.rfftfreq(len(yh_) - fs, 1 / fs)
def _db_at(freq):
    return 20 * np.log10(Yh[np.argmin(np.abs(fh - freq))] + 1e-12)
h2 = _db_at(80) - _db_at(40)
h3 = _db_at(120) - _db_at(40)
check("Bass enhancer: 2nd AND 3rd harmonics present",
      h2 > -40 and h3 > -35, f"(H2 {h2:.1f} dBc, H3 {h3:.1f} dBc)")

# ---- 13. FDN delays scale with fs ---------------------------------------------
from dsp.reverb import _fdn_delays_for_fs
d44 = _fdn_delays_for_fs(44100)
d48 = _fdn_delays_for_fs(48000)
check("FDN delays scale to 44.1 kHz",
      np.all(np.abs(d44 / 44100 - d48 / 48000) < 0.001))

# ---- 14. Device output copy/downmix -------------------------------------------
from audio_io import _write_output

stereo = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)
mono_out = np.empty((2, 1), dtype=np.float32)
multi_out = np.ones((2, 4), dtype=np.float32)
_write_output(mono_out, stereo)
_write_output(multi_out, stereo)
check("Mono devices receive a proper stereo downmix",
      np.allclose(mono_out[:, 0], stereo.mean(axis=1)))
check("Multichannel devices receive stereo only",
      np.allclose(multi_out[:, :2], stereo) and np.all(multi_out[:, 2:] == 0))

# ---- 15. CLI/config trust-boundary validation ---------------------------------
from main import parse_args
from pi_runner import validate_config
from contextlib import redirect_stderr
from io import StringIO

def rejects(call):
    try:
        with redirect_stderr(StringIO()):
            call()
        return False
    except SystemExit:
        return True

check("CLI rejects invalid DSP parameters",
      rejects(lambda: parse_args(["--fs", "0"]))
      and rejects(lambda: parse_args(["--rt60", "0"]))
      and rejects(lambda: parse_args(["--block-size", "999999"])))
bad_pi = {"input_device": 0, "sample_rate": "bad", "speakers": [
    {"device": 1, "az": 0}, {"device": 2, "az": 30}]}
check("Pi config rejects invalid DSP parameters",
      rejects(lambda: validate_config(bad_pi)))

# ---- 16. Single-output chain swaps --------------------------------------------
from app import ModAudioApp

class ConstantChain:
    def __init__(self, value): self.value = value
    def process(self, block): return np.full_like(block, self.value)

class TheaterState:
    pass

state = TheaterState()
old_chain, new_chain = ConstantChain(1.0), ConstantChain(-1.0)
state._chain = new_chain
state._chain_transition = (new_chain, old_chain, 3)
fade = [ModAudioApp._process_theater(state, np.zeros((512, 2), np.float32))
        for _ in range(3)]
check("Single-output chain swaps crossfade without a jump",
      fade[0][0, 0] == 1.0 and fade[-1][-1, 0] == -1.0
      and state._chain_transition is None)

print()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)

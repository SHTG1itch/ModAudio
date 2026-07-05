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

print()
print("ALL PASS" if ok else "SOME CHECKS FAILED")

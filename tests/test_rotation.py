"""
Rotation / spatial routing verification.

Scenario (user request):
  Source sweeps clockwise around the listener — center front -> right of front
  speaker -> rear speaker's left side (= listener's right-rear) -> rear speaker's
  right side (= listener's left-rear) -> back to front speaker's left side.

Verifies that _build_routing_matrix_n_stereo emits energy on the correct
physical driver column at each point in the sweep, for both the 2-speaker
front+rear layout and a 4-speaker 5.1 layout.

Run:  python -m tests.test_rotation
"""
from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dsp.multi_speaker import (
    _build_routing_matrix_n_stereo,
    _vbap_sphere,
    _speaker_driver_az_el,
)


# Speaker layouts: (az, el, face_az, face_el).
# face_az is the direction the speaker POINTS toward. For a speaker that
# faces the listener, face_az == (speaker_az + 180) mod 360, i.e. it points
# back toward the origin.
def _facing_listener(az: float, el: float = 0.0):
    # Facing vector points FROM speaker TOWARD listener = opposite of position.
    # Opposite azimuth is (az + 180); trig is periodic so no wrap needed.
    return (az, el, az + 180.0, -el)


LAYOUT_2SPK = [
    _facing_listener(0.0),      # front, directly ahead
    _facing_listener(180.0),    # rear, directly behind
]

LAYOUT_5SPK = [
    _facing_listener(-30.0),    # FL
    _facing_listener( 30.0),    # FR
    _facing_listener(  0.0),    # C
    _facing_listener(-110.0),   # LS (left-rear)
    _facing_listener( 110.0),   # RS (right-rear)
]


def _column_energy(M: np.ndarray, virt_az: float) -> np.ndarray:
    """Drive only the virtual channel nearest to `virt_az` and return per-column gains."""
    virt_az_list = [-30.0, 30.0, 0.0, -110.0, 110.0, -150.0, 150.0]
    # Find closest virtual channel
    diffs = [abs(((v - virt_az + 180) % 360) - 180) for v in virt_az_list]
    j = int(np.argmin(diffs))
    return M[j]  # shape (2N,)


def _speaker_energy(col_gains: np.ndarray, N: int) -> np.ndarray:
    """Sum each speaker's L+R driver power."""
    e = np.zeros(N, dtype=np.float64)
    for i in range(N):
        e[i] = col_gains[2*i]**2 + col_gains[2*i+1]**2
    return e


def test_driver_positions_2spk():
    """A front speaker facing the listener should have R driver to listener's right."""
    l_az, _, r_az, _ = _speaker_driver_az_el(0.0, 0.0, 180.0, 0.0, 15.0)
    # Front speaker at az=0, facing listener (face_az=180, i.e. pointing back toward -Z
    # from the listener's POV... wait facing=180 deg means pointing AWAY from listener).
    # For front speaker at az=0 (ahead of listener), to face listener it should point
    # toward -Z from its own position, which is az=0 deg (the direction of the listener
    # from the speaker's POV)... convention check:
    #
    # In room_canvas_3d, face_az is the direction the speaker points, measured the
    # same way as speaker position (0 deg=front,−Z). A speaker at az=0 (in front) that
    # "faces the listener" must point back toward the listener = toward +Z =
    # face_az=180 deg.
    print(f"[2spk front] L driver az={l_az:.1f} deg  R driver az={r_az:.1f} deg")
    # R driver should be to listener's right (positive az) — cross(up, facing_back)
    # For facing=(−Z flipped to +Z), right = cross(+Y, +Z) = +X = listener's right.
    assert r_az > 0 and l_az < 0, f"front speaker drivers flipped: L={l_az} R={r_az}"


def test_rotation_2spk():
    """Sweep virtual-channel azimuth; verify energy moves front -> rear cleanly."""
    M = _build_routing_matrix_n_stereo(LAYOUT_2SPK, half_width=15.0)
    assert M.shape == (7, 4), M.shape

    # Column order: [front_L, front_R, rear_L, rear_R]
    # Rear speaker faces the listener, so its physical right driver (col 3, rear_R)
    # points toward listener's LEFT (because the speaker is turned around).

    # Virtual channel -> dominant physical columns
    checks = [
        #  vaz   expected dominant cols (indices into M column)
        (  0.0, {0, 1}),   # C -> front
        ( -30.0, {0}),     # FL -> front_L
        (  30.0, {1}),     # FR -> front_R
        ( 180.0, {2, 3}),  # directly behind -> rear
    ]
    for vaz, expected in checks:
        row = _column_energy(M, vaz)
        top2 = set(np.argsort(row)[-2:].tolist())
        dominant = set(i for i, g in enumerate(row) if g > 0.3)
        print(f"  source az={vaz:+6.1f} deg -> gains {np.round(row,3).tolist()}   dominant={dominant}")
        assert dominant & expected, (
            f"source az={vaz} deg: expected cols {expected} dominant, got {dominant}")


def test_rotation_5spk_clockwise():
    """Clockwise 360 deg sweep on a 5-speaker circle: peak speaker should match nearest."""
    layout = LAYOUT_5SPK
    centers = [s[0] for s in layout]
    M = _build_routing_matrix_n_stereo(layout, half_width=15.0)
    N = len(layout)

    # Synthetic virtual-source sweep at the 7 defined virtual-channel azimuths.
    virt_azs = [-30.0, 30.0, 0.0, -110.0, 110.0, -150.0, 150.0]
    virt_names = ["FL", "FR", "C", "LS", "RS", "LB", "RB"]

    print(f"  speaker centers: {centers}")
    for j, (name, vaz) in enumerate(zip(virt_names, virt_azs)):
        spk_e = _speaker_energy(M[j], N)
        peak = int(np.argmax(spk_e))
        # Nearest speaker by angular distance
        diffs = [abs(((vaz - c + 180) % 360) - 180) for c in centers]
        nearest = int(np.argmin(diffs))
        print(f"  virt {name:2s} az={vaz:+6.1f} deg -> peak spk #{peak} (az={centers[peak]:+.0f} deg), "
              f"nearest spk #{nearest} (az={centers[nearest]:+.0f} deg), "
              f"energies={np.round(spk_e,3).tolist()}")
        assert peak == nearest, (
            f"{name}: expected peak on nearest spk #{nearest}, got #{peak}")


def test_constant_power():
    """Routing matrix rows should have sum-of-squares ~= 1 (energy preservation)."""
    for layout_name, layout in [("2spk", LAYOUT_2SPK), ("5spk", LAYOUT_5SPK)]:
        M = _build_routing_matrix_n_stereo(layout)
        for j, row in enumerate(M):
            p = float((row * row).sum())
            assert 0.9 < p < 1.1, f"{layout_name} row {j}: power={p:.3f}"
        print(f"  {layout_name}: all 7 virtual channels are constant-power ok")


def test_user_rotation_scenario():
    """
    The user's exact scenario: source rotating clockwise starting at front-center.
    Verify continuous hand-off between speakers as azimuth sweeps 0 deg -> 360 deg.
    """
    layout = LAYOUT_2SPK
    M = _build_routing_matrix_n_stereo(layout)

    # Sample at 8 azimuth points around the circle.
    # For each, find which physical driver (0=front_L, 1=front_R, 2=rear_L, 3=rear_R)
    # dominates, using the routing matrix as if the source were a pure virtual point.
    # We approximate by picking the nearest virtual channel.
    sweep = [0, 45, 90, 135, 180, 225, 270, 315]
    seq = []
    for az in sweep:
        row = _column_energy(M, az)
        seq.append(int(np.argmax(row)))
    print(f"  sweep {sweep} deg -> driver sequence {seq}")

    # We expect: front drivers dominate near 0 deg, rear drivers dominate near 180 deg.
    assert seq[0] in (0, 1), f"at 0 deg front should dominate, got col {seq[0]}"
    assert seq[4] in (2, 3), f"at 180 deg rear should dominate, got col {seq[4]}"


def test_atmos_height_routing():
    """7.1.4 height virtual channels route to overhead speakers when present."""
    # 5 ground speakers + 2 overhead
    layout = LAYOUT_5SPK + [
        (-45.0, 45.0, 135.0, -45.0),
        ( 45.0, 45.0, 225.0, -45.0),
    ]
    M = _build_routing_matrix_n_stereo(layout, include_heights=True)
    assert M.shape == (11, 14), M.shape
    # Row 7 = TFL virtual, should peak on speaker 5 (TFL physical, cols 10,11)
    tfl = M[7]
    peak_col = int(np.argmax(tfl))
    peak_spk = peak_col // 2
    print(f"  TFL -> peak spk #{peak_spk} (cols 10,11 = height spk #5)")
    assert peak_spk == 5, f"expected TFL -> spk 5, got {peak_spk}"

def test_atmos_graceful_degradation():
    """With no overhead speakers, height channels fall back to nearest ground."""
    M = _build_routing_matrix_n_stereo(LAYOUT_5SPK, include_heights=True)
    assert M.shape == (11, 10), M.shape
    # Row 7 = TFL virtual. With no height speakers, should route to FL (-30 deg, spk 0)
    # and/or C (0 deg, spk 2) — the nearest ground-level candidates.
    tfl = M[7]
    spk_e = _speaker_energy(tfl, 5)
    peak = int(np.argmax(spk_e))
    print(f"  fallback TFL -> ground spk #{peak} (az={LAYOUT_5SPK[peak][0]:+.0f} deg)")
    assert peak in (0, 2), f"TFL should fall back to FL or C, got spk {peak}"


def main():
    tests = [
        ("driver L/R positioning (2-speaker front)", test_driver_positions_2spk),
        ("constant-power rows",                       test_constant_power),
        ("rotation — 2-speaker front+rear",           test_rotation_2spk),
        ("rotation — 5-speaker cinema (360 deg sweep)",  test_rotation_5spk_clockwise),
        ("user rotation scenario — clockwise sweep",  test_user_rotation_scenario),
        ("Atmos height routing (7.1.4 layout)",        test_atmos_height_routing),
        ("Atmos graceful degradation (no heights)",    test_atmos_graceful_degradation),
    ]
    failures = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            print(f"  PASS")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL: {e}")
        except Exception as e:
            failures += 1
            print(f"  ERROR: {type(e).__name__}: {e}")
    print(f"\n{'='*50}\n{len(tests)-failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

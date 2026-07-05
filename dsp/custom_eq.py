"""
Custom parametric EQ — applied on top of the cinema EQ chain.

10 fixed bands:
  0  : High-pass  (HP) — protects against extreme sub-bass rumble
  1  : Low shelf  (LS)
  2–8: Peaking bands
  9  : High shelf (HS)

Thread safety
-------------
The UI thread calls update_band() which writes to _target_* arrays.
The audio thread reads _target_coeffs and smooths _current_coeffs toward
them at each process() call.  Numpy scalar/array element assignment is
GIL-atomic, so no explicit lock is needed.

The filter state (_zi) is private to the audio thread and is never
touched from the UI thread.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, freqz
from .filters import _peaking, _lowshelf, _highshelf, _highpass, _lowpass


DEFAULT_BANDS: list[tuple] = [
    # (type,        freq_hz,  gain_db,  q     )
    ("highpass",    20.0,     0.0,      0.707 ),
    ("lowshelf",    80.0,     0.0,      0.707 ),
    ("peaking",    125.0,     0.0,      1.41  ),
    ("peaking",    250.0,     0.0,      1.41  ),
    ("peaking",    500.0,     0.0,      1.41  ),
    ("peaking",   1000.0,     0.0,      1.41  ),
    ("peaking",   2000.0,     0.0,      1.41  ),
    ("peaking",   4000.0,     0.0,      1.41  ),
    ("peaking",   8000.0,     0.0,      1.41  ),
    ("highshelf", 12000.0,    0.0,      0.707 ),
]


class ParametricEQ:
    """
    10-band parametric EQ with smooth, artefact-free parameter transitions.

    During interactive EQ editing the UI calls update_band() which writes
    target coefficients.  The audio thread smooths toward the targets
    at _SMOOTH fraction per block (≈50 ms to settle at 512 sample blocks).
    This prevents audible pops when dragging EQ nodes.
    """

    N_BANDS = len(DEFAULT_BANDS)
    _SMOOTH = 0.50   # blend fraction per block toward target coefficients

    def __init__(
        self,
        fs: int = 48000,
        num_channels: int = 2,
        bands: list | None = None,
    ):
        self._fs = fs
        self._ch = num_channels

        if bands is None:
            bands = list(DEFAULT_BANDS)

        self._band_types: list[str] = [b[0] for b in bands]

        # Target arrays written by UI thread, read by audio thread.
        self._target_freq    = np.array([b[1] for b in bands], dtype=np.float64)
        self._target_gain    = np.array([b[2] for b in bands], dtype=np.float64)
        self._target_q       = np.array([b[3] for b in bands], dtype=np.float64)
        self._target_enabled = np.ones(len(bands), dtype=bool)

        # [N_bands, 5] = [b0, b1, b2, a1, a2] — written by UI, read by audio.
        self._target_coeffs = np.zeros((len(bands), 5), dtype=np.float64)

        # Current coefficients — audio thread only.
        self._current_coeffs = np.zeros((len(bands), 5), dtype=np.float64)

        # Per-band biquad filter state — audio thread only.
        self._zi: list[np.ndarray] = [
            np.zeros((2, num_channels), dtype=np.float64)
            for _ in range(len(bands))
        ]

        # Compute initial flat-response coefficients.
        self._recompute_all_targets()
        self._current_coeffs[:] = self._target_coeffs

    # ------------------------------------------------------------------
    # Internal coefficient computation

    @staticmethod
    def _flat() -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _compute(self, idx: int) -> np.ndarray:
        btype = self._band_types[idx]
        freq  = float(np.clip(self._target_freq[idx], 10.0, self._fs * 0.499))
        gain  = float(self._target_gain[idx])
        q     = float(np.clip(self._target_q[idx], 0.1, 20.0))

        if not self._target_enabled[idx]:
            return self._flat()
        if btype in ("peaking", "lowshelf", "highshelf") and abs(gain) < 0.01:
            return self._flat()

        try:
            if btype == "peaking":
                b, a = _peaking(freq, gain, q, self._fs)
            elif btype == "lowshelf":
                b, a = _lowshelf(freq, gain, q, self._fs)
            elif btype == "highshelf":
                b, a = _highshelf(freq, gain, q, self._fs)
            elif btype == "highpass":
                b, a = _highpass(freq, q, self._fs)
            elif btype == "lowpass":
                b, a = _lowpass(freq, q, self._fs)
            else:
                return self._flat()
        except Exception:
            return self._flat()

        return np.array([b[0], b[1], b[2], a[1], a[2]], dtype=np.float64)

    def _recompute_all_targets(self) -> None:
        for i in range(self.N_BANDS):
            self._target_coeffs[i] = self._compute(i)

    # ------------------------------------------------------------------
    # UI-thread interface

    def update_band(
        self,
        idx: int,
        freq:     float | None = None,
        gain_db:  float | None = None,
        q:        float | None = None,
        enabled:  bool  | None = None,
    ) -> None:
        """Update one band's parameters.  Safe to call from the UI thread."""
        if freq    is not None: self._target_freq[idx]    = float(freq)
        if gain_db is not None: self._target_gain[idx]    = float(gain_db)
        if q       is not None: self._target_q[idx]       = float(q)
        if enabled is not None: self._target_enabled[idx] = bool(enabled)
        self._target_coeffs[idx] = self._compute(idx)

    def set_all_bands(self, bands: list[dict]) -> None:
        """Set all bands from a serialisable list of dicts.  UI thread."""
        for i, p in enumerate(bands[:self.N_BANDS]):
            self.update_band(
                i,
                freq    = p.get("freq"),
                gain_db = p.get("gain_db"),
                q       = p.get("q"),
                enabled = p.get("enabled"),
            )

    def get_band_params(self) -> list[dict]:
        """Return current target parameters as a serialisable list of dicts."""
        return [
            {
                "type":     self._band_types[i],
                "freq":     float(self._target_freq[i]),
                "gain_db":  float(self._target_gain[i]),
                "q":        float(self._target_q[i]),
                "enabled":  bool(self._target_enabled[i]),
            }
            for i in range(self.N_BANDS)
        ]

    def reset_to_flat(self) -> None:
        """Set all gains to 0 dB (flat response).  UI thread."""
        self._target_gain[:] = 0.0
        self._recompute_all_targets()

    def is_flat(self) -> bool:
        """True when all gains are effectively 0 dB."""
        return bool(np.all(np.abs(self._target_gain) < 0.01))

    # ------------------------------------------------------------------
    # Audio-thread processing

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        Apply all EQ bands to block (N, C) → (N, C).

        Smoothly interpolates toward target coefficients each call to
        avoid clicks during live parameter updates.
        """
        x = block.astype(np.float64, copy=True)

        for i in range(self.N_BANDS):
            target  = self._target_coeffs[i]
            current = self._current_coeffs[i]
            diff    = target - current
            if np.any(np.abs(diff) > 1e-12):
                self._current_coeffs[i] = current + self._SMOOTH * diff

            c  = self._current_coeffs[i]
            b  = c[:3]
            a  = np.array([1.0, c[3], c[4]])

            # Skip identity filter (saves CPU for flat bands)
            if (abs(b[0] - 1.0) < 1e-10
                    and abs(b[1]) < 1e-10
                    and abs(b[2]) < 1e-10
                    and abs(a[1]) < 1e-10
                    and abs(a[2]) < 1e-10):
                continue

            out = np.empty_like(x)
            for ch in range(self._ch):
                col, self._zi[i][:, ch] = lfilter(
                    b, a, x[:, ch], zi=self._zi[i][:, ch])
                out[:, ch] = col
            x = out

        return x.astype(block.dtype, copy=False)

    def reset(self) -> None:
        """Clear biquad state.  Must be called from the audio thread."""
        for i in range(self.N_BANDS):
            self._zi[i][:] = 0.0

    # ------------------------------------------------------------------
    # Visualisation helpers (UI thread)

    def get_frequency_response(self, freqs: np.ndarray) -> np.ndarray:
        """Combined EQ gain in dB at each frequency (uses target coefficients)."""
        H = np.ones(len(freqs), dtype=np.complex128)
        for i in range(self.N_BANDS):
            c = self._target_coeffs[i]
            b = c[:3]
            a = np.array([1.0, c[3], c[4]])
            if (abs(b[0] - 1.0) < 1e-10
                    and abs(b[1]) < 1e-10
                    and abs(b[2]) < 1e-10):
                continue
            w = 2.0 * np.pi * freqs / self._fs
            _, h = freqz(b, a, worN=w)
            H *= h
        return 20.0 * np.log10(np.maximum(np.abs(H), 1e-10))

    def get_band_response(self, band_idx: int, freqs: np.ndarray) -> np.ndarray:
        """Frequency response in dB for a single band."""
        c = self._target_coeffs[band_idx]
        b = c[:3]
        a = np.array([1.0, c[3], c[4]])
        if abs(b[0] - 1.0) < 1e-10 and abs(b[1]) < 1e-10 and abs(b[2]) < 1e-10:
            return np.zeros(len(freqs))
        w = 2.0 * np.pi * freqs / self._fs
        _, h = freqz(b, a, worN=w)
        return 20.0 * np.log10(np.maximum(np.abs(h), 1e-10))

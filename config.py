"""
ModAudio - Theater Experience Configuration
"""

# -- Audio I/O -----------------------------------------------------------------

SAMPLE_RATE = 48000   # Hz  (48k is standard for cinema / pro audio)
BLOCK_SIZE  = 512     # samples (~10.7 ms latency at 48 kHz)
CHANNELS    = 2       # stereo I/O

# -- Theater DSP Preset - HEADPHONES ------------------------------------------
#
# Full binaural rendering: Brown-Duda HRTF, virtual 5-channel surround,
# psychoacoustic bass synthesis, multi-band dynamics, FDN reverb.

HEADPHONES_PRESET = {
    "mode": "headphones",

    # -- Room acoustics (large commercial cinema) ------------------------------
    "rt60":                1.3,    # Reverberation time (s)
    "rt60_hf":             0.65,   # HF RT60 (air absorbs treble faster)
    "reverb_predelay_ms": 22.0,    # Silence before reverb (sense of scale)
    "reverb_mix":          0.25,   # Reverb tail wet level
    "early_ref_mix":       0.45,   # Early reflections level

    # -- Spatialization (headphones - binaural HRTF) ---------------------------
    "stereo_width":   2.0,         # Air-band M/S width (1.0 = neutral)
    "surround_level": 0.72,        # LS/RS channel level
    "center_level":   0.88,        # Center channel level
    "lfe_level":      0.85,        # Sub-bass LFE level

    # Virtual speaker azimuths (degrees, positive = right)
    "speaker_L_az":  -30.0,
    "speaker_R_az":   30.0,
    "speaker_C_az":    0.0,
    "speaker_LS_az": -110.0,
    "speaker_RS_az":  110.0,

    # -- Cinema EQ (X-curve + bass extension) ---------------------------------
    "sub_bass_hz":    30,
    "sub_bass_db":    4.5,
    "bass_boost_hz":  80,
    "bass_boost_db":  6.0,
    "presence_hz":    3500,
    "presence_db":    2.0,
    "xcurve_hz":      2000,
    "xcurve_db":     -6.0,

    # -- Harmonic enhancement -------------------------------------------------
    "bass_harm_drive":  2.8,       # Sub-bass saturation drive (generates harmonics)
    "bass_harm_level":  0.50,      # Harmonic bass mix level
    "air_exciter_level": 0.18,     # Air-band exciter mix level (sparkle)

    # -- Dynamics (multi-band compressor + transient enhancer) -----------------
    "mb_compress_drive":   1.6,    # 1.0=gentle, 2.0=theatrical
    "transient_amount":    0.55,   # Transient punch (0=off, 1=max)

    # -- Output ---------------------------------------------------------------
    "output_gain_db":    -1.5,
    "limiter_threshold":  0.93,
    "limiter_release_ms": 80.0,
}

# -- Theater DSP Preset - SPEAKERS --------------------------------------------
#
# Stereo speakers: no HRTF (room does natural head-filtering), instead uses
# stereo widening, Haas-effect depth, reverb, and dynamics.

SPEAKERS_PRESET = {
    "mode": "speakers",

    # -- Room acoustics -------------------------------------------------------
    "rt60":                1.3,
    "rt60_hf":             0.65,
    "reverb_predelay_ms": 22.0,
    "reverb_mix":          0.30,   # Slightly more reverb (room absorption varies)
    "early_ref_mix":       0.40,

    # -- Spatialization (speakers - no HRTF) ----------------------------------
    "stereo_width":   1.8,         # Width beyond the speaker baseline
    "surround_level": 0.65,        # Haas-delay surround level
    "center_level":   0.90,
    "lfe_level":      0.90,        # More bass for speakers (typically better LF)
    "haas_delay_ms":  22.0,        # Haas delay for stereo depth

    # -- Cinema EQ ------------------------------------------------------------
    "sub_bass_hz":    30,
    "sub_bass_db":    3.5,
    "bass_boost_hz":  80,
    "bass_boost_db":  4.5,
    "presence_hz":    3500,
    "presence_db":    1.5,
    "xcurve_hz":      2000,
    "xcurve_db":     -4.0,         # Less rolloff for typical room HF absorption

    # -- Harmonic enhancement -------------------------------------------------
    "bass_harm_drive":   2.0,
    "bass_harm_level":   0.35,
    "air_exciter_level": 0.12,

    # -- Dynamics -------------------------------------------------------------
    "mb_compress_drive": 1.4,
    "transient_amount":  0.45,

    # -- Output ---------------------------------------------------------------
    "output_gain_db":    -2.0,
    "limiter_threshold":  0.93,
    "limiter_release_ms": 80.0,
}

# Default preset alias
THEATER_PRESET = HEADPHONES_PRESET

# -- Head model (binaural HRTF) -----------------------------------------------
HEAD_RADIUS_M  = 0.0875   # average human head radius in metres
SOUND_SPEED_MS = 343.0    # speed of sound at 20 deg C

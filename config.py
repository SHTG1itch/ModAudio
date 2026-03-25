"""
ModAudio - Theater Experience Configuration
"""

# -- Audio I/O -----------------------------------------------------------------

SAMPLE_RATE = 48000   # Hz  (48k is standard for cinema / pro audio)
BLOCK_SIZE  = 512     # samples (~10.7 ms latency at 48 kHz)
CHANNELS    = 2       # stereo I/O

# -- Theater DSP Preset -------------------------------------------------------
#
# All parameters are tuned to recreate a large commercial movie theater:
#   * RT60 ~1.2 s  (slightly drier than a concert hall, more focused)
#   * Wide soundstage with prominent surround envelopment
#   * Cinema X-curve EQ (flat to 2 kHz, then gentle HF rolloff)
#   * Sub-bass reinforcement for the "chest thump" LFE sensation

THEATER_PRESET = {
    # -- Room acoustics --------------------------------------------------------
    "rt60": 1.2,                # Reverberation time in seconds
    "rt60_hf": 0.6,             # HF RT60 (air absorbs HF faster)
    "reverb_mix": 0.22,         # Wet/dry ratio for reverb tail (0-1)
    "early_ref_mix": 0.40,      # Early reflections level
    "early_ref_delay_ms": 28,   # First reflection delay in ms

    # -- Spatialization --------------------------------------------------------
    "stereo_width": 1.9,        # M/S side channel gain (1.0 = neutral)
    "surround_level": 0.70,     # LS/RS channel level relative to L/R
    "center_level": 0.85,       # Center channel level
    "lfe_level": 0.80,          # Sub-bass LFE channel level

    # Virtual speaker azimuths (degrees, positive = right)
    "speaker_L_az":  -30.0,
    "speaker_R_az":   30.0,
    "speaker_C_az":    0.0,
    "speaker_LS_az": -110.0,
    "speaker_RS_az":  110.0,

    # -- Cinema EQ (X-curve + bass extension) ---------------------------------
    "sub_bass_hz":    30,       # Sub-bass shelf frequency
    "sub_bass_db":    4.0,      # Sub-bass boost
    "bass_boost_hz":  80,       # Bass boost frequency (LFE feel)
    "bass_boost_db":  5.0,      # Bass boost in dB
    "presence_hz":    3500,     # Presence peak for dialog clarity
    "presence_db":    1.5,      # Presence boost in dB
    "xcurve_hz":      2000,     # X-curve HF rolloff start frequency
    "xcurve_db":     -6.0,      # Total HF attenuation at Nyquist

    # -- Dynamics --------------------------------------------------------------
    "output_gain_db":      -2.0,   # Master output trim (headroom)
    "limiter_threshold":    0.93,  # Peak limiter ceiling
    "limiter_release_ms":  80.0,   # Limiter release time
}

# -- Head model (for binaural HRTF) -------------------------------------------
HEAD_RADIUS_M  = 0.0875   # average human head radius in metres
SOUND_SPEED_MS = 343.0    # speed of sound at 20 deg C
MAX_ITD_SAMPLES = int(round(
    (HEAD_RADIUS_M / SOUND_SPEED_MS) * (1.0 + 1.0) * SAMPLE_RATE
))  # upper bound on interaural delay in samples (~25 samples at 48 kHz)

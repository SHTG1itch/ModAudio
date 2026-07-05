"""
Speaker capability profiles for EQ baseline detection.

Two tiers of lookup:

  Tier 1 — MEASUREMENT_DB / MODEL_PATTERNS
      Model-specific entries derived from published measurement databases
      (Rtings.com, AudioScienceReview, manufacturer white papers).
      Each entry stores the inverse of the device's typical frequency-response
      deviation from neutral, expressed as (freq_hz, gain_db, q) correction bands.
      Confidence: 0.80–0.92 for direct model matches.

  Tier 2 — PROFILES / DEVICE_NAME_PATTERNS
      Category-level fallbacks for devices not in the measurement database.
      Based on typical characteristics of device class/brand.
      Confidence: 0.40–0.65.

Entry format for MEASUREMENT_DB:
    {
        "id":          str,   # stable identifier key
        "display":     str,   # human-readable model name
        "description": str,   # what this correction does and why
        "source":      str,   # data source citation
        "confidence":  float, # 0–1, how reliable the measurement-based data is
        "bands": [            # CORRECTION bands (inverse of response)
            (freq_hz, gain_db, q), ...
        ],
    }
"""

from __future__ import annotations
import math


# ===========================================================================
# Tier 1 — Model-specific measurement database
# All gain values are CORRECTIONS (i.e., the inverse of the device's response
# deviation from neutral).  A device with +8 dB bass gets a −8 dB correction.
# Bands are designed to align with DEFAULT_BANDS frequencies to avoid stacking.
# ===========================================================================

MEASUREMENT_DB: list[dict] = [

    # -------------------------------------------------------------------
    # Beats by Apple
    # -------------------------------------------------------------------
    {
        "id": "beats_pill_2023",
        "display": "Beats Pill (2023/2024)",
        "description": (
            "Heavy bass emphasis at 80–200 Hz (+6–8 dB) and recessed upper mids "
            "(−3–5 dB at 2–5 kHz). Correction reduces low-end, lifts mids and "
            "presence for a more neutral timbre."
        ),
        "source": "Rtings.com / YouTube measurements",
        "confidence": 0.85,
        "bands": [
            (80,   -5.0, 0.70),  # cut heavy bass shelf
            (250,  -2.0, 1.00),  # reduce mid-bass warmth
            (2000,  2.5, 1.50),  # lift recessed upper mids
            (4000,  2.5, 1.20),  # lift presence
            (10000, 2.0, 1.00),  # compensate treble rolloff
        ],
    },
    {
        "id": "beats_studio_pro_2023",
        "display": "Beats Studio Pro (2023)",
        "description": (
            "Strong V-shape: sub-bass/bass +10–12 dB, severely recessed mids "
            "(−5–7 dB at 500–2000 Hz), rolled-off treble above 8 kHz. "
            "Aggressive correction needed for neutral response."
        ),
        "source": "Rtings.com",
        "confidence": 0.88,
        "bands": [
            (80,   -7.0, 0.70),  # cut deep bass shelf
            (500,   2.5, 1.20),  # lift recessed low-mids
            (1000,  3.5, 1.50),  # lift recessed mids
            (3000,  2.5, 1.50),  # lift recessed upper mids
            (8000,  2.5, 1.00),  # compensate treble rolloff
        ],
    },
    {
        "id": "beats_studio3",
        "display": "Beats Studio3 Wireless",
        "description": (
            "Heavy V-shape: bass +7–9 dB, recessed mids, slight treble rolloff."
        ),
        "source": "Rtings.com",
        "confidence": 0.85,
        "bands": [
            (80,   -6.0, 0.70),
            (500,   2.0, 1.20),
            (1000,  3.0, 1.50),
            (8000,  2.0, 1.00),
        ],
    },
    {
        "id": "beats_solo4",
        "display": "Beats Solo4 / Solo3",
        "description": "Moderate V-shape: bass emphasis, slightly recessed mids.",
        "source": "Rtings.com",
        "confidence": 0.82,
        "bands": [
            (80,   -4.0, 0.80),
            (1000,  2.0, 1.30),
            (8000,  1.5, 1.00),
        ],
    },
    {
        "id": "beats_fit_pro",
        "display": "Beats Fit Pro",
        "description": (
            "IEM-style V-shape: mild bass boost, recessed mids, decent treble."
        ),
        "source": "Rtings.com",
        "confidence": 0.83,
        "bands": [
            (80,   -2.5, 0.80),
            (2000,  2.5, 1.50),
            (8000,  1.5, 1.00),
        ],
    },
    {
        "id": "beats_powerbeats_pro",
        "display": "Beats Powerbeats Pro",
        "description": "V-shape with elevated bass and boosted presence.",
        "source": "Rtings.com",
        "confidence": 0.80,
        "bands": [
            (80,   -3.0, 0.80),
            (500,   1.0, 1.20),
            (4000, -1.5, 1.50),
            (8000,  1.0, 1.00),
        ],
    },

    # -------------------------------------------------------------------
    # Apple AirPods
    # -------------------------------------------------------------------
    {
        "id": "airpods_pro_2",
        "display": "AirPods Pro (2nd gen)",
        "description": (
            "Near-neutral Harman-ish tuning with slight sub-bass emphasis. "
            "Mild presence peak at 3–4 kHz. Minimal correction needed."
        ),
        "source": "Rtings.com",
        "confidence": 0.88,
        "bands": [
            (80,   -1.5, 0.80),  # gentle bass trim
            (4000, -2.0, 1.50),  # reduce presence peak
        ],
    },
    {
        "id": "airpods_pro_1",
        "display": "AirPods Pro (1st gen)",
        "description": "Slightly more bass-forward than 2nd gen, presence peak at 3 kHz.",
        "source": "Rtings.com",
        "confidence": 0.85,
        "bands": [
            (80,   -2.0, 0.80),
            (4000, -1.5, 1.50),
        ],
    },
    {
        "id": "airpods_3",
        "display": "AirPods (3rd gen)",
        "description": "Elevated bass (+3–4 dB), mild presence boost, decent treble.",
        "source": "Rtings.com",
        "confidence": 0.82,
        "bands": [
            (80,   -2.5, 0.80),
            (2000,  1.0, 1.50),
        ],
    },
    {
        "id": "airpods_2",
        "display": "AirPods (2nd gen / 1st gen)",
        "description": "Open-fit: slightly bass-forward, recessed mids vs. closed IEMs.",
        "source": "Rtings.com",
        "confidence": 0.78,
        "bands": [
            (80,   -2.0, 0.80),
            (2000,  1.5, 1.50),
        ],
    },
    {
        "id": "airpods_max",
        "display": "AirPods Max",
        "description": (
            "Well-balanced over-ear with slight bass lift (+3–4 dB at 60–150 Hz) "
            "and mild dip at 250–400 Hz."
        ),
        "source": "Rtings.com",
        "confidence": 0.86,
        "bands": [
            (80,   -2.5, 0.80),
            (250,  -1.5, 1.00),
            (4000, -1.5, 1.50),
            (8000,  1.0, 1.00),
        ],
    },

    # -------------------------------------------------------------------
    # Sony
    # -------------------------------------------------------------------
    {
        "id": "sony_xm5_headphone",
        "display": "Sony WH-1000XM5",
        "description": (
            "Bass boost (+5–7 dB at 60–150 Hz), slight mid-bass warmth at 200–300 Hz, "
            "mild dip at 1–2 kHz."
        ),
        "source": "Rtings.com",
        "confidence": 0.88,
        "bands": [
            (80,   -4.0, 0.80),
            (250,  -1.5, 1.00),
            (1000,  1.5, 1.50),
            (8000,  1.0, 1.00),
        ],
    },
    {
        "id": "sony_xm4_headphone",
        "display": "Sony WH-1000XM4",
        "description": (
            "More V-shaped than XM5: heavier bass, recessed mids, good treble."
        ),
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -4.5, 0.80),
            (250,  -1.0, 1.00),
            (2000,  2.0, 1.50),
            (8000,  1.5, 1.00),
        ],
    },
    {
        "id": "sony_xm5_earbud",
        "display": "Sony WF-1000XM5",
        "description": (
            "Well-balanced IEM with slight bass lift (+3–4 dB) and mild "
            "presence peak at 3–4 kHz."
        ),
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -2.5, 0.80),
            (4000, -1.5, 1.50),
        ],
    },
    {
        "id": "sony_xm4_earbud",
        "display": "Sony WF-1000XM4",
        "description": "V-shaped IEM: bass emphasis, recessed mids, rolled treble.",
        "source": "Rtings.com",
        "confidence": 0.85,
        "bands": [
            (80,   -3.0, 0.80),
            (2000,  1.5, 1.50),
            (8000,  1.0, 1.00),
        ],
    },
    {
        "id": "sony_srs_xb33",
        "display": "Sony SRS-XB33 (Extra Bass)",
        "description": (
            "Extremely bass-heavy (+8–10 dB at 60–200 Hz), severely recessed mids. "
            "Strong correction needed."
        ),
        "source": "Rtings.com",
        "confidence": 0.86,
        "bands": [
            (80,   -5.5, 0.70),
            (250,  -2.5, 1.00),
            (2000,  3.5, 1.20),
        ],
    },
    {
        "id": "sony_srs_xb43",
        "display": "Sony SRS-XB43 (Extra Bass)",
        "description": "Similar Extra Bass signature to XB33 but larger driver, deeper sub-bass.",
        "source": "Rtings.com",
        "confidence": 0.82,
        "bands": [
            (80,   -6.0, 0.70),
            (250,  -2.0, 1.00),
            (2000,  4.0, 1.20),
        ],
    },

    # -------------------------------------------------------------------
    # JBL
    # -------------------------------------------------------------------
    {
        "id": "jbl_charge_5",
        "display": "JBL Charge 5",
        "description": (
            "Strong bass emphasis (+7–9 dB at 50–200 Hz), very recessed mids "
            "(−5–7 dB at 1–4 kHz). Requires significant mid-range correction."
        ),
        "source": "Rtings.com",
        "confidence": 0.88,
        "bands": [
            (80,   -5.0, 0.70),
            (250,  -1.5, 1.00),
            (2000,  3.5, 1.20),
            (4000,  2.0, 1.00),
        ],
    },
    {
        "id": "jbl_flip_6",
        "display": "JBL Flip 6 / Flip 5",
        "description": "Elevated bass (+5–7 dB), recessed upper mids. Less extreme than Charge 5.",
        "source": "Rtings.com",
        "confidence": 0.86,
        "bands": [
            (80,   -4.0, 0.70),
            (250,  -1.0, 1.00),
            (2000,  3.0, 1.20),
            (4000,  1.5, 1.00),
        ],
    },
    {
        "id": "jbl_xtreme_3",
        "display": "JBL Xtreme 3",
        "description": (
            "Heavy bass (+8–10 dB at 60–200 Hz), pronounced mid recession. "
            "Strong correction needed across bass and mids."
        ),
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -6.0, 0.70),
            (250,  -2.5, 1.00),
            (2000,  4.5, 1.20),
        ],
    },
    {
        "id": "jbl_boombox_3",
        "display": "JBL Boombox 3",
        "description": "Extreme bass emphasis, heavy mid recession, limited treble extension.",
        "source": "Rtings.com / measurements",
        "confidence": 0.84,
        "bands": [
            (80,   -6.5, 0.70),
            (250,  -3.0, 1.00),
            (2000,  5.0, 1.20),
        ],
    },
    {
        "id": "jbl_clip_4",
        "display": "JBL Clip 4 / Clip 3",
        "description": (
            "Small speaker with limited LF extension. Mild upper-bass boost, "
            "recessed mids at 2 kHz."
        ),
        "source": "Rtings.com",
        "confidence": 0.82,
        "bands": [
            (250,  -2.0, 0.90),
            (500,  -1.0, 1.00),
            (2000,  1.5, 1.20),
        ],
    },
    {
        "id": "jbl_partybox_310",
        "display": "JBL PartyBox 310",
        "description": "Party speaker with heavy bass emphasis and elevated presence.",
        "source": "Measurements / community data",
        "confidence": 0.78,
        "bands": [
            (80,   -5.0, 0.70),
            (250,  -2.0, 1.00),
            (2000,  3.0, 1.20),
            (8000, -1.5, 1.00),
        ],
    },

    # -------------------------------------------------------------------
    # Bose
    # -------------------------------------------------------------------
    {
        "id": "bose_qc45",
        "display": "Bose QuietComfort 45",
        "description": (
            "Modest bass lift (+3–4 dB at 80–200 Hz), neutral mids, smooth treble."
        ),
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -2.5, 0.80),
            (2000,  1.0, 1.30),
        ],
    },
    {
        "id": "bose_nc700",
        "display": "Bose Noise Cancelling 700",
        "description": (
            "Moderate bass boost (+5–6 dB at 60–150 Hz), recessed low-mids at "
            "250 Hz, good overall balance."
        ),
        "source": "Rtings.com",
        "confidence": 0.86,
        "bands": [
            (80,   -3.5, 0.80),
            (250,  -1.5, 1.00),
            (8000,  1.0, 1.00),
        ],
    },
    {
        "id": "bose_qc_earbuds2",
        "display": "Bose QuietComfort Earbuds II",
        "description": "Well-balanced IEM with slight bass lift and mild presence peak.",
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -2.0, 0.80),
            (4000, -1.0, 1.50),
        ],
    },
    {
        "id": "bose_soundlink_flex",
        "display": "Bose SoundLink Flex",
        "description": (
            "Moderate bass boost (+5–7 dB at 70–200 Hz), slightly recessed mids."
        ),
        "source": "Rtings.com",
        "confidence": 0.86,
        "bands": [
            (80,   -3.5, 0.80),
            (250,  -1.5, 1.00),
            (2000,  2.0, 1.20),
        ],
    },
    {
        "id": "bose_soundlink_mini2",
        "display": "Bose SoundLink Mini II",
        "description": "Warm, bass-forward sound signature; mild mid recession.",
        "source": "Community measurements",
        "confidence": 0.80,
        "bands": [
            (80,   -3.0, 0.80),
            (2000,  2.0, 1.20),
        ],
    },

    # -------------------------------------------------------------------
    # Samsung
    # -------------------------------------------------------------------
    {
        "id": "galaxy_buds2_pro",
        "display": "Samsung Galaxy Buds2 Pro",
        "description": (
            "Slight V-shape with modest bass lift (+3–4 dB), mild dip at "
            "300–500 Hz, presence peak at 3–4 kHz."
        ),
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -2.0, 0.80),
            (500,   1.0, 1.20),
            (4000, -1.5, 1.50),
        ],
    },
    {
        "id": "galaxy_buds3_pro",
        "display": "Samsung Galaxy Buds3 Pro",
        "description": (
            "Well-tuned, close to Harman target. Slight bass lift, mild presence peak."
        ),
        "source": "Rtings.com",
        "confidence": 0.88,
        "bands": [
            (80,   -1.5, 0.80),
            (4000, -1.0, 1.50),
        ],
    },
    {
        "id": "galaxy_buds2",
        "display": "Samsung Galaxy Buds2",
        "description": "Mild V-shape IEM, slightly more bass-forward than Buds2 Pro.",
        "source": "Rtings.com",
        "confidence": 0.83,
        "bands": [
            (80,   -2.5, 0.80),
            (2000,  1.5, 1.50),
            (8000,  1.0, 1.00),
        ],
    },

    # -------------------------------------------------------------------
    # Sennheiser
    # -------------------------------------------------------------------
    {
        "id": "sennheiser_momentum4",
        "display": "Sennheiser Momentum 4 Wireless",
        "description": "Good overall balance with slight V-shape and mild bass lift.",
        "source": "Rtings.com",
        "confidence": 0.86,
        "bands": [
            (80,   -2.5, 0.80),
            (4000,  1.0, 1.30),
            (8000,  1.0, 1.00),
        ],
    },
    {
        "id": "sennheiser_hd600",
        "display": "Sennheiser HD 600 / HD 650",
        "description": (
            "Reference-class open-back. Near-flat response with mild sub-bass "
            "rolloff below 50 Hz. Tiny 3 kHz peak."
        ),
        "source": "InnerFidelity / Rtings.com",
        "confidence": 0.90,
        "bands": [
            (80,    1.5, 0.70),   # lift slight sub-bass rolloff
            (4000, -0.5, 2.00),   # smooth tiny presence peak
        ],
    },
    {
        "id": "sennheiser_hd560s",
        "display": "Sennheiser HD 560S",
        "description": (
            "Neutral with slight bass rolloff; bright at 5–6 kHz (+3 dB)."
        ),
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,    1.0, 0.80),
            (5000, -2.5, 1.50),
        ],
    },

    # -------------------------------------------------------------------
    # Audio-Technica
    # -------------------------------------------------------------------
    {
        "id": "ath_m50x",
        "display": "Audio-Technica ATH-M50x",
        "description": (
            "V-shaped studio monitor: bass +4–6 dB at 50–120 Hz, "
            "mid recession at 500–2000 Hz, elevated treble."
        ),
        "source": "InnerFidelity / Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -3.5, 0.80),
            (1000,  2.5, 1.30),
            (8000,  1.5, 1.00),
        ],
    },
    {
        "id": "ath_m40x",
        "display": "Audio-Technica ATH-M40x",
        "description": "More balanced than M50x, mild bass lift, slightly bright treble.",
        "source": "InnerFidelity",
        "confidence": 0.84,
        "bands": [
            (80,   -1.5, 0.80),
            (1000,  1.5, 1.30),
            (8000,  1.0, 1.00),
        ],
    },

    # -------------------------------------------------------------------
    # Jabra
    # -------------------------------------------------------------------
    {
        "id": "jabra_evolve2_85",
        "display": "Jabra Evolve2 85",
        "description": (
            "Business headset tuned for speech clarity. Near-flat with slight "
            "presence boost at 2–4 kHz. Minimal correction needed."
        ),
        "source": "Rtings.com",
        "confidence": 0.85,
        "bands": [
            (2000, -1.0, 1.50),
        ],
    },
    {
        "id": "jabra_elite_85h",
        "display": "Jabra Elite 85h",
        "description": "Consumer ANC headphone: moderate bass lift, relatively neutral mids.",
        "source": "Rtings.com",
        "confidence": 0.82,
        "bands": [
            (80,   -2.5, 0.80),
            (4000,  1.0, 1.30),
        ],
    },

    # -------------------------------------------------------------------
    # Marshall
    # -------------------------------------------------------------------
    {
        "id": "marshall_emberton_2",
        "display": "Marshall Emberton II",
        "description": (
            "Retro V-signature: bass +4–6 dB, recessed mids, elevated treble "
            "at 6–10 kHz."
        ),
        "source": "Community measurements",
        "confidence": 0.80,
        "bands": [
            (80,   -3.5, 0.80),
            (1000,  2.5, 1.30),
            (8000, -1.5, 1.00),
        ],
    },
    {
        "id": "marshall_stanmore_3",
        "display": "Marshall Stanmore III / Woburn III",
        "description": "Warm vintage-inspired sound: bass emphasis, recessed upper mids.",
        "source": "Community measurements",
        "confidence": 0.75,
        "bands": [
            (80,   -3.0, 0.80),
            (2000,  2.0, 1.30),
            (8000, -1.0, 1.00),
        ],
    },

    # -------------------------------------------------------------------
    # Ultimate Ears
    # -------------------------------------------------------------------
    {
        "id": "ue_wonderboom_3",
        "display": "Ultimate Ears WONDERBOOM 3",
        "description": (
            "Heavy bass boost (+6–8 dB at 60–150 Hz), strongly recessed mids."
        ),
        "source": "Rtings.com",
        "confidence": 0.86,
        "bands": [
            (80,   -5.0, 0.70),
            (250,  -2.0, 1.00),
            (2000,  3.5, 1.20),
        ],
    },
    {
        "id": "ue_megaboom_3",
        "display": "Ultimate Ears MEGABOOM 3",
        "description": "Bass-forward portable: heavy emphasis, recessed mids, decent treble.",
        "source": "Rtings.com",
        "confidence": 0.85,
        "bands": [
            (80,   -4.5, 0.70),
            (250,  -1.5, 1.00),
            (2000,  3.0, 1.20),
        ],
    },
    {
        "id": "ue_hyperboom",
        "display": "Ultimate Ears HYPERBOOM",
        "description": "Large party speaker with very heavy bass, strong mid recession.",
        "source": "Rtings.com",
        "confidence": 0.84,
        "bands": [
            (80,   -5.5, 0.70),
            (250,  -2.5, 1.00),
            (2000,  4.0, 1.20),
        ],
    },

    # -------------------------------------------------------------------
    # Smart speakers
    # -------------------------------------------------------------------
    {
        "id": "apple_homepod_2",
        "display": "Apple HomePod (2nd gen)",
        "description": (
            "Well-tuned with internal DSP (Spatial Audio). Slight bass warmth, "
            "minimal correction needed."
        ),
        "source": "Apple / community measurements",
        "confidence": 0.80,
        "bands": [
            (80,   -1.5, 0.80),
            (8000, -0.5, 1.00),
        ],
    },
    {
        "id": "amazon_echo_studio",
        "display": "Amazon Echo Studio",
        "description": "Well-balanced smart speaker; mild warmth below 200 Hz.",
        "source": "Community measurements",
        "confidence": 0.78,
        "bands": [
            (80,   -1.0, 0.80),
            (4000, -0.5, 1.30),
        ],
    },
    {
        "id": "sonos_era_100",
        "display": "Sonos Era 100",
        "description": "Balanced output with adaptive Trueplay tuning. Mild bass warmth.",
        "source": "Community measurements",
        "confidence": 0.78,
        "bands": [
            (80,   -1.0, 0.80),
        ],
    },

    # -------------------------------------------------------------------
    # Anker / Soundcore
    # -------------------------------------------------------------------
    {
        "id": "anker_q45",
        "display": "Anker Soundcore Q45",
        "description": "Mild V-shape: modest bass lift, slightly recessed mids.",
        "source": "Rtings.com",
        "confidence": 0.82,
        "bands": [
            (80,   -2.5, 0.80),
            (2000,  1.5, 1.30),
        ],
    },
    {
        "id": "anker_q35",
        "display": "Anker Soundcore Q35",
        "description": "Budget ANC headphone: bass-forward, slightly recessed mids.",
        "source": "Community measurements",
        "confidence": 0.75,
        "bands": [
            (80,   -3.0, 0.80),
            (2000,  2.0, 1.30),
        ],
    },

    # -------------------------------------------------------------------
    # Beyerdynamic
    # -------------------------------------------------------------------
    {
        "id": "beyerdynamic_dt990",
        "display": "Beyerdynamic DT 990 Pro",
        "description": (
            "Bright V-shape: bass +4 dB, dip at 1 kHz, strong treble peak at 8–10 kHz "
            "(+5–7 dB). Significant treble reduction needed."
        ),
        "source": "InnerFidelity / Rtings.com",
        "confidence": 0.88,
        "bands": [
            (80,   -2.5, 0.80),
            (1000,  1.5, 1.50),
            (8000, -5.0, 1.20),
        ],
    },
    {
        "id": "beyerdynamic_dt770",
        "display": "Beyerdynamic DT 770 Pro",
        "description": "Closed-back: elevated bass, neutral mids, bright treble peak at 8 kHz.",
        "source": "InnerFidelity",
        "confidence": 0.87,
        "bands": [
            (80,   -3.0, 0.80),
            (8000, -3.5, 1.20),
        ],
    },
    {
        "id": "beyerdynamic_dt700_pro_x",
        "display": "Beyerdynamic DT 700 Pro X",
        "description": "Modern, relatively neutral closed-back with mild bass lift and smooth treble.",
        "source": "Rtings.com",
        "confidence": 0.87,
        "bands": [
            (80,   -1.5, 0.80),
            (4000, -0.5, 2.00),
        ],
    },
]

# Build O(1) lookup by ID
_MEASUREMENT_DB_BY_ID: dict[str, dict] = {e["id"]: e for e in MEASUREMENT_DB}


# ---------------------------------------------------------------------------
# Tier 1: Model-specific keyword patterns → MEASUREMENT_DB ID
# Checked BEFORE DEVICE_NAME_PATTERNS.  Order: most specific first.
# ---------------------------------------------------------------------------

MODEL_PATTERNS: list[tuple[list[str], str]] = [
    # Beats Pill
    (["beats pill 2024", "beats pill 2023", "beats pill+",
      "beats pill plus"],                               "beats_pill_2023"),
    (["beats pill"],                                    "beats_pill_2023"),
    # Beats headphones (specific models)
    (["beats studio pro"],                              "beats_studio_pro_2023"),
    (["beats studio3", "beats studio 3"],               "beats_studio3"),
    (["beats solo4", "beats solo 4"],                   "beats_solo4"),
    (["beats solo3", "beats solo 3"],                   "beats_solo4"),  # same tuning
    (["beats fit pro"],                                 "beats_fit_pro"),
    (["powerbeats pro", "beats powerbeats pro"],        "beats_powerbeats_pro"),
    # AirPods
    (["airpods pro (2nd", "airpods pro 2",
      "airpods pro2", "airpods pro - 2nd"],             "airpods_pro_2"),
    (["airpods pro (1st", "airpods pro 1",
      "airpods pro1", "airpods pro - 1st"],             "airpods_pro_1"),
    (["airpods pro"],                                   "airpods_pro_2"),  # assume latest
    (["airpods (3rd", "airpods 3rd", "airpods3"],       "airpods_3"),
    (["airpods (2nd", "airpods 2nd", "airpods2"],       "airpods_2"),
    (["airpods max"],                                   "airpods_max"),
    (["airpods"],                                       "airpods_2"),  # generic fallback
    # Sony WH headphones
    (["wh-1000xm5", "wh1000xm5"],                      "sony_xm5_headphone"),
    (["wh-1000xm4", "wh1000xm4"],                      "sony_xm4_headphone"),
    (["wh-1000xm3", "wh1000xm3"],                      "sony_xm4_headphone"),  # similar
    # Sony WF earbuds
    (["wf-1000xm5", "wf1000xm5"],                      "sony_xm5_earbud"),
    (["wf-1000xm4", "wf1000xm4"],                      "sony_xm4_earbud"),
    (["wf-1000xm3", "wf1000xm3"],                      "sony_xm4_earbud"),
    # Sony SRS Extra Bass
    (["srs-xb43", "srs xb43"],                         "sony_srs_xb43"),
    (["srs-xb33", "srs xb33"],                         "sony_srs_xb33"),
    (["extra bass"],                                    "sony_srs_xb33"),
    # JBL portables
    (["jbl boombox 3", "jbl boombox3"],                "jbl_boombox_3"),
    (["jbl boombox 2", "jbl boombox2"],                "jbl_boombox_3"),
    (["jbl partybox 310", "jbl partybox310"],          "jbl_partybox_310"),
    (["jbl xtreme 3", "jbl xtreme3"],                  "jbl_xtreme_3"),
    (["jbl xtreme 2", "jbl xtreme2"],                  "jbl_xtreme_3"),
    (["jbl charge 5", "jbl charge5"],                  "jbl_charge_5"),
    (["jbl charge 4", "jbl charge4"],                  "jbl_charge_5"),  # similar
    (["jbl flip 6", "jbl flip6"],                      "jbl_flip_6"),
    (["jbl flip 5", "jbl flip5"],                      "jbl_flip_6"),
    (["jbl clip 4", "jbl clip4"],                      "jbl_clip_4"),
    (["jbl clip 3", "jbl clip3"],                      "jbl_clip_4"),
    # Bose headphones
    (["quietcomfort 45", "qc45", "qc 45"],             "bose_qc45"),
    (["quietcomfort 35", "qc35", "qc 35"],             "bose_qc45"),  # similar
    (["noise cancelling 700", "nc 700", "nc700"],      "bose_nc700"),
    (["quietcomfort earbuds ii", "qc earbuds ii",
      "quietcomfort earbuds 2"],                        "bose_qc_earbuds2"),
    (["bose soundlink flex"],                          "bose_soundlink_flex"),
    (["bose soundlink mini"],                          "bose_soundlink_mini2"),
    # Samsung Galaxy Buds
    (["galaxy buds3 pro", "galaxy buds 3 pro"],        "galaxy_buds3_pro"),
    (["galaxy buds2 pro", "galaxy buds 2 pro"],        "galaxy_buds2_pro"),
    (["galaxy buds2", "galaxy buds 2"],                "galaxy_buds2"),
    # Sennheiser
    (["momentum 4"],                                   "sennheiser_momentum4"),
    (["hd 600", "hd600"],                              "sennheiser_hd600"),
    (["hd 650", "hd650"],                              "sennheiser_hd600"),
    (["hd 660", "hd660"],                              "sennheiser_hd600"),
    (["hd 560s", "hd560s"],                            "sennheiser_hd560s"),
    # Audio-Technica
    (["ath-m50x", "m50x"],                             "ath_m50x"),
    (["ath-m40x", "m40x"],                             "ath_m40x"),
    # Jabra
    (["evolve2 85", "evolve 2 85", "evolve2-85"],      "jabra_evolve2_85"),
    (["elite 85h"],                                    "jabra_elite_85h"),
    # Marshall
    (["marshall emberton"],                            "marshall_emberton_2"),
    (["marshall stanmore", "marshall woburn"],         "marshall_stanmore_3"),
    # Ultimate Ears
    (["hyperboom"],                                    "ue_hyperboom"),
    (["megaboom 3", "megaboom3"],                      "ue_megaboom_3"),
    (["wonderboom 3", "wonderboom3"],                  "ue_wonderboom_3"),
    (["wonderboom"],                                   "ue_wonderboom_3"),
    # Smart speakers
    (["homepod"],                                      "apple_homepod_2"),
    (["echo studio"],                                  "amazon_echo_studio"),
    (["sonos era"],                                    "sonos_era_100"),
    # Anker Soundcore
    (["soundcore q45"],                                "anker_q45"),
    (["soundcore q35"],                                "anker_q35"),
    # Beyerdynamic
    (["dt 990", "dt990"],                              "beyerdynamic_dt990"),
    (["dt 770", "dt770"],                              "beyerdynamic_dt770"),
    (["dt 700 pro x", "dt700 pro x"],                  "beyerdynamic_dt700_pro_x"),
    (["dt 900 pro x", "dt900 pro x"],                  "beyerdynamic_dt700_pro_x"),  # similar
]


# ===========================================================================
# Tier 2 — Category-level fallback profiles
# ===========================================================================

PROFILES: dict[str, dict] = {
    "reference": {
        "display": "Reference / Studio Monitor",
        "description": "Flat response — no EQ correction applied.",
        "bands": [],
    },
    "headphone_consumer": {
        "display": "Consumer Over-Ear Headphones",
        "description": "Slight Harman-style correction: reduce bass bloom, "
                       "lift upper-mids and air.",
        "bands": [
            (60,    -1.5, 0.8),
            (3000,   2.0, 1.5),
            (10000,  2.5, 1.0),
        ],
    },
    "beats_style": {
        "display": "Bass-Forward Portable (Beats, etc.)",
        "description": "Heavy bass reduction, boosted midrange clarity, "
                       "lifted treble — corrects 'V-shape' coloration.",
        "bands": [
            (80,   -5.0, 0.7),
            (200,  -1.0, 1.0),
            (2000,  2.5, 1.2),
            (6000,  2.0, 1.0),
        ],
    },
    "jbl_balanced": {
        "display": "Balanced Portable Bluetooth Speaker",
        "description": "Moderate bass reduction, presence and treble lift.",
        "bands": [
            (80,   -2.0, 0.8),
            (3000,  1.5, 1.2),
            (8000,  1.5, 1.0),
        ],
    },
    "bluetooth_portable_small": {
        "display": "Small Portable Bluetooth Speaker",
        "description": "Limited LF capability — lift upper-bass for warmth, "
                       "boost presence.",
        "bands": [
            (150,  2.5, 0.8),
            (250,  1.0, 1.0),
            (3000, 1.5, 1.2),
        ],
    },
    "laptop_speaker": {
        "display": "Laptop / Tablet Built-in Speaker",
        "description": "Very limited LF — lift upper-bass and low-mid for body, "
                       "modest presence boost.",
        "bands": [
            (200,  3.0, 0.8),
            (400,  1.5, 1.0),
            (3000, 1.0, 1.2),
        ],
    },
    "soundbar": {
        "display": "TV Soundbar",
        "description": "Boost upper-bass for warmth, lift presence for "
                       "speech clarity.",
        "bands": [
            (100,  1.5, 0.8),
            (200,  1.0, 1.0),
            (2500, 2.0, 1.2),
        ],
    },
    "iem_earbuds": {
        "display": "IEM / Earbuds",
        "description": "Typical V-shape correction: mild bass and low-mid cut, "
                       "midrange lift, air boost.",
        "bands": [
            (80,   -1.5, 0.8),
            (500,  -1.0, 1.0),
            (3000,  2.5, 1.5),
            (10000, 1.5, 1.0),
        ],
    },
    "phone_speaker": {
        "display": "Smartphone Speaker",
        "description": "Very limited LF — lift low-mid and presence for body "
                       "and speech clarity.",
        "bands": [
            (300,  2.5, 0.9),
            (600,  1.5, 1.0),
            (3000, 1.5, 1.2),
        ],
    },
    "smart_speaker": {
        "display": "Smart Speaker (HomePod, Sonos, Echo, etc.)",
        "description": "Internal DSP usually well-tuned — minor presence lift.",
        "bands": [
            (3000, 1.0, 1.2),
            (10000, 1.0, 1.0),
        ],
    },
    "subwoofer": {
        "display": "Subwoofer / Bass Extension Speaker",
        "description": "Boost deep sub-bass (sub only, no HF correction).",
        "bands": [
            (40, 3.0, 0.8),
            (80, 2.0, 0.8),
        ],
    },
}


# ---------------------------------------------------------------------------
# Tier 2: Name → category profile (checked after MODEL_PATTERNS)
# ---------------------------------------------------------------------------

DEVICE_NAME_PATTERNS: list[tuple[list[str], str]] = [
    # Specific Beats models (not in MEASUREMENT_DB)
    (["beats studio", "beats solo", "beats pro",
      "beats flex", "beats wireless", "beatsx"],        "headphone_consumer"),
    (["beats pill", "beats box", "beatbox"],            "beats_style"),
    (["beats"],                                         "beats_style"),
    # Apple
    (["airpods"],                                       "iem_earbuds"),
    (["apple homepod", "homepod"],                      "smart_speaker"),
    # Samsung
    (["galaxy buds", "galaxy bud"],                     "iem_earbuds"),
    # Jabra
    (["jabra evolve", "jabra engage", "jabra elite 85h",
      "jabra evolve2"],                                  "headphone_consumer"),
    (["jabra"],                                         "iem_earbuds"),
    # Bose
    (["bose quietcomfort", "bose 700", "bose nc700",
      "bose qc ", "bose qc4", "bose qc3"],               "headphone_consumer"),
    (["bose soundlink revolve", "bose soundlink mini",
      "bose soundlink flex", "bose portable home"],      "jbl_balanced"),
    (["bose soundbar", "bose tv speaker",
      "bose lifestyle"],                                  "soundbar"),
    # JBL
    (["jbl charge", "jbl flip", "jbl link", "jbl partybox",
      "jbl pulse", "jbl xtreme", "jbl boombox"],         "jbl_balanced"),
    (["jbl clip", "jbl go", "jbl wind"],                 "bluetooth_portable_small"),
    (["jbl tour", "jbl free", "jbl tune",
      "jbl live", "jbl reflect", "jbl endurance"],       "iem_earbuds"),
    (["jbl"],                                            "jbl_balanced"),
    # Sony
    (["sony wh-", "sony wf-", "sony xm", "sony 1000x"],  "headphone_consumer"),
    (["sony srs", "sony extra bass"],                    "jbl_balanced"),
    # Smart speakers
    (["amazon echo", "echo dot", "echo studio",
      "echo show", "alexa"],                              "smart_speaker"),
    (["google home", "google nest", "nest audio"],       "smart_speaker"),
    (["sonos"],                                          "smart_speaker"),
    # Other headphone brands
    (["marshall headphone", "marshall major",
      "marshall monitor", "marshall mid"],               "headphone_consumer"),
    (["marshall"],                                       "jbl_balanced"),
    (["ultimate ears", "ue boom", "ue roll",
      "ue wonderboom"],                                   "jbl_balanced"),
    (["anker soundcore"],                                "jbl_balanced"),
    (["harman kardon"],                                  "jbl_balanced"),
    # Reference monitors
    (["kef", "focal", "dynaudio", "adam audio",
      "yamaha hs", "genelec", "neumann kh",
      "audioengine", "klipsch r-", "klipsch rp-"],       "reference"),
    # Sennheiser / Audio-Technica / Beyerdynamic
    (["sennheiser hd ", "sennheiser he",
      "sennheiser momentum"],                            "headphone_consumer"),
    (["audio-technica", "audiotechnica"],                "headphone_consumer"),
    (["beyerdynamic", "beyer dynamic"],                  "headphone_consumer"),
    # Generic keywords — must come after specific brands
    (["soundbar", "sound bar", "tv speaker",
      "home theater speaker"],                           "soundbar"),
    (["subwoofer", "sub woofer", "woofer"],              "subwoofer"),
    (["iphone", "android phone", "pixel phone"],        "phone_speaker"),
    (["ipad", "tablet speaker"],                        "laptop_speaker"),
    (["macbook", "macbook pro", "macbook air",
      "laptop speaker", "notebook speaker"],             "laptop_speaker"),
    (["built-in output", "built-in microphone",
      "built in speaker", "internal speaker",
      "realtek", "conexant", "cirrus logic"],            "laptop_speaker"),
    (["studio monitor", "active monitor",
      "powered monitor"],                                "reference"),
    (["over-ear headphone", "on-ear headphone",
      "wireless headphone", "headphone"],                "headphone_consumer"),
    (["in-ear monitor", "iem", "in-ear"],               "iem_earbuds"),
    (["earbud", "true wireless", "tws"],                "iem_earbuds"),
    (["bluetooth speaker", "bt speaker"],               "jbl_balanced"),
]


# ===========================================================================
# Lookup functions
# ===========================================================================

def detect_speaker_profile(device_name: str) -> tuple[str, dict]:
    """
    Match a device name to the best known speaker profile (Tier 2 fallback).

    Parameters
    ----------
    device_name : str  — audio device name from sounddevice

    Returns
    -------
    (profile_key, profile_dict)
    """
    name_lc = device_name.lower()
    for keywords, profile_key in DEVICE_NAME_PATTERNS:
        if any(kw in name_lc for kw in keywords):
            return profile_key, PROFILES[profile_key]
    return "reference", PROFILES["reference"]


def get_correction_bands(device_name: str) -> tuple[str, list, float, str, str]:
    """
    Look up device-specific correction bands.

    Checks MODEL_PATTERNS (Tier 1, model-specific) before DEVICE_NAME_PATTERNS
    (Tier 2, category-level).

    Returns
    -------
    (id_or_key, bands, confidence, display_name, description)
        id_or_key   : model ID (from MEASUREMENT_DB) or profile key (from PROFILES)
        bands       : list of (freq_hz, gain_db, q) correction tuples
        confidence  : 0.0–1.0
        display_name: human-readable device/profile name
        description : correction rationale
    """
    name_lc = device_name.lower()

    # --- Tier 1: specific model ---
    for keywords, model_id in MODEL_PATTERNS:
        if any(kw in name_lc for kw in keywords):
            entry = _MEASUREMENT_DB_BY_ID.get(model_id)
            if entry:
                return (
                    model_id,
                    entry["bands"],
                    entry["confidence"],
                    entry["display"],
                    entry["description"],
                )

    # --- Tier 2: category profile ---
    profile_key, profile = detect_speaker_profile(device_name)
    confidence = 0.55 if profile_key != "reference" else 0.30
    return (
        profile_key,
        profile["bands"],
        confidence,
        profile["display"],
        profile["description"],
    )


def _bands_to_gains(bands: list, band_freqs: list) -> list[float]:
    """
    Map a list of (freq_hz, gain_db, q) correction bands to EQ band gains
    using log-nearest-neighbour assignment.

    When multiple correction bands map to the same EQ band their gains are
    *averaged* (not summed) to prevent over-correction from stacked profiles.

    Parameters
    ----------
    bands      : list of (freq_hz, gain_db, q) tuples
    band_freqs : list of EQ band centre frequencies

    Returns
    -------
    list[float] — gain_db for each EQ band
    """
    gains  = [0.0] * len(band_freqs)
    counts = [0]   * len(band_freqs)

    if not bands:
        return gains

    for t_freq, t_gain, _q in bands:
        best_idx  = 0
        best_dist = float("inf")
        for j, bfreq in enumerate(band_freqs):
            dist = abs(math.log(max(t_freq, 1.0)) - math.log(max(bfreq, 1.0)))
            if dist < best_dist:
                best_dist = dist
                best_idx  = j
        gains[best_idx]  += t_gain
        counts[best_idx] += 1

    # Average when multiple bands collide on the same EQ slot
    for j in range(len(band_freqs)):
        if counts[j] > 1:
            gains[j] /= counts[j]

    return gains


def profile_to_band_gains(profile_key: str, band_freqs: list) -> list:
    """
    Map a category profile's correction bands to the ParametricEQ band
    frequencies (legacy interface, preserves summing behaviour).

    Parameters
    ----------
    profile_key : str         — key into PROFILES
    band_freqs  : list[float] — list of EQ band centre frequencies

    Returns
    -------
    list[float] — gain_db for each EQ band
    """
    profile      = PROFILES.get(profile_key, PROFILES["reference"])
    target_bands = profile["bands"]

    gains = [0.0] * len(band_freqs)
    if not target_bands:
        return gains

    for t_freq, t_gain, _t_q in target_bands:
        best_idx  = 0
        best_dist = float("inf")
        for j, bfreq in enumerate(band_freqs):
            dist = abs(math.log(max(t_freq, 1.0)) - math.log(max(bfreq, 1.0)))
            if dist < best_dist:
                best_dist = dist
                best_idx  = j
        gains[best_idx] += t_gain

    return gains

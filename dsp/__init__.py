"""ModAudio DSP sub-package."""
from .theater_chain    import TheaterChain
from .custom_eq        import ParametricEQ, DEFAULT_BANDS
from .speaker_profiles import (
    detect_speaker_profile, profile_to_band_gains, PROFILES,
    get_correction_bands, _bands_to_gains,
)
from .device_analyzer  import DeviceAnalyzer, AnalysisResult

__all__ = [
    "TheaterChain",
    "ParametricEQ",
    "DEFAULT_BANDS",
    "detect_speaker_profile",
    "profile_to_band_gains",
    "get_correction_bands",
    "_bands_to_gains",
    "PROFILES",
    "DeviceAnalyzer",
    "AnalysisResult",
]

# ModAudio - Theater Experience for Your Speakers & Headphones

Transform your audio playback into an immersive theater experience with **ModAudio**, a professional-grade DSP (Digital Signal Processing) application that brings cinema-quality surround sound to any speaker setup or headphones.

## Features

### 🎬 Immersive Theater Processing
- **Binaural HRTF rendering** for headphones with accurate spatial audio using the Brown-Duda head-related transfer function
- **Advanced room simulation** modeling large commercial cinema acoustics with configurable reverberation and early reflections
- **Multi-band dynamics** with theatrical compression and transient enhancement for punchy, articulate sound
- **Cinema EQ (X-curve)** with psychoacoustic bass extension and presence boost

### 🎤 Dual Audio Mode Support

#### Single Speaker Mode
- Stereo playback through headphones or stereo speakers
- Full theater DSP chain with binaural spatialization
- Customizable reverb, width, bass, and dynamics
- 4 theater presets (Cinema, IMAX, Dolby, Home)

#### Multi-Speaker Mode (Two Physical Speakers)
- **VB-Cable integration** — automatically captures audio to route through both speakers
- **VBAP routing** (Vector Base Amplitude Panning) maps 7 virtual surround channels to 2 physical speakers
- **True surround sound** that moves around your listening position (front → side → rear → side → front)
- **Customizable rear speaker placement** — adjust azimuth angle (90°–170°) to match your room layout
- **Speaker orientation awareness** — "Faces Me" or "Faces Away" settings adapt surround routing to your setup
- **Acoustic delay calibration** — time-align speakers for coherent wavefront merging
- **3D spatial modeling** — simulates how two speakers interact to create a sound field that envelops you
- **Behind spectral coloring** — psychoacoustic processing helps you localize the rear speaker as coming from behind

### 🎚️ Theater Presets
- **Cinema** — Standard commercial cinema sound (RT60 1.3s, 6 dB bass boost)
- **IMAX** — Massive room, overwhelming scale, deep bass impact (RT60 1.9s, 10 dB bass boost)
- **Dolby** — Precision-tuned, tighter room, clean dialog intelligibility (RT60 0.95s, 4 dB bass boost)
- **Home** — Subtle theater enhancement for residential listening (RT60 0.8s, 3 dB bass boost)

### 🎛️ Real-Time Control
- Interactive sliders for **Reverb**, **Stereo Width**, **Bass**, and **Dynamics**
- Per-channel level adjustment for surround, center, and back channels
- Bass priority mode selection ("Equal", "Front", "Rear")
- Theater preset selection with one click

---

## System Requirements

### Windows 10/11 (64-bit)
- **Python 3.8+** (we recommend 3.11 or later)
- **~200 MB** disk space for dependencies
- **Audio device** (headphones, speakers, or USB audio interface)

### Optional: Multi-Speaker Support
- **VB-Cable** (see setup instructions below)
- Two physical speakers positioned in your listening environment
- Audio interface or multi-channel capable soundcard (optional, for separate speaker routing)

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ModAudio.git
cd ModAudio
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` — numerical computation
- `scipy` — signal processing (filters, FFT)
- `sounddevice` — audio I/O
- `customtkinter` — modern GUI framework
- `pyaudiowpatch` — loopback audio capture (Windows only)
- `comtypes` — COM interface for audio device control (Windows only)

### Step 3: Run the Application
```bash
python app.py
```

The ModAudio GUI window will open. You're ready to use single-speaker mode immediately.

---

## Quick Start Guide

### Single Speaker Mode (Headphones or Stereo Speakers)

1. **Launch the app** — `python app.py`
2. **Select your audio device** in the "Device" dropdown
3. **Choose a theater preset** — Cinema (default), IMAX, Dolby, or Home
4. **Adjust sliders** as desired:
   - **Reverb** — Room acoustics (0.3–2.5 seconds)
   - **Width** — Stereo field expansion (1.0–2.8×)
   - **Bass** — Sub-bass boost (0–12 dB)
   - **Dynamics** — Compression punch (1.0–2.2)
5. **Click "Start"** to begin processing
6. **Click "Stop"** to pause or switch devices

**Single-speaker audio flow:**
```
Input Audio → Bass Enhancement → EQ → Reverb → Dynamics →
Transient Enhancement → Binaural Rendering → Peak Limiting → Output
```

For **headphones**, you'll hear:
- Immersive 5.1 virtual surround positioning via HRTF spatialization
- Rear channels rendered as diffuse surround field with psychoacoustic cues
- Full theater acoustics and bass enhancement

For **stereo speakers**, you'll hear:
- Stereo image expansion via M/S widening and Haas-delay effects
- Subtle surround illusion from time-domain processing
- Warm, theatrical bass with compression-driven punch

---

## Multi-Speaker Mode Setup

### Prerequisites

1. **Install VB-Cable** (free, by VB-Audio)
   - **Download:** https://vb-audio.com/Cable/
   - **Install** and restart your computer
   - VB-Cable will appear as a playback device in Windows Sound settings

2. **Two physical speakers** placed in your room:
   - **Front speaker** — facing your listening position (typical setup)
   - **Rear speaker** — positioned behind or to the side of your listening area
   - Any speaker type (bookshelf, studio monitors, powered, passive, etc.)

### Connecting Speakers

**Option A: Loopback Mode (Simplest)**
- Front speaker: connected to your system's main audio output (speakers or amp)
- Rear speaker: connected via VB-Cable virtual loopback
- ModAudio auto-routes: front plays clean audio, rear gets surround-enhanced signal

**Option B: Full Control (Best Fidelity)**
- Both speakers: routed through a **2-channel USB audio interface** or **multichannel soundcard**
- Requires administrator privileges to install the virtual audio driver
- ModAudio manages both speaker outputs independently, each with full DSP processing

### Configuration in ModAudio

1. **Switch to "Multi-Speaker" tab**
2. **THEATER MODE** — Select a preset:
   - **Cinema** — balanced theater sound, 6 dB bass boost
   - **IMAX** — overwhelming scale, 10 dB bass boost, massive reverb (RT60 1.9s)
   - **Dolby** — precision sound, tight room, excellent dialog (RT60 0.95s)
   - **Home** — subtle enhancement for residential listening
3. **SPEAKER PLACEMENT** — Configure your setup:
   - **Orientation** — Select "Faces Me" (both speakers point toward you) or "Faces Away" (rear faces away)
     - *Different orientations change surround routing to match speaker direction*
   - **Rear Position** — Azimuth slider (90°–170°)
     - 90° = side-mounted (lateral)
     - 150° = directly behind (typical)
     - 170° = deep rear
     - *Adjusting this changes how virtual surround channels map to your physical speakers*
   - **Speaker Acoustic Delay** — Time alignment (0–20 ms)
     - *Compensates for physical distance between front and rear speakers*
     - Use ~3 ms per meter of distance

4. **DEVICE SETUP** — Choose audio devices:
   - **Front Speaker** — your primary playback device
   - **Rear Speaker** — VB-Cable or second audio output

5. **Click "Start"** — ProcessIng begins on both buses simultaneously

### Multi-Speaker Audio Flow

```
Input Audio → Bass Enhancement → EQ → Reverb → Dynamics →
Transient Enhancement → Master Gain →
     ↓ (split into sub-bass and mid/high)
     ├─ SUB-BASS: Routed to both front and rear (mono, priority-adjustable)
     └─ MID/HIGH: 7-channel adaptive upmix →
          FL, FR, C, LS, RS, LB, RB →
          VBAP routing matrix →
          [front_L, front_R] → Front Speaker Bus
          [rear_L, rear_R]  → Rear Speaker Bus
               ↓
          Spectral coloring (rear bus)
               ↓
          Per-bus peak limiting
               ↓
          Output
```

### Understanding VBAP Routing

**Vector Base Amplitude Panning** distributes 7 virtual cinema channels to your 2 physical speakers using constant-power mathematics. For example, with rear azimuth = 150° (directly behind):

| Virtual Channel | Front-Left | Front-Right | Rear-Left | Rear-Right |
|---|---|---|---|---|
| FL (−30°, front-left) | 100% | 0% | 0% | 0% |
| FR (+30°, front-right) | 0% | 100% | 0% | 0% |
| C (0°, center) | 71% | 71% | 0% | 0% |
| LS (−110°, left surround) | 50% | 0% | 87% | 0% |
| RS (+110°, right surround) | 0% | 50% | 0% | 87% |
| LB (−150°, left back) | 0% | 0% | 100% | 0% |
| RB (+150°, right back) | 0% | 0% | 0% | 100% |

**Result:** When surround audio pans around you (right → rear → left), VBAP smoothly transitions power between front and rear speakers, creating a "wrap-around head" effect. With rear speaker orientation awareness, these mappings adapt to match whether your rear speaker faces you or faces away.

---

## Theater Preset Descriptions

### Cinema (Default)
Standard commercial cinema sound mixing and mastering is calibrated against.
- **Room:** Large multiplex (RT60 1.3s, modest early reflections)
- **Bass:** 6 dB boost at 80 Hz, extended to 30 Hz (typical cinema subwoofer)
- **Surround:** Balanced presence (surround 78%, rear 65% of center)
- **Use for:** Most movies, shows, music content

### IMAX
Extreme-scale theatrical experience mimicking IMAX auditoriums (70mm film, massive screens).
- **Room:** Cavernous space (RT60 1.9s, rich early reflections, 32 ms pre-delay)
- **Bass:** Aggressive, 10 dB boost at 80 Hz with 7 dB sub-bass extension (designed for massive LFE)
- **Surround:** Prominent (surround 90%, rear 78%)
- **Dynamics:** 2.0× compression, 0.85× transient punch
- **Use for:** Action blockbusters, immersive entertainment, bass-heavy content

### Dolby
Precision-engineered for **Dolby Cinema** standards — clarity, articulation, and accurate localization.
- **Room:** Tight, acoustically controlled (RT60 0.95s, minimal early reflections)
- **Bass:** Conservative, 4 dB boost (relies on system headroom)
- **Surround:** Subtle but precise (surround 68%, rear 58%)
- **Dynamics:** Gentle compression (1.2×), restrained transients
- **Use for:** Dialog-heavy content, dramatic performances, audiophile listening

### Home
Residential listening preset with minimal bass boost and moderate room simulation.
- **Room:** Small-to-medium room (RT60 0.8s, subtle early reflections)
- **Bass:** Moderate, 3 dB boost (safe for small speakers)
- **Surround:** Conservative (surround 55%, rear 48%)
- **Dynamics:** Mild compression, gentle transients
- **Use for:** Music listening, podcasts, all-day playback without fatigue

---

## Advanced Configuration

### Real-Time Sliders (Single-Speaker & Multi-Speaker)
All modes support four master sliders:

| Slider | Range | Effect |
|---|---|---|
| **Reverb** | 0.3–2.5 s | Reverberation time (RT60). Higher = larger virtual room. |
| **Width** | 1.0–2.8× | Stereo field expansion via M/S processing. 1.0 = narrow, 2.8 = wide. |
| **Bass** | 0–12 dB | Sub-bass boost at 80 Hz. Adds perceived depth and impact. |
| **Dynamics** | 1.0–2.2 | Multiband compression intensity. 1.0 = off, 2.2 = theatrical punch. |

### Per-Channel Levels (Multi-Speaker Tab)
- **Center Level** — Reduce if dialog is too loud (0.0–1.0)
- **Surround Level** — Control ambient surround presence (0.0–1.0)
- **Rear Level** — Adjust rear speaker prominence (0.0–1.0)
- **LFE Level** — Sub-bass mixing level (0.0–1.0)

### Bass Priority (Multi-Speaker Only)
Choose how sub-bass (0–120 Hz) is distributed:
- **Equal** — 100% to front, 40% to rear (balanced)
- **Front** — 115% to front, 6% to rear (emphasize front bass)
- **Rear** — 6% to front, 115% to rear (emphasize rear bass)

---

## VB-Cable Setup in Detail

### Installation Steps

1. **Download VB-Cable:**
   - Visit https://vb-audio.com/Cable/
   - Download the installer (free download, donation requested)
   - Run `VB-Cable_Setup.exe` as administrator

2. **Install & Configure:**
   - Follow the installer prompts
   - Restart your computer when prompted
   - Open Windows Settings → Sound → Volume mixer
   - Verify **VB-Cable** appears in the playback devices list

3. **Set VB-Cable as Default (Optional):**
   - Right-click speaker icon → Sound settings
   - Under "App volume and device preferences", set ModAudio to output to VB-Cable
   - Or configure within ModAudio's Device dropdown

### Using VB-Cable with ModAudio

**Loopback Routing:**
```
System Audio Output → VB-Cable (Virtual Input)
                   → Rear Speaker Amplifier
```

ModAudio automatically detects VB-Cable and:
- Routes front speaker audio to your default playback device
- Routes rear speaker audio through VB-Cable's virtual loopback
- Applies independent DSP to each bus

**Full Control Mode (Advanced):**
If you have a multi-channel audio interface:
```
ModAudio → USB Interface Left (Front Speaker)
        → USB Interface Right (Rear Speaker)
```
Requires manual device selection in ModAudio's Device dropdown.

---

## Troubleshooting

### No Sound Output
1. **Check device selection** — Ensure "Front Speaker" device matches your connected playback device
2. **Verify system volume** — Make sure Windows volume is not muted
3. **Check format** — Restart ModAudio if you switch devices
4. **Test with silence** — Play a test tone to rule out source issues

### Rear Speaker is Silent (Multi-Speaker Mode)
1. **Verify VB-Cable installation** — Open Sound settings and look for "VB-Cable" device
2. **Check rear speaker connection** — Ensure speakers are powered and cables are connected
3. **Adjust rear level slider** — Increase "Rear Level" in multi-speaker controls
4. **Check audio source** — Some content has minimal surround information; test with surround-encoded material

### VB-Cable Driver Installation Fails
1. **Run as Administrator** — Right-click installer and select "Run as Administrator"
2. **Disable Secure Boot** — Temporarily disable in BIOS if installation still fails
3. **Restart after install** — VB-Cable requires a system restart to activate

### Audio Crackling or Dropouts
1. **Increase buffer size** — Raise BLOCK_SIZE in config.py (larger = more latency but fewer dropouts)
2. **Close CPU-intensive apps** — Reduce load from other applications
3. **Check USB power** — Ensure USB audio interface has adequate power
4. **Update drivers** — Update audio drivers for your device

### Surround Not Noticeable (Multi-Speaker Mode)
1. **Check rear speaker placement** — Rear speaker should be 3–6 ft away from listening position
2. **Test with surround content** — Some audio has minimal surround encoding; use a movie or surround-encoded music
3. **Increase surround level** — Raise "Surround Level" slider in multi-speaker tab
4. **Verify rear azimuth** — Adjust rear position slider to match your physical speaker placement

---

## Audio Specifications

| Parameter | Value |
|---|---|
| **Sample Rate** | 48 kHz (cinema standard) |
| **Bit Depth** | 32-bit float (internal processing) |
| **Buffer Size** | 512 samples (~10.7 ms latency) |
| **HRTF** | Brown-Duda (headphones mode) |
| **Surround Channels** | 5.1 virtual (6 channels + LFE) |
| **Physical Speakers** | 2-channel stereo (multi-speaker mode) |

---

## Performance & Compatibility

### Recommended System
- **CPU:** Intel i7/Ryzen 7 or better (quad-core minimum)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Audio Interface:** USB 2.0+ (USB 3.0 recommended for multi-speaker)
- **Latency:** ~10–20 ms typical roundtrip

### Tested On
- Windows 10 22H2
- Windows 11 Home & Pro
- VB-Cable 4.x
- Python 3.9–3.12

---

## Project Structure

```
ModAudio/
├── app.py                 # Main GUI application
├── config.py              # Audio parameters & theater presets
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── audio_io.py            # Device enumeration and management
├── audio_multi.py         # Multi-speaker audio streaming
├── virtual_device.py      # Virtual audio driver setup (Windows)
├── dsp/
│   ├── __init__.py
│   ├── surround_engine.py # Adaptive 7.1 surround upmix
│   ├── multi_speaker.py   # VBAP routing & multi-speaker DSP
│   ├── filters.py         # IIR filter implementations
│   ├── equalizer.py       # Cinema EQ (X-curve + bass)
│   ├── reverb.py          # FDN reverb + early reflections
│   ├── dynamics.py        # Compressor, limiter, transient enhancer
│   ├── enhancer.py        # Bass harmonics & air-band exciter
│   ├── hrtf_processor.py  # Brown-Duda binaural HRTF
│   ├── utils.py           # Helper functions
│   └── lpc.py             # LPC-based spectral modeling
└── .gitignore
```

---

## License

This project is provided as-is for personal and educational use.

---

## Contributing & Support

### Issues & Feature Requests
If you encounter bugs or have feature suggestions:
1. Test with the included presets
2. Check troubleshooting section above
3. Provide details: audio device, OS version, steps to reproduce

### Credits
- **Brown-Duda HRTF** — Spatial audio rendering
- **VB-Cable** — Virtual audio routing (https://vb-audio.com)
- **NumPy/SciPy** — Signal processing backbone
- **customtkinter** — Modern cross-platform GUI

---

## FAQ

**Q: Can I use ModAudio with wireless headphones (Bluetooth)?**
A: Yes. Select your Bluetooth device in the Device dropdown. Note that Bluetooth may add additional latency (~100–200 ms) on top of ModAudio's ~10 ms.

**Q: Will ModAudio work with my gaming headset?**
A: Absolutely. Gaming headsets are stereo devices, so single-speaker mode works well. Surround games will be downmixed to 2-channel and processed through the binaural engine.

**Q: Do I need an expensive audio interface for multi-speaker?**
A: No. VB-Cable provides a free virtual audio loopback. You can route one speaker through your soundcard's default output and the other through VB-Cable. For best results with balanced L/R output, a modest $100–200 USB interface is helpful.

**Q: How much CPU does ModAudio use?**
A: Single-speaker mode uses ~5–10% CPU on a modern i7. Multi-speaker mode adds ~2–3% (processing two buses in parallel). Adjust BLOCK_SIZE in config.py if you experience dropouts.

**Q: Can I adjust the reverb while audio is playing?**
A: Yes. All sliders (Reverb, Width, Bass, Dynamics) update in real-time while processing continues.

**Q: My rear speaker sounds quieter than the front. How do I fix it?**
A: This is normal for a diffuse surround field. Try these steps:
1. Increase the "Rear Level" slider (multi-speaker tab)
2. Reduce the "Center Level" slider to emphasize surround
3. Switch to IMAX preset, which boosts rear channel levels

**Q: Can I save custom presets?**
A: Currently, you can edit `PRESETS` in `app.py` and rerun to use custom settings. A custom preset saving UI may be added in future versions.

---

**Enjoy theater-quality audio! 🎬🔊**

# ModAudio - Theater Experience for Your Speakers & Headphones
DISLCAIMER: If it sounds too quiet, then change the volume of your speakers (like through the OS's volume control). ESPECIALLY ON WINDOWS.

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

#### Multi-Speaker Mode (2 or More Physical Speakers)
- **VB-Cable integration** — automatically captures audio to route through multiple speakers
- **2-speaker mode** — full 4-channel VBAP (front-L/R, rear-L/R) with swap-rear-LR for correct spatial mapping
- **N-speaker mode (3+)** — add any number of speakers via the "Add Speaker" button; each gets independent device assignment and DSP
- **VBAP routing** (Vector Base Amplitude Panning) maps 7 virtual surround channels to your physical speakers
- **3D great-circle VBAP** — true spherical VBAP that correctly routes to height speakers (Dolby Atmos) in addition to horizontal layouts
- **True surround sound** that moves around your listening position (front → side → rear → side → front)
- **Customizable rear speaker placement** — adjust azimuth angle (90°–170°) to match your room layout
- **Speaker orientation awareness** — "Faces Me" or "Faces Away" settings adapt surround routing to your setup
- **Acoustic delay calibration** — time-align speakers for coherent wavefront merging
- **Behind spectral coloring** — psychoacoustic processing helps you localize rear speakers as coming from behind
- **3D Room Visualization** — interactive 3D room canvas for speaker placement, device assignment, and real-time sound wave animation

### 📺 Raspberry Pi / TV Add-on
- **Headless runner (`pi_runner.py`)** — drop a Pi behind your TV and it becomes a dedicated surround-sound processor
- **No GUI dependency** — runs on Pi OS Lite; drives HDMI, USB DACs, 3.5 mm, or Bluetooth speakers
- **YAML config** — declare your speaker layout once, start at boot via systemd
- **Atmos-style heights work too** — add elevated speakers to the config and the Dolby preset sends overhead content to them

### 🎚️ Theater Presets
- **Cinema** — Standard commercial cinema sound (RT60 1.3s, 6 dB bass boost)
- **IMAX** — Massive room, overwhelming scale, deep bass impact (RT60 1.9s, 10 dB bass boost)
- **Dolby** — Precision-tuned, tighter room, clean dialog intelligibility (RT60 0.95s, 4 dB bass boost) + **Atmos-style 7.1.4 height rendering** that degrades gracefully on smaller setups
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
- **Windows audio API** — WASAPI (Windows Audio Session API) for low-latency audio I/O

### macOS 10.13+ (Intel or Apple Silicon)
- **Python 3.8+** (we recommend 3.11 or later)
- **~200 MB** disk space for dependencies
- **Audio device** (headphones, speakers, or USB audio interface)
- **Core Audio** — macOS native audio framework (built-in)

### Linux (Ubuntu 20.04+, Debian, Fedora)
- **Python 3.8+** (we recommend 3.11 or later)
- **~200 MB** disk space for dependencies
- **Audio device** (headphones, speakers, or USB audio interface)
- **PulseAudio** or **ALSA** — Linux audio server (usually pre-installed)

### Optional: Multi-Speaker Support
- **Virtual audio loopback device** (platform-dependent; see "Virtual Audio Solutions" below)
- Two or more physical speakers positioned in your listening environment
- Audio interface or multi-channel capable soundcard (optional, for separate speaker routing)
- For N-speaker mode (3+): one audio output device per speaker (USB interfaces, multi-channel soundcards, or virtual cables)

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

**Required packages (all platforms):**
- `numpy` — numerical computation
- `scipy` — signal processing (filters, FFT)
- `sounddevice` — cross-platform audio I/O (wraps WASAPI/Core Audio/ALSA)
- `customtkinter` — modern cross-platform GUI framework

**Windows-only packages:**
- `pyaudiowpatch` — WASAPI loopback audio capture
- `comtypes` — COM interface for audio device control

**All packages gracefully handle platform-specific imports.** If you're on macOS or Linux, the Windows-specific packages are safely skipped.

### Step 3: Run the Application

**GUI Mode (Recommended):**
```bash
python app.py
```
The ModAudio interactive GUI window will open. All features (single-speaker, multi-speaker, presets, real-time controls) are available.

**CLI Mode (Headless/Streaming):**
```bash
python main.py                      # Auto-detect devices, headphones mode
python main.py --mode speakers      # Stereo speaker mode
python main.py --list-devices       # Show all available audio devices
python main.py -i 22 -o 15          # Use specific input/output device indices
python main.py --rt60 1.5 --gain -3 # Custom parameters (RT60, gain, etc.)
```
CLI mode is useful for:
- Running on headless systems (servers, single-board computers)
- Streaming scenarios where you don't need GUI controls
- Automation and scripting
- Integration with other audio pipelines

See **CLI Reference** section below for all available options.

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

## Virtual Audio Solutions for Multi-Speaker Mode

To route audio through two independent speakers, you need a **virtual audio loopback device**. This software-based "virtual cable" allows one speaker to play the standard audio while the other plays a modified surround signal.

### Windows: VB-Cable (Recommended)

**VB-Cable A & B** is the gold standard for Windows:
- **Download:** https://vb-audio.com/Cable/
- **Cost:** Free (donations requested)
- **Features:** Low latency (~1 ms), multiple virtual cables available (Cable A, Cable B, etc.)
- **Setup:** Run installer as admin, restart, device appears in Sound settings

**Alternative: Stereo Mix (Built-in, No Installation)**
- Already on your system (if enabled)
- Enable in Settings → Sound → Volume mixer → App volume and device preferences
- Limitations: Slightly higher latency, consumes one audio format slot

### macOS: VB-Cable for Mac or BlackHole

**VB-Cable for Mac** (Recommended)
- **Download:** https://vb-audio.com/Cable/
- **Cost:** Free (donations requested)
- **Features:** Identical to Windows version, seamless integration with Core Audio
- **Setup:** Install and restart; device appears in System Preferences

**BlackHole** (Open-source Alternative)
- **Download:** https://github.com/ExistentialAudio/BlackHole
- **Cost:** Free and open-source
- **Features:** Modern design, virtual audio routing ecosystem
- **Setup:** Install and restart; appears in System Preferences

**Soundflower** (Legacy)
- Older virtual audio device; still works but less maintained
- Not recommended for new setups; prefer VB-Cable or BlackHole

### Linux: VB-Cable or Native PulseAudio Routing

**VB-Cable for Linux**
- **Download:** https://vb-audio.com/Cable/
- **Cost:** Free (donations requested)
- **Features:** Works with PulseAudio and ALSA backends
- **Setup:** Install via package manager or source

**Native PulseAudio Loopback** (Built-in)
- Linux systems typically have PulseAudio with built-in loopback modules
- ModAudio can auto-detect and use `module-loopback`
- No additional software required; zero latency

### Which Should I Use?

| Platform | Recommended | Alternative | Notes |
|---|---|---|---|
| **Windows** | VB-Cable | Stereo Mix | VB-Cable is more reliable and has lower latency |
| **macOS** | VB-Cable for Mac | BlackHole | Both work equally well; BlackHole is open-source |
| **Linux** | Native PulseAudio | VB-Cable | PulseAudio loopback is zero-config and free |

---

## Multi-Speaker Mode Setup

### Prerequisites

1. **Install VB-Cable** (free, by VB-Audio)
   - **Download:** https://vb-audio.com/Cable/
   - **Install** and restart your computer
   - VB-Cable will appear as a playback device in Windows Sound settings

2. **Two or more physical speakers** placed in your room:
   - **Front speaker** — facing your listening position (typical setup)
   - **Rear speaker** — positioned behind or to the side of your listening area
   - Any speaker type (bookshelf, studio monitors, powered, passive, etc.)
   - **For 3+ speakers:** each additional speaker needs its own audio output device (USB interface, virtual cable, etc.)

### Connecting Speakers

**Option A: Loopback Mode (Simplest)**
- Front speaker: connected to your system's main audio output (speakers or amp)
- Rear speaker: connected via VB-Cable virtual loopback
- ModAudio auto-routes: front plays clean audio, rear gets surround-enhanced signal

**Option B: Full Control (Best Fidelity)**
- Both speakers: routed through a **2-channel USB audio interface** or **multichannel soundcard**. However, if you have VB-Cable, you can bypass this entirely. Hence, we recommend VB-Cable as the primary service provider for full control.
- Requires administrator privileges to install the virtual audio driver
- ModAudio manages both speaker outputs independently, each with full DSP processing

### Configuration in ModAudio

1. **Switch to "Multi-Speaker" tab**
2. **THEATER MODE** — Select a preset:
   - **Cinema** — balanced theater sound, 6 dB bass boost
   - **IMAX** — overwhelming scale, 10 dB bass boost, massive reverb (RT60 1.9s)
   - **Dolby** — precision sound, tight room, excellent dialog (RT60 0.95s)
   - **Home** — subtle enhancement for residential listening
3. **Add speakers if needed** — click **"Add Speaker"** to add a third, fourth, or more speakers. Each must be assigned to its own audio output device.
4. **SPEAKER PLACEMENT** — Configure your setup (2-speaker mode):
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

5. **For N-speaker mode (3+):** Use the **3D View** tab to drag each speaker to its physical position, set its facing direction, and assign its audio device. See "3D Room Visualization" for full controls.

   > **All speakers must be assigned to audio devices before clicking "Start" — unassigned speakers produce no audio.**

6. **DEVICE SETUP** — Choose audio devices:
   - **Front Speaker** — your primary playback device
   - **Rear Speaker** — VB-Cable or second audio output

7. **Click "Start"** — Processing begins on all buses simultaneously

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

## N-Speaker Mode (3 or More Speakers)

ModAudio supports **any number of physical speakers** — not just two. Click the **"Add Speaker"** button in the Multi-Speaker tab to add a third, fourth, or more speakers. Each additional speaker gets its own audio device assignment, azimuth, elevation, and facing direction.

### Adding Speakers

1. **Switch to the "Multi-Speaker" tab**
2. Click **"Add Speaker"** at the top of the controls panel — a new speaker entry appears
3. **Assign each speaker** to a physical audio device via its device dropdown
   - Every speaker must be assigned to a device before audio will route to it
4. **Configure speaker positions** (azimuth, elevation) and facing direction in the 3D view (see "3D Room Visualization" below)
5. **Click "Start"** when all speakers are assigned and positioned

### How N-Speaker Routing Works

With 3 or more speakers, ModAudio uses **MultiSpeakerStreamN** — a dedicated audio engine for multi-point speaker arrays. It applies a **two-step VBAP routing** algorithm:

**Step 1 — Speaker selection (3D great-circle VBAP):**
- For each of the 7 virtual surround channels, the two speakers with the smallest angular distance (measured as great-circle arc on a sphere) are found.
- This uses true 3D geometry — height speakers (e.g. Dolby Atmos ceiling channels) are handled correctly and only receive signal from virtual channels that are genuinely close to them in 3D space.
- Gain is distributed between the two closest speakers using constant-power panning.

**Step 2 — Per-speaker L/R driver pan:**
- Within each winning speaker, audio is panned between the speaker's physical left and right driver positions using signed-difference angular panning.
- This avoids VBAP degeneracy that occurs with inward-facing 5.1/7.1 layouts where adjacent speakers' driver positions overlap.

Each speaker in N-speaker mode independently receives:
- Mid/high content from its VBAP gain allocation
- Sub-bass mixed in from the global LFE bus
- Spectral coloring (behind-filter) if positioned behind the listener
- Independent peak limiting to prevent clipping

### Theater Presets in N-Speaker Mode

All four presets (Cinema, IMAX, Dolby, Home) are fully supported in N-speaker mode. You can switch presets at any time — even while audio is playing — by clicking the preset button. The DSP chain rebuilds instantly with all speaker positions preserved.

### Multi-Speaker Audio Flow (N Speakers)

```
Input Audio → Bass Enhancement → EQ → Reverb → Dynamics →
Transient Enhancement → Master Gain →
     ↓ (split into sub-bass and mid/high)
     ├─ SUB-BASS: mixed into each speaker bus with priority weighting
     └─ MID/HIGH: 7-channel adaptive upmix →
          FL, FR, C, LS, RS, LB, RB →
          Two-step VBAP (3D great-circle + per-speaker L/R pan) →
          Speaker 1 bus … Speaker N bus →
               ↓ (per bus)
          Spectral coloring (if rear/side facing)
               ↓
          Per-bus peak limiting
               ↓
          Output to assigned device
```

---

## 3D Room Visualization

The **3D Room View** tab provides a real-time three-dimensional visualization of your speaker layout, your virtual room, and the sound waves generated by each speaker. It is not just decorative — it is also the **primary interface for assigning speakers to audio devices and positioning them accurately in 3D space**.

> **⚠ IMPORTANT: Speakers MUST be assigned to audio devices in the 3D view before audio will route to them.** A speaker that appears in the 3D view but has no device assigned will produce no audio output, even if the system is running. Always verify that every speaker has a device selected before clicking "Start".

### Opening the 3D View

Click the **"3D View"** tab inside the Multi-Speaker tab to open the 3D room canvas.

### Camera Controls

The 3D room canvas supports full interactive camera control with mouse:

| Action | Effect |
|---|---|
| **Left-click + drag** | Orbit the camera — rotate viewpoint around the room |
| **Scroll wheel** | Zoom in / zoom out |
| **Shift + left-click + drag** | Raise or lower a speaker's height (elevation in the vertical axis) |
| **Right-click on a speaker** | Rotate the speaker's facing direction |

### Scaling the Room to Your Real Space

The 3D view renders speakers inside a virtual room that you can scale to match your **actual physical room dimensions**. By default the room is set to a generic 5 × 4 × 2.5 m space.

| Setting | What it controls |
|---|---|
| **Room Width** (`room_width_m`) | Left-to-right dimension of your listening space (metres) |
| **Room Depth** (`room_depth_m`) | Front-to-back dimension (metres) |
| **Room Height** (`room_height_m`) | Ceiling height (metres) |

Setting accurate room dimensions means the speaker positions in the 3D view correspond 1-to-1 with real physical positions. Sound wave propagation distances and wall-reflection angles are calculated from these measurements, so the visualization reflects what is actually happening in your room.

### Placing Speakers

1. **Drag speakers** to their physical positions in the 3D room (left-click + drag a speaker icon)
2. **Rotate their facing direction** (right-click + drag on the speaker) to match how each physical speaker is pointed — front speakers typically face toward you, rear speakers may face you or away
3. **Adjust height** with Shift + drag if your speakers are at different elevations (e.g. floor-level subwoofers vs. ceiling-mounted Atmos height speakers)
4. **Assign each speaker to an audio device** by clicking the speaker and choosing its output device from the assignment panel

> Speakers without a device assignment will be visible in the 3D view but will not produce any audio. Make sure every speaker is assigned before starting playback.

The DSP routing (VBAP gains, behind-filter, L/R driver positions) is **automatically recomputed** whenever you move a speaker or change its facing direction, so the audio routing always stays in sync with what you see in the 3D view.

### Sound Wave Visualization

The 3D view animates the actual sound waves being emitted by each speaker in real time. Rather than simple expanding spheres, the waves are rendered as **directional ray beams** that trace the accurate propagation path from each speaker — including a single wall reflection — through your virtual room.

#### How Rays Are Generated

Each audio tick, a burst of rays ("pulse") is fired from the speaker. The ray cluster is centred on the speaker's facing axis and spread across a 72° emission cone. Rays travel at 343 m/s (scaled to the room) and fade out at approximately 7.5 m. Each ray is pre-traced — direct path plus one wall-bounce reflection — so the visualization has no per-frame physics cost.

**Left-channel rays** originate from the speaker's left driver position (~11 cm left of the speaker centre). **Right-channel rays** originate from the right driver position (~11 cm right). When audio pans left-to-right, the visible ray origin visibly shifts from the left driver to the right driver.

#### Pan Direction Tracking

When the audio through a speaker is louder on one channel than the other (e.g. a sound effect panning from left to right), the **emission cone shifts** up to 28° toward the active channel. A fully right-panned source shifts the cone 28° along the speaker's right axis, causing the ray cluster to sweep across the speaker face as the audio moves. This lets you visually follow a moving sound source (a car driving off-screen, a bird flying past) in the 3D view.

#### Spike vs. Ambient Intensity

The visualization uses a **slow-tracking ambient floor** (exponential moving average with ~1 second half-life) to separate transient spikes from steady background:

| Visual appearance | Meaning |
|---|---|
| **Thin, dim rays** | Ambient background level — steady music, room tone, nature sounds |
| **Thick, bright rays (up to 3× width)** | Transient spike — a sudden sound exceeding 1.4× the ambient floor |

This makes it easy to identify dynamic events (a bird flapping, an explosion, a door slam) even when a constant ambient noise floor is present. In a nature scene with birds: the **thin rays** represent the ambient nature sounds; the **bright, thick rays** fire when a bird makes a sudden movement or call.

#### Ambient Halo

Each speaker also displays a **soft ambient halo** — a translucent oval that expands proportionally to the speaker's average output level. Even when no spike is occurring, the halo indicates whether a speaker is loud or quiet at a glance. Larger halo = more ambient output. This is especially useful for multi-speaker setups where some speakers may be carrying much more signal than others.

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
Precision-engineered for **Dolby Cinema** standards — clarity, articulation, and accurate localization. Also enables **Atmos-style height rendering**.
- **Room:** Tight, acoustically controlled (RT60 0.95s, minimal early reflections)
- **Bass:** Conservative, 4 dB boost (relies on system headroom)
- **Surround:** Subtle but precise (surround 68%, rear 58%)
- **Dynamics:** Gentle compression (1.2×), restrained transients
- **Heights:** 7.1.4-style overhead virtual channels (TFL/TFR/TRL/TRR at ±45° azimuth, +45° elevation). If you place elevated speakers in the 3D room canvas they receive genuine height content; if you do not, the height sends gracefully fall back to your nearest ground speakers, preserving the spatial intent even on small setups.
- **Use for:** Dialog-heavy content, dramatic performances, audiophile listening, any source you want rendered with an Atmos-style spatial impression.

> **Note on Dolby Atmos compliance.** True Dolby Atmos delivery requires E-AC-3 JOC or MAT object metadata, which is only present in the original Atmos bitstream. ModAudio processes decoded PCM stereo, so it cannot play back native Atmos objects. What the Dolby preset does instead is render an **Atmos-inspired virtual 7.1.4 spatial field** from stereo input — height channels are derived from the surround/rear field through a specular-reflection-shaped filter and routed via 3D great-circle VBAP to whichever physical speakers you have. The result is a convincing overhead impression on any speaker layout without pretending to be an Atmos decoder.

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

## Audio APIs Explained

ModAudio uses **platform-native audio APIs** to provide low-latency, high-quality audio I/O. Understanding which API your system uses can help with troubleshooting and optimization.

### WASAPI (Windows Audio Session API)

**Platform:** Windows 10/11 only
**Used by:** ModAudio's `sounddevice` library via PortAudio backend

WASAPI is Microsoft's modern audio API that provides:
- **Low-latency I/O** — ~10–20 ms roundtrip on properly configured systems
- **Hardware mixing** — Windows kernel-mode audio engine (KMixer)
- **Exclusive mode** (optional) — Direct hardware access for professional audio applications
- **Device routing** — Full control over playback and recording devices

ModAudio uses WASAPI in **shared mode** (default, compatible with all apps) rather than exclusive mode (better latency but blocks other audio).

**When you see "WASAPI" in ModAudio:**
- The device is a Windows audio output
- Audio is routed through the Windows audio session manager
- Multiple apps can play through the same device simultaneously

### Core Audio (macOS)

**Platform:** macOS 10.13+ (Intel and Apple Silicon)
**Used by:** ModAudio's `sounddevice` library via PortAudio backend

Core Audio is Apple's native audio framework:
- **Ultra-low latency** — ~5–10 ms on modern Macs (Apple Silicon even lower)
- **Hardware abstraction** — Universal driver interface (HAL)
- **Professional tools** — Used by Logic Pro, Final Cut Pro, etc.
- **Sample-accurate timing** — Precise synchronization for multi-speaker setups

Core Audio handles both **input** (audio capture) and **output** (playback) seamlessly, making it ideal for ModAudio's multi-speaker routing.

**When you see "Core Audio" in ModAudio:**
- The device is a macOS audio input or output
- Audio is routed through Apple's Audio HAL
- Professional latency performance is guaranteed

### ALSA & PulseAudio (Linux)

**Platform:** Linux (Ubuntu, Debian, Fedora, etc.)
**Used by:** ModAudio's `sounddevice` library (configurable backend)

Linux has two main audio systems:

**ALSA (Advanced Linux Sound Architecture)**
- **Kernel-level audio driver interface**
- Direct hardware access, very low latency (~2–5 ms possible)
- Used for professional/gaming audio
- Requires understanding of ALSA device naming conventions

**PulseAudio (Audio Daemon)**
- **User-level audio server** running on top of ALSA
- Simplified device management, automatic mixing
- Standard on most modern Linux distributions
- Slightly higher latency (~10–20 ms) but much more flexible

ModAudio typically auto-detects and uses **PulseAudio** on modern Linux systems (easier setup), but can fall back to **ALSA** if PulseAudio isn't available.

**When you see "PulseAudio" or "ALSA" in ModAudio:**
- PulseAudio = user-friendly, automatic device routing
- ALSA = direct kernel driver, lower latency if properly configured

### Host API Detection

ModAudio automatically detects your system's audio API:
```python
# Internal: device["hostapi"] tells us which API the device uses
"WASAPI" (Windows) → use WASAPI device filtering
"Core Audio" (macOS) → use Core Audio device filtering
"PulseAudio" (Linux) → use PulseAudio device filtering
"ALSA" (Linux) → use ALSA device filtering
```

For **multi-speaker mode**, ModAudio filters devices by API to ensure both speakers use the same audio subsystem for consistent behavior.

---

## CLI Reference

Full command-line options for `main.py` (headless/streaming mode):

```bash
python main.py [OPTIONS]
```

### Device Selection
- `-i, --input DEV` — Input device index (default: auto-detect)
- `-o, --output DEV` — Output device index (default: auto-detect)
- `--list-devices` — Print all available audio devices and exit

### Audio Mode
- `--mode {headphones|speakers}` — Processing mode (default: `headphones`)
  - `headphones` = binaural HRTF rendering with spatial 5.1 processing
  - `speakers` = stereo widening without binaural effects

### Audio Parameters (all optional)
- `--rt60 SECONDS` — Reverberation time (default: preset-dependent, typically 1.3s)
- `--reverb-mix LEVEL` — Reverb wet level 0–1 (default: 0.25, i.e., 25%)
- `--width MULTIPLIER` — Stereo width expansion (default: 2.0)
- `--gain dB` — Output gain in dB (default: -1.5 dB)
- `--drive INTENSITY` — Dynamics compression drive 1.0–2.0 (default: 1.6)

### Audio Config
- `--fs RATE` — Sample rate in Hz (default: 48000)
- `--block-size SIZE` — Buffer size in samples (default: 512, ~10.7 ms latency)

### Examples

**Auto-detect all devices, apply cinema theater effect:**
```bash
python main.py
```

**Use specific devices with custom reverb and bass:**
```bash
python main.py -i 4 -o 7 --rt60 1.8 --width 2.5
```

**Streaming mode for headphones with aggressive dynamics:**
```bash
python main.py --mode headphones --drive 1.9 --gain -2
```

**List all devices to find indices:**
```bash
python main.py --list-devices
```

---

## Raspberry Pi / TV Add-on

ModAudio ships with a separate **headless runner** for Raspberry Pi (or any Linux box) that plugs into your TV as a surround-sound processor. It's isolated from the GUI code — the Pi runner never imports `customtkinter`, so you can install it on Pi OS Lite without a display server.

### What This Is For
Turn a Pi 4/5 (or even a Pi Zero 2) into a dedicated cinema-DSP box for your living-room TV. It captures the TV's audio (via HDMI-ARC, a USB audio capture card, or a PulseAudio monitor), processes it through the same theater chain the desktop app uses, and drives two to many physical speakers — HDMI, 3.5 mm, USB DACs, Bluetooth, or any mix. The Atmos-style height rendering from the Dolby preset works here too: add elevated speakers to the config and they receive overhead content.

### Install on the Pi
```bash
git clone https://github.com/<your-fork>/ModAudio.git
cd ModAudio
sudo apt install python3-pip portaudio19-dev
pip install -r requirements.txt pyyaml
```

### Configure Your Speaker Layout
Discover device indices first:
```bash
python pi_runner.py --list-devices
```

Copy the example config and edit it:
```bash
cp pi_config.example.yaml pi_config.yaml
# edit `input_device`, `speakers[*].device`, and the azimuth/elevation angles
```

Speaker entries use the same conventions as the GUI's 3D room canvas:
`az` measures azimuth in degrees (0 = directly ahead, +90 = listener's right, +180 = directly behind), `el` is elevation above the listener, and `face_az` is the direction the speaker points (defaults to `az + 180`, i.e. facing the listener). A minimal 4-speaker surround layout looks like this:

```yaml
sample_rate: 48000
block_size: 512
preset: Cinema            # Cinema | IMAX | Dolby | Home
bass_priority: equal
input_device: 0           # your HDMI-ARC / USB capture / PulseAudio monitor
speakers:
  - { name: FL, device: 2, az:  -30 }
  - { name: FR, device: 3, az:   30 }
  - { name: RL, device: 4, az: -110 }
  - { name: RR, device: 5, az:  110 }
```

Validate before you run:
```bash
python pi_runner.py -c pi_config.yaml --check
```

### Run It
```bash
python pi_runner.py -c pi_config.yaml
```
Ctrl-C to stop. It stays in the foreground and prints elapsed time plus xrun counts every few seconds.

### Make It Start at Boot (systemd)
```bash
python pi_runner.py --print-systemd -c /home/pi/ModAudio/pi_config.yaml | \
    sudo tee /etc/systemd/system/modaudio.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now modaudio.service
sudo journalctl -u modaudio -f    # watch the log
```

The unit runs under your user account with `Nice=-10` so the DSP thread gets reliable scheduling. It auto-restarts on failure.

### Audio Capture on the Pi
- **HDMI-ARC capture:** easiest, plug the TV's ARC port into a cheap HDMI capture stick.
- **USB audio in:** any class-compliant USB audio adapter with a line-in.
- **Bluetooth A2DP sink:** Pi acts as a Bluetooth speaker target; the TV sends audio over Bluetooth. Enable via `bluetoothctl`.
- **PulseAudio monitor:** if the Pi itself plays back media, capture its own output via the `.monitor` source.

### Atmos on the Pi
Set `preset: Dolby` and add elevated speakers — e.g. a ceiling pair at `az: ±45, el: 45`. No height speakers? Leave them out; heights gracefully fall back to your ground speakers. Same two-step VBAP engine as the desktop app.

---

## Troubleshooting

### General Issues

#### No Sound Output
1. **Check device selection** — Ensure "Front Speaker" device matches your connected playback device
2. **Verify system volume** — Ensure volume is not muted in system preferences
3. **Check format** — Restart ModAudio if you switch devices
4. **Test with silence** — Play a test tone to rule out source issues
5. **Check audio source** — Ensure your input device is actually producing sound

#### Audio Crackling or Dropouts
1. **Increase buffer size** — Raise `BLOCK_SIZE` in `config.py` (larger = more latency but fewer dropouts)
2. **Close CPU-intensive apps** — Reduce load from other applications (browsers, video players, etc.)
3. **Check USB power** — Ensure USB audio interface has adequate power supply
4. **Update drivers** — Update audio drivers for your device
   - **Windows:** Device Manager → Audio devices → Right-click → Update driver
   - **macOS:** System Preferences → Software Update
   - **Linux:** `sudo apt update && sudo apt upgrade` (Debian/Ubuntu)

#### App Crashes on Launch
1. **Check Python version** — Ensure Python 3.8+ is installed (`python --version`)
2. **Reinstall dependencies** — `pip install -r requirements.txt --force-reinstall`
3. **Check for missing modules** — Run `python -c "import sounddevice; print('OK')"` to verify sound library

---

### Windows-Specific Issues

#### "No suitable input device found" Error
**Error message:**
```
ERROR: No suitable input device found.
Install VB-Cable or enable Stereo Mix, then re-run.
```

**Solution 1: Install VB-Cable (Recommended)**
1. Download from https://vb-audio.com/Cable/
2. Run `VB-Cable_Setup.exe` as Administrator
3. Restart your computer
4. Verify in Settings → Sound → Volume mixer → "VB-Cable" appears

**Solution 2: Enable Stereo Mix (Built-in)**
1. Right-click speaker icon → Sound settings
2. Scroll down → Volume mixer → App volume and device preferences
3. Under "Recording", right-click → Show disabled devices
4. Find "Stereo Mix" → Enable
5. Right-click → Set as default recording device
6. Restart ModAudio

#### VB-Cable Driver Installation Fails
1. **Run as Administrator** — Right-click installer and select "Run as Administrator"
2. **Disable Secure Boot** (if needed) — Temporarily disable in BIOS/UEFI if installation fails
3. **Restart after install** — VB-Cable requires a full system restart to activate
4. **Check Windows Defender** — Temporarily disable real-time protection if it blocks installation
5. **Use Compatibility Mode** (last resort) — Right-click installer → Properties → Compatibility → Run in compatibility mode for older Windows version

#### Rear Speaker is Silent (Multi-Speaker Mode)
1. **Verify VB-Cable installation** — Open Sound settings and look for "VB-Cable" device
2. **Check rear speaker connection** — Ensure speakers are powered and cables are connected
3. **Adjust rear level slider** — Increase "Rear Level" in multi-speaker controls
4. **Check audio source** — Some content has minimal surround information; test with surround-encoded movie or music
5. **Verify loopback mode is selected** — In Multi-Speaker tab, check that rear device is set to VB-Cable

---

### macOS-Specific Issues

#### "No suitable input device found" Error
**Error message:**
```
ERROR: No suitable input device found.
Install VB-Cable for Mac or BlackHole, then re-run.
```

**Solution 1: Install VB-Cable for Mac (Recommended)**
1. Download from https://vb-audio.com/Cable/
2. Mount the DMG file and run the installer
3. Restart your Mac
4. Verify in System Preferences → Sound → Input tab → "VB-Cable" appears

**Solution 2: Install BlackHole (Open-source Alternative)**
1. Download from https://github.com/ExistentialAudio/BlackHole/releases
2. Mount the DMG and run the installer
3. Restart your Mac
4. Verify in System Preferences → Sound → Input tab → "BlackHole" appears

#### Core Audio Permission Issues
- If ModAudio fails to access audio devices, verify:
  - System Preferences → Security & Privacy → Microphone (allow ModAudio/Python)
  - Restart ModAudio after granting permissions

#### Virtual Audio Device Not Detected
1. **Verify installation** — Open System Preferences → Sound → Input tab
2. **Look for VB-Cable or BlackHole** — Check both input and output tabs
3. **Restart audio system** — Run: `sudo launchctl stop com.apple.audio.AudioComponentRegistrar && sleep 2 && sudo launchctl start com.apple.audio.AudioComponentRegistrar`
4. **Restart Mac** — Full restart may be required for Core Audio to recognize new devices

#### Low Audio or No Output on Apple Silicon (M1/M2)
- Apple Silicon may have reduced audio buffer performance on first run
- Increase `BLOCK_SIZE` in `config.py` from 512 to 1024
- Restart ModAudio

---

### Linux-Specific Issues

#### "No suitable input device found" Error
**Error message:**
```
ERROR: No suitable input device found.
Install a virtual loopback device (e.g. VB-Cable), then re-run.
```

**Solution 1: Use Native PulseAudio Loopback (Recommended)**
1. PulseAudio loopback is usually pre-installed on most Linux distributions
2. Verify loopback module exists:
   ```bash
   pactl list modules | grep loopback
   ```
3. If missing, install PulseAudio:
   ```bash
   sudo apt install pulseaudio pulseaudio-utils  # Debian/Ubuntu
   sudo dnf install pulseaudio                    # Fedora
   ```
4. Restart audio daemon:
   ```bash
   systemctl --user restart pulseaudio
   ```
5. ModAudio should auto-detect the loopback device on next run

**Solution 2: Install VB-Cable for Linux**
1. Visit https://vb-audio.com/Cable/
2. Follow Linux installation instructions
3. Compile and install module
4. Verify: `pactl list modules | grep vcable`

#### ALSA Device Numbering Issues
If `--list-devices` shows device indices but they change after reboot:
1. **Create ALSA device mapping:**
   ```bash
   sudo nano /etc/asound.conf
   ```
2. Add permanent device aliases:
   ```conf
   pcm.default {
       type hw
       card PCH
   }
   ctl.default {
       type hw
       card PCH
   }
   ```
3. Save and restart audio: `systemctl --user restart pulseaudio`

#### PulseAudio/ALSA Conflicts
If both are present and conflicting:
1. Check which is primary: `pactl info | grep "Server name"`
2. Force PulseAudio: `export PULSE_SERVER=tcp:127.0.0.1:4713` before running ModAudio
3. Force ALSA: Install PulseAudio ALSA plugin: `sudo apt install libasound2-plugins`

#### Permission Denied on Audio Device
```bash
# If you see: "Permission denied" errors
# Add your user to audio group:
sudo usermod -a -G audio $USER

# Apply group changes (logout/login or):
newgrp audio
```

---

### Multi-Speaker Mode Issues (All Platforms)

#### Rear Speaker is Silent
1. **Verify virtual loopback installed** — See OS-specific sections above
2. **Check rear speaker connection** — Ensure speakers are powered and cables are connected
3. **Adjust rear level slider** — Increase "Rear Level" in multi-speaker controls
4. **Check audio source** — Test with surround-encoded content (movies in 5.1 format)
5. **Verify device selection** — In Multi-Speaker tab, ensure rear device is the loopback device (not speakers)

#### Surround Not Noticeable
1. **Check rear speaker placement** — Rear speaker should be 3–6 feet away from listening position
2. **Test with surround content** — Use movies in surround formats (5.1, 7.1) or surround-encoded music
3. **Increase surround level** — Raise "Surround Level" slider in multi-speaker tab
4. **Verify rear azimuth** — Adjust rear position slider (90°–170°) to match your physical speaker placement
5. **Check center level** — Reduce "Center Level" to emphasize surround channels

#### Audio Latency Issues (Multi-Speaker)
1. **Verify loopback latency** — Loopback devices add ~50–100 ms naturally
2. **Reduce buffer size** — Lower `BLOCK_SIZE` in `config.py` to 256 (risks dropouts but lowers latency)
3. **Check speaker acoustic delay** — Ensure "Speaker Acoustic Delay" slider is configured (~3 ms per meter of distance)
4. **Update drivers** — Outdated audio drivers can cause latency; update via system settings

---

## Audio Specifications

| Parameter | Value |
|---|---|
| **Sample Rate** | 48 kHz (cinema standard) |
| **Bit Depth** | 32-bit float (internal processing) |
| **Buffer Size** | 512 samples (~10.7 ms latency) |
| **HRTF** | Brown-Duda (headphones mode) |
| **Surround Channels** | 7 virtual cinema channels (FL, FR, C, LS, RS, LB, RB) |
| **Physical Speakers** | 2-channel stereo or N-speaker (2+, each independently routed) |
| **VBAP Algorithm** | Two-step: 3D great-circle speaker selection + per-speaker L/R constant-power pan |
| **3D Visualization** | Ray-traced directional sound waves, 1 wall reflection, spike/ambient separation |

---

## Performance & Compatibility

### Recommended System (All Platforms)
- **CPU:**
  - **Windows:** Intel i7/Ryzen 7 or better (quad-core minimum)
  - **macOS:** Apple M1/M2 or Intel i7 equivalent
  - **Linux:** Ryzen 5/i5 or better (quad-core minimum)
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** ~200 MB for dependencies
- **Audio Interface:** USB 2.0+ (USB 3.0 recommended for multi-speaker)
- **Latency:**
  - Windows WASAPI: ~10–20 ms typical roundtrip
  - macOS Core Audio: ~5–10 ms (Apple Silicon even lower)
  - Linux PulseAudio: ~10–20 ms typical roundtrip

### Tested On

**Windows:**
- Windows 10 22H2
- Windows 11 Home & Pro
- Python 3.9–3.12

**macOS:**
- macOS 12.x, 13.x, 14.x (Intel)
- macOS 13.x, 14.x (Apple Silicon M1/M2)
- Python 3.9–3.12

**Linux:**
- Ubuntu 20.04 LTS, 22.04 LTS
- Debian 11, 12
- Fedora 37, 38
- Python 3.9–3.12

**Virtual Audio Devices:**
- VB-Cable 4.x (Windows & macOS)
- BlackHole 0.6+ (macOS)
- PulseAudio loopback module (Linux)

---

## Project Structure

```
ModAudio/
├── app.py                 # Main GUI application (Windows, macOS, Linux)
├── main.py                # CLI application for headless/streaming use
├── config.py              # Audio parameters & theater presets
├── requirements.txt       # Python dependencies (cross-platform)
├── README.md              # This file
├── audio_io.py            # Device enumeration (cross-platform)
├── audio_multi.py         # Multi-speaker audio streaming (cross-platform)
├── virtual_device.py      # Virtual audio setup (cross-platform with platform-specific branches)
├── room_canvas.py         # Interactive 2D top-down room map with speaker drag-and-drop
├── room_canvas_3d.py      # Interactive 3D room visualization with ray-traced sound waves
├── dsp/
│   ├── __init__.py
│   ├── surround_engine.py # Adaptive 7.1 surround upmix
│   ├── multi_speaker.py   # VBAP routing & multi-speaker DSP (2-speaker and N-speaker)
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

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Contributing & Support

**ModAudio is an open-source project, and contributions are welcome!** Whether you're fixing bugs, adding features, improving documentation, or testing on new platforms, your help makes ModAudio better for everyone.

### Reporting Issues

If you encounter bugs or unexpected behavior:

1. **Check existing issues** — Search https://github.com/yourusername/ModAudio/issues to avoid duplicates
2. **Test with included presets** — Try the default presets (Cinema, IMAX, Dolby, Home) to isolate custom vs. preset issues
3. **Review troubleshooting** — See the Troubleshooting section above for common solutions
4. **Provide diagnostic information:**
   - Operating system and version (e.g., "Windows 11 22H2", "macOS 14.2 M1", "Ubuntu 22.04")
   - Python version (`python --version`)
   - Audio device name and type (e.g., "Realtek ALC1200", "USB interface XYZ")
   - Steps to reproduce the issue
   - Error message or console output (if any)
   - Audio configuration (single-speaker vs. multi-speaker mode)

**Example issue:**
```
Title: Crackling audio on multi-speaker mode with VB-Cable

OS: Windows 11 Pro 22H2
Python: 3.11.2
Devices: Logitech USB Headset (front), VB-Cable (rear)
Mode: Multi-speaker, IMAX preset

Steps:
1. Set up two speakers
2. Select VB-Cable as rear device
3. Click "Start" in multi-speaker tab
4. Play 5.1 movie

Result: Audio crackles every ~2 seconds on rear speaker

Expected: Smooth multi-speaker playback
```

### Feature Requests & Enhancement Ideas

Have an idea for ModAudio? We'd love to hear it:

- **New audio effects or presets** — Propose cinema modes for different genres or styles
- **UI/UX improvements** — Suggest better control layouts or visualization
- **Platform support** — Help test on new systems or Linux distributions
- **Performance optimization** — Ideas for reducing CPU usage or improving latency
- **Documentation** — Help improve guides for specific platforms or workflows

Open a GitHub issue with tag `[Feature Request]` and describe your idea.

### Contributing Code

**Setup for development:**

```bash
git clone https://github.com/yourusername/ModAudio.git
cd ModAudio
git checkout -b feature/your-feature-name  # Create feature branch
pip install -r requirements.txt
python app.py  # Test your changes
```

**Guidelines for pull requests:**

1. **One feature per PR** — Keep PRs focused and reviewable
2. **Test on your platform** — Verify changes work on Windows/macOS/Linux as applicable
3. **No breaking changes** — Ensure backward compatibility with existing presets and settings
4. **Maintain code style** — Follow existing naming and formatting conventions
5. **Document complex code** — Add comments for non-obvious algorithms or DSP operations
6. **Update README if needed** — If you add features, update documentation

**Areas where contributions are especially welcome:**

- Cross-platform testing and bug fixes
- Audio API improvements (WASAPI/Core Audio/ALSA optimizations)
- GUI enhancements (preset management, custom controls, visualization)
- DSP algorithms (new filters, reverb designs, spatial processing)
- Performance optimization (reduce CPU usage, lower latency)
- Documentation and tutorials
- Linux package support (AppImage, Snap, etc.)

### Testing & Quality Assurance

Help us test ModAudio on different systems:

- **Windows:** Test on latest Windows 11, older Windows 10 versions, different audio devices
- **macOS:** Test on Intel and Apple Silicon machines; both VB-Cable and BlackHole
- **Linux:** Test on Ubuntu, Debian, Fedora; with PulseAudio and ALSA
- **Audio devices:** Test with USB interfaces, Bluetooth devices, built-in audio
- **Multi-speaker scenarios:** Test different speaker placements and configurations

Report findings (working/broken combinations) to help guide development.

### Getting Help

- **Documentation:** Check this README first, especially the Troubleshooting section
- **GitHub Discussions:** Ask questions in GitHub Discussions (if enabled)
- **GitHub Issues:** Report bugs or request features
- **Community:** Check existing closed issues for solutions to similar problems

### Credits

ModAudio stands on the shoulders of excellent open-source and commercial projects:

- **Brown-Duda HRTF** — Spatial audio rendering reference
- **NumPy/SciPy** — Signal processing backbone
- **sounddevice** — Cross-platform audio I/O (https://github.com/spatialaudio/python-sounddevice)
- **customtkinter** — Modern cross-platform GUI (https://github.com/TomSchimansky/CustomTkinter)
- **VB-Cable** — Virtual audio routing (https://vb-audio.com)
- **BlackHole** — Open-source virtual audio device (https://github.com/ExistentialAudio/BlackHole)

Special thanks to all contributors and the audio DSP community.

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

"""
virtual_device.py — ModAudio virtual speaker manager.

Creates a "ModAudio Surround" virtual audio speaker in Windows so the app
appears as a real playback device in Sound Settings.  Audio routed to this
virtual speaker is captured by ModAudio via WASAPI loopback and processed
through the theater DSP chain before being sent to physical speakers.

Driver backend
--------------
Uses the MIT-licensed VirtualDrivers/Virtual-Audio-Driver:
  https://github.com/VirtualDrivers/Virtual-Audio-Driver

The driver is production-signed (SignPath.io Foundation certificate) and
does NOT require Windows test-signing mode.  It is downloaded from the
project's GitHub Releases on first use and cached in
  %APPDATA%\\ModAudio\\drivers\\

Fallback chain (for users who skip the driver install)
-------------------------------------------------------
  ModAudio virtual speaker  →  Stereo Mix  →  WASAPI loopback (pyaudiowpatch)

The WASAPI loopback path (pyaudiowpatch) requires zero installation and
works on any Windows 10/11 PC — it is always available as a last resort.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

try:
    import winreg
    _HAS_WINREG = True
except ImportError:
    winreg = None  # type: ignore
    _HAS_WINREG = False

try:
    import sounddevice as sd
    _HAS_SD = True
except ImportError:
    sd = None  # type: ignore
    _HAS_SD = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODAUDIO_DEVICE_NAME = "ModAudio Surround"
DRIVER_DISPLAY_NAME  = "Virtual Audio Driver"   # name used in the INF file

# GitHub release of VirtualDrivers/Virtual-Audio-Driver
# MIT license, production-signed — https://github.com/VirtualDrivers/Virtual-Audio-Driver
DRIVER_GITHUB_API_URL = (
    "https://api.github.com/repos/VirtualDrivers/Virtual-Audio-Driver/releases/latest"
)
DRIVER_CACHE_DIR = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")), "ModAudio", "drivers"
)

# Registry paths
_MMDEVICES_RENDER   = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render"
_PROP_FRIENDLY_NAME = "{a45c254e-df1c-4efd-8020-67d146a850e0},2"

# Patterns that identify the VirtualDrivers device
_VDRV_PATTERNS = (
    "virtual audio driver",
    "virtualaudiospeaker",
    "virtual audio device",
    "modaudio surround",
)

# Stereo Mix / system loopback capture keywords
_STEREO_MIX_KW = (
    "stereo mix",
    "what u hear",
    "wave out mix",
    "what you hear",
)


# ---------------------------------------------------------------------------
# Admin check
# ---------------------------------------------------------------------------

def is_admin() -> bool:
    """Return True if the current process has administrator privileges."""
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_as_admin() -> None:
    """Re-launch the current Python script with elevated privileges."""
    try:
        import ctypes
        script = os.path.abspath(sys.argv[0])
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, f'"{script}"', None, 1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Driver download
# ---------------------------------------------------------------------------

def get_latest_driver_release() -> dict | None:
    """
    Fetch the latest VirtualDrivers/Virtual-Audio-Driver release info from
    the GitHub API.  Returns a dict with 'tag_name', 'zip_url', 'zip_name',
    or None on failure.
    """
    try:
        req = urllib.request.Request(
            DRIVER_GITHUB_API_URL,
            headers={"User-Agent": "ModAudio/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        tag  = data.get("tag_name", "")
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(".zip"):
                return {
                    "tag_name": tag,
                    "zip_url":  asset["browser_download_url"],
                    "zip_name": name,
                }
    except Exception:
        pass
    return None


def download_driver(progress_cb=None) -> str | None:
    """
    Download the latest VirtualDrivers driver ZIP to DRIVER_CACHE_DIR.

    Parameters
    ----------
    progress_cb : callable(float) | None
        Called with a fraction [0.0, 1.0] as bytes arrive.

    Returns
    -------
    str | None — path to the extracted .inf file, or None on failure.
    """
    os.makedirs(DRIVER_CACHE_DIR, exist_ok=True)

    release = get_latest_driver_release()
    if release is None:
        return None

    zip_path = os.path.join(DRIVER_CACHE_DIR, release["zip_name"])

    # Download (skip if already cached)
    if not os.path.exists(zip_path):
        try:
            def _report(block_num, block_size, total_size):
                if progress_cb and total_size > 0:
                    progress_cb(min(1.0, block_num * block_size / total_size))

            req = urllib.request.Request(
                release["zip_url"],
                headers={"User-Agent": "ModAudio/1.0"},
            )
            # Stream download manually for progress reporting
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk = 65536
                with open(zip_path, "wb") as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
                        if progress_cb and total > 0:
                            progress_cb(min(0.95, downloaded / total))
        except Exception:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return None

    # Extract
    try:
        extract_dir = os.path.join(DRIVER_CACHE_DIR, release["tag_name"])
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Find .inf file
        for root, _dirs, files in os.walk(extract_dir):
            for fname in files:
                if fname.lower().endswith(".inf"):
                    if progress_cb:
                        progress_cb(1.0)
                    return os.path.join(root, fname)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Driver installation
# ---------------------------------------------------------------------------

def install_driver(inf_path: str) -> tuple[bool, str]:
    """
    Install the virtual audio driver using pnputil.

    Requires administrator privileges.  The driver is production-signed so
    Windows test-signing mode is NOT needed.

    Returns (success, message).
    """
    if not os.path.exists(inf_path):
        return False, f"INF file not found: {inf_path}"
    if not is_admin():
        return (False,
                "Administrator privileges required.\n"
                "Right-click ModAudio and choose 'Run as administrator',\n"
                "or click the 'Install (Run as Admin)' button.")
    try:
        result = subprocess.run(
            ["pnputil", "/add-driver", inf_path, "/install"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 or result.returncode == 3010:
            # 3010 = success + reboot required (audio drivers rarely need this)
            return True, "Driver installed successfully."
        return False, (
            f"pnputil failed (code {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    except FileNotFoundError:
        return False, "pnputil not found — please install the driver manually."
    except subprocess.TimeoutExpired:
        return False, "Driver installation timed out."
    except Exception as exc:
        return False, str(exc)


def uninstall_driver() -> tuple[bool, str]:
    """
    Remove the VirtualDrivers virtual audio driver from the system.
    Requires administrator privileges.
    """
    if not is_admin():
        return False, "Administrator privileges required."

    # Find the published INF name via pnputil
    try:
        result = subprocess.run(
            ["pnputil", "/enum-drivers"],
            capture_output=True, text=True, timeout=15,
        )
        lines   = result.stdout.splitlines()
        oem_inf = None
        for i, line in enumerate(lines):
            if any(p in line.lower() for p in _VDRV_PATTERNS):
                # Look backwards for the Published Name line
                for j in range(i, max(i - 8, -1), -1):
                    if "published name" in lines[j].lower():
                        oem_inf = lines[j].split(":", 1)[-1].strip()
                        break
                if oem_inf:
                    break

        if not oem_inf:
            return False, "VirtualDrivers driver not found in pnputil."

        result2 = subprocess.run(
            ["pnputil", "/delete-driver", oem_inf, "/uninstall", "/force"],
            capture_output=True, text=True, timeout=30,
        )
        if result2.returncode in (0, 3010):
            return True, "Driver uninstalled."
        return False, f"Uninstall failed: {result2.stdout}{result2.stderr}"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _scan_mmdevices_for_pattern(patterns: tuple) -> list[tuple[str, str]]:
    """
    Scan HKLM MMDevices Render for endpoints whose friendly names match any
    of the given lowercase patterns.  Returns list of (guid, friendly_name).
    """
    matches = []
    if not _HAS_WINREG:
        return matches
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _MMDEVICES_RENDER,
                            0, winreg.KEY_READ) as root:
            i = 0
            while True:
                try:
                    guid = winreg.EnumKey(root, i)
                    i += 1
                except OSError:
                    break
                prop_path = rf"{_MMDEVICES_RENDER}\{guid}\Properties"
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, prop_path,
                                        0, winreg.KEY_READ) as pk:
                        name, _ = winreg.QueryValueEx(pk, _PROP_FRIENDLY_NAME)
                        if any(p in name.lower() for p in patterns):
                            matches.append((guid, name))
                except OSError:
                    continue
    except Exception:
        pass
    return matches


def find_virtual_driver_device() -> dict:
    """
    Scan sounddevice list for the installed VirtualDrivers endpoint.

    Returns dict with keys:
      found, output_idx, output_name, input_idx, input_name, guid
    """
    result = {
        "found": False,
        "output_idx":  None, "output_name": None,
        "input_idx":   None, "input_name":  None,
        "guid":        None,
    }
    if not _HAS_SD:
        return result

    # Registry lookup for GUID
    matches = _scan_mmdevices_for_pattern(_VDRV_PATTERNS)
    if matches:
        result["guid"] = matches[0][0]

    for i, d in enumerate(sd.query_devices()):
        nl = d["name"].lower()
        if not any(p in nl for p in _VDRV_PATTERNS):
            continue
        if d["max_output_channels"] >= 1 and result["output_idx"] is None:
            result["output_idx"]  = i
            result["output_name"] = d["name"]
            result["found"]       = True
        if d["max_input_channels"] >= 1 and result["input_idx"] is None:
            result["input_idx"]  = i
            result["input_name"] = d["name"]

    return result


def find_stereo_mix_device() -> dict:
    """Return the first Stereo Mix / system-loopback capture device found."""
    result = {"found": False, "input_idx": None, "input_name": None}
    if not _HAS_SD:
        return result
    for i, d in enumerate(sd.query_devices()):
        nl = d["name"].lower()
        if d["max_input_channels"] >= 1 and any(k in nl for k in _STEREO_MIX_KW):
            result["found"]       = True
            result["input_idx"]   = i
            result["input_name"]  = d["name"]
            break
    return result


def find_best_capture_source() -> dict:
    """
    Return the best available Full-Control capture source.

    Priority: ModAudio Surround (VirtualDriver)  →  Stereo Mix  →  None

    Returns dict with keys: idx, name, source_type, found,
                             output_idx, output_name
    """
    vd = find_virtual_driver_device()
    if vd["found"] and vd["input_idx"] is not None:
        return {
            "idx":         vd["input_idx"],
            "name":        vd["input_name"],
            "source_type": "virtual_driver",
            "found":       True,
            "output_idx":  vd["output_idx"],
            "output_name": vd["output_name"],
        }

    sm = find_stereo_mix_device()
    if sm["found"]:
        return {
            "idx":         sm["input_idx"],
            "name":        sm["input_name"],
            "source_type": "stereo_mix",
            "found":       True,
            "output_idx":  None,
            "output_name": None,
        }

    return {"idx": None, "name": None, "source_type": None,
            "found": False, "output_idx": None, "output_name": None}


# ---------------------------------------------------------------------------
# Device renaming
# ---------------------------------------------------------------------------

def rename_device(guid: str, new_name: str) -> tuple[bool, str]:
    """
    Rename the audio endpoint in the Windows MMDevice property store.
    Requires administrator privileges.
    """
    if not _HAS_WINREG:
        return False, "winreg not available"
    if not is_admin():
        return False, "Administrator privileges required."
    prop_path = rf"{_MMDEVICES_RENDER}\{guid}\Properties"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, prop_path,
                            0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, _PROP_FRIENDLY_NAME, 0, winreg.REG_SZ, new_name)
        return True, f"Device renamed to '{new_name}'"
    except PermissionError:
        return False, "Registry write denied — run as administrator."
    except Exception as exc:
        return False, str(exc)


def rename_virtual_driver_to_modaudio() -> tuple[bool, str]:
    """Rename the VirtualDrivers device to 'ModAudio Surround'."""
    matches = _scan_mmdevices_for_pattern(_VDRV_PATTERNS)
    if not matches:
        return False, "VirtualDrivers device not found in registry."
    guid, current = matches[0]
    if current == MODAUDIO_DEVICE_NAME:
        return True, "Already named 'ModAudio Surround'."
    return rename_device(guid, MODAUDIO_DEVICE_NAME)


# ---------------------------------------------------------------------------
# Windows default audio device (IPolicyConfigVista COM interface)
# ---------------------------------------------------------------------------

def set_default_output_device(device_name: str) -> tuple[bool, str]:
    """
    Set the Windows default audio playback device by friendly name.
    Uses the undocumented but widely-used IPolicyConfigVista COM interface.
    Does NOT require administrator privileges.

    Returns (success, message).
    """
    # Find the endpoint ID in the registry
    if not _HAS_WINREG:
        return False, "winreg not available (Windows only)"

    endpoint_id = None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _MMDEVICES_RENDER,
                            0, winreg.KEY_READ) as root:
            i = 0
            while True:
                try:
                    guid = winreg.EnumKey(root, i)
                    i += 1
                except OSError:
                    break
                prop_path = rf"{_MMDEVICES_RENDER}\{guid}\Properties"
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, prop_path,
                                        0, winreg.KEY_READ) as pk:
                        name, _ = winreg.QueryValueEx(pk, _PROP_FRIENDLY_NAME)
                        if device_name.lower() in name.lower():
                            endpoint_id = f"{{{guid}}}"
                            break
                except OSError:
                    continue
    except Exception as exc:
        return False, f"Registry scan failed: {exc}"

    if endpoint_id is None:
        return False, f"Device '{device_name}' not found in MMDevice store."

    # Call IPolicyConfigVista::SetDefaultEndpoint via comtypes
    try:
        import ctypes
        import ctypes.wintypes

        # CLSID and IID for the undocumented IPolicyConfigVista
        _CLSID = "{870af99c-171d-4f9e-af0d-e63df40c2bc9}"
        _IID   = "{f8679f50-850a-41cf-9c72-430f290290c8}"

        # Load comtypes dynamically (optional dependency)
        import comtypes
        from comtypes import GUID, CLSCTX_ALL

        class IPolicyConfigVista(comtypes.IUnknown):
            _iid_ = GUID(_IID)
            _methods_ = [
                comtypes.STDMETHOD(ctypes.HRESULT, "GetMixFormat",       []),
                comtypes.STDMETHOD(ctypes.HRESULT, "GetDeviceFormat",    []),
                comtypes.STDMETHOD(ctypes.HRESULT, "ResetDeviceFormat",  []),
                comtypes.STDMETHOD(ctypes.HRESULT, "SetDeviceFormat",    []),
                comtypes.STDMETHOD(ctypes.HRESULT, "GetProcessingPeriod",[]),
                comtypes.STDMETHOD(ctypes.HRESULT, "SetProcessingPeriod",[]),
                comtypes.STDMETHOD(ctypes.HRESULT, "GetShareMode",       []),
                comtypes.STDMETHOD(ctypes.HRESULT, "SetShareMode",       []),
                comtypes.STDMETHOD(ctypes.HRESULT, "GetPropertyValue",   []),
                comtypes.STDMETHOD(ctypes.HRESULT, "SetPropertyValue",   []),
                comtypes.STDMETHOD(ctypes.HRESULT, "SetDefaultEndpoint",
                                   [ctypes.c_wchar_p, ctypes.c_uint]),
                comtypes.STDMETHOD(ctypes.HRESULT, "SetEndpointVisibility",[]),
            ]

        pc = comtypes.client.CreateObject(
            GUID(_CLSID), interface=IPolicyConfigVista, clsctx=CLSCTX_ALL)

        for role in (0, 1, 2):     # eConsole, eMultimedia, eCommunications
            hr = pc.SetDefaultEndpoint(endpoint_id, role)
            if hr != 0:
                return False, f"SetDefaultEndpoint HRESULT={hr:#010x}"

        return True, f"Windows default output → '{device_name}'"

    except ImportError:
        # comtypes not available — try PowerShell fallback
        return _set_default_via_powershell(device_name)
    except Exception as exc:
        return False, str(exc)


def _set_default_via_powershell(device_name: str) -> tuple[bool, str]:
    """
    Fallback: set default device via PowerShell AudioDeviceCmdlets module.
    Only works if the module is installed.
    """
    try:
        ps_cmd = (
            f"$dev = Get-AudioDevice -Playback | "
            f"Where-Object {{$_.Name -like '*{device_name}*'}} | "
            f"Select-Object -First 1; "
            f"if ($dev) {{ $dev | Set-AudioDevice | Out-Null; exit 0 }} else {{ exit 1 }}"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return True, f"Windows default output → '{device_name}' (via PowerShell)"
        return False, "AudioDeviceCmdlets module not installed — set default manually in Sound Settings."
    except Exception as exc:
        return False, f"PowerShell fallback failed: {exc}"


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def setup_virtual_device(progress_cb=None) -> dict:
    """
    Full setup flow for the ModAudio virtual speaker:

    1. Check if the VirtualDrivers driver is already installed.
    2. If not, download it from GitHub Releases.
    3. Install it via pnputil (requires admin).
    4. Rename to 'ModAudio Surround' via registry (requires admin).

    progress_cb(float, str) — called with (fraction, status_text).

    Returns a dict:
      success    : bool
      message    : str
      need_admin : bool   — True if admin elevation is required
      output_idx : int | None  — sounddevice output index of the virtual device
      input_idx  : int | None  — sounddevice loopback/capture index
    """
    def _prog(frac, txt):
        if progress_cb:
            progress_cb(frac, txt)

    # 1. Already installed?
    vd = find_virtual_driver_device()
    if vd["found"]:
        _prog(0.5, "Driver already installed, renaming…")
        if is_admin() and vd["guid"]:
            rename_virtual_driver_to_modaudio()
            sd.query_devices.cache_clear() if hasattr(sd.query_devices, "cache_clear") else None
            vd = find_virtual_driver_device()
        _prog(1.0, "Ready")
        return {
            "success":    True,
            "message":    f"ModAudio Surround is ready ({vd['output_name'] or 'found'}).",
            "need_admin": False,
            "output_idx": vd["output_idx"],
            "input_idx":  vd["input_idx"],
        }

    if not is_admin():
        return {
            "success":    False,
            "message":    "Driver installation requires administrator privileges.",
            "need_admin": True,
            "output_idx": None,
            "input_idx":  None,
        }

    # 2. Download
    _prog(0.05, "Fetching driver info from GitHub…")
    inf_path = download_driver(
        progress_cb=lambda f: _prog(0.05 + f * 0.55, "Downloading driver…")
    )
    if inf_path is None:
        return {
            "success":    False,
            "message":    "Failed to download the driver.  Check your internet connection.",
            "need_admin": False,
            "output_idx": None,
            "input_idx":  None,
        }

    # 3. Install
    _prog(0.65, "Installing driver (pnputil)…")
    ok, msg = install_driver(inf_path)
    if not ok:
        return {
            "success": False, "message": msg,
            "need_admin": "admin" in msg.lower(),
            "output_idx": None, "input_idx": None,
        }

    # 4. Wait briefly for Windows to enumerate the new device
    _prog(0.80, "Waiting for Windows to register device…")
    time.sleep(2.5)

    # 5. Rename to ModAudio Surround
    _prog(0.90, "Renaming device to 'ModAudio Surround'…")
    rename_virtual_driver_to_modaudio()
    time.sleep(0.5)

    # 6. Verify
    _prog(0.98, "Verifying…")
    vd = find_virtual_driver_device()
    if not vd["found"]:
        return {
            "success":    False,
            "message":    "Driver installed but device not yet visible — try restarting ModAudio.",
            "need_admin": False,
            "output_idx": None,
            "input_idx":  None,
        }

    _prog(1.0, "Done")
    return {
        "success":    True,
        "message":    "ModAudio Surround installed and ready!",
        "need_admin": False,
        "output_idx": vd["output_idx"],
        "input_idx":  vd["input_idx"],
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def get_status() -> dict:
    """
    Return a complete status snapshot.

    Keys
    ----
    driver_installed  : bool
    device_name       : str | None    — current friendly name
    is_modaudio       : bool          — renamed to MODAUDIO_DEVICE_NAME
    output_idx        : int | None
    input_idx         : int | None
    stereo_mix_found  : bool
    stereo_mix_idx    : int | None
    stereo_mix_name   : str | None
    best_capture      : dict
    is_admin          : bool
    """
    vd  = find_virtual_driver_device()
    sm  = find_stereo_mix_device()
    cap = find_best_capture_source()

    device_name = None
    is_modaudio = False
    if vd["found"] and vd["output_name"]:
        device_name = vd["output_name"]
        is_modaudio = MODAUDIO_DEVICE_NAME.lower() in device_name.lower()

    return {
        "driver_installed": vd["found"],
        "device_name":      device_name,
        "is_modaudio":      is_modaudio,
        "output_idx":       vd["output_idx"],
        "input_idx":        vd["input_idx"],
        "guid":             vd.get("guid"),
        "stereo_mix_found": sm["found"],
        "stereo_mix_idx":   sm["input_idx"],
        "stereo_mix_name":  sm["input_name"],
        "best_capture":     cap,
        "is_admin":         is_admin(),
    }


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def open_sound_settings() -> None:
    """Open Windows Sound Settings (Playback devices panel)."""
    import subprocess as _sp
    try:
        _sp.Popen("start ms-settings:sound", shell=True)
    except Exception:
        pass
    try:
        _sp.Popen("control mmsys.cpl", shell=True)
    except Exception:
        pass

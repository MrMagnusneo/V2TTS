# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# In PyInstaller spec context `__file__` may be undefined.
# `SPECPATH` points to the directory containing this spec file.
ROOT = Path(globals().get("SPECPATH", ".")).resolve().parent
SAM_ROOT = ROOT / "tts" / "sam-python"
RU_TTS_ROOT = ROOT / "tts" / "ru_tts-python"
DECTALK_ROOT = ROOT / "tts" / "dectalk-python"
SILERO_ROOT = ROOT / "tts" / "silero-tts-wrapper"
COQUI_ROOT = ROOT / "tts" / "coqui-tts-wrapper"

for package_root in (
    SAM_ROOT,
    RU_TTS_ROOT,
    DECTALK_ROOT,
    SILERO_ROOT,
    COQUI_ROOT,
):
    sys.path.insert(0, str(package_root))

binaries = []
datas = []
hiddenimports = [
    "faster_whisper",
    "faster_whisper.audio",
    "faster_whisper.feature_extractor",
    "faster_whisper.tokenizer",
    "faster_whisper.transcribe",
    "faster_whisper.utils",
    "faster_whisper.vad",
    "faster_whisper.version",
]
excluded_modules = [
    "black",
    "boto3",
    "botocore",
    "cv2",
    "IPython",
    "jedi",
    "keras",
    "lxml",
    "notebook",
    "pandas",
    "PIL",
    "pyarrow",
    "pytest",
    "tensorflow",
    "torchvision",
    "triton",
]

# Whisper runtime assets (including silero_vad_v6.onnx)
datas += collect_data_files("faster_whisper")
datas += collect_data_files("onnx_asr")
datas += copy_metadata("onnx-asr")
binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += collect_submodules("onnx_asr")

# Local streaming STT runtime. Model weights stay in user data and are not
# collected here.
datas += collect_data_files("sherpa_onnx")
datas += copy_metadata("sherpa-onnx")
binaries += collect_dynamic_libs("sherpa_onnx")
hiddenimports += collect_submodules("sherpa_onnx")

# Vendored Python TTS engines.
hiddenimports += collect_submodules("sam_python")
hiddenimports += collect_submodules("ru_tts_python")
hiddenimports += collect_submodules("dectalk_python")
hiddenimports += collect_submodules("silero_tts")
hiddenimports += collect_submodules("coqui_tts")

# Neural TTS runtimes. Model weights remain in their normal user caches and
# are downloaded on first use.
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
datas += torch_datas
binaries += torch_binaries
hiddenimports += torch_hiddenimports

torchaudio_datas, torchaudio_binaries, torchaudio_hiddenimports = collect_all("torchaudio")
datas += torchaudio_datas
binaries += torchaudio_binaries
hiddenimports += torchaudio_hiddenimports

coqui_datas, coqui_binaries, coqui_hiddenimports = collect_all("TTS")
datas += coqui_datas
binaries += coqui_binaries
hiddenimports += coqui_hiddenimports

# Native ru_tts backend is built by installer/build.py before PyInstaller.
ru_tts_bin = RU_TTS_ROOT / "bin"
if ru_tts_bin.exists():
    for file_path in ru_tts_bin.iterdir():
        if file_path.is_file():
            binaries.append((str(file_path), "bin"))


a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[
        str(ROOT),
        str(SAM_ROOT),
        str(RU_TTS_ROOT),
        str(DECTALK_ROOT),
        str(SILERO_ROOT),
        str(COQUI_ROOT),
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="V2TTS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

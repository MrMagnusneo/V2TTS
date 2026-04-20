# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, Tree

ROOT = Path(__file__).resolve().parent.parent

binaries = []
datas = []
hiddenimports = []

# Whisper runtime assets (including silero_vad_v6.onnx)
datas += collect_data_files("faster_whisper")
hiddenimports += collect_submodules("faster_whisper")
binaries += collect_dynamic_libs("onnxruntime")

# App TTS assets
datas += Tree(str(ROOT / "tts"), prefix="tts")

# pyttsx3 and backends
datas += collect_data_files("pyttsx3")
hiddenimports += collect_submodules("pyttsx3")


a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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

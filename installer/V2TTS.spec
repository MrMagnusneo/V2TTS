# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# In PyInstaller spec context `__file__` may be undefined.
# `SPECPATH` points to the directory containing this spec file.
ROOT = Path(globals().get("SPECPATH", ".")).resolve().parent

binaries = []
datas = []
hiddenimports = []

# Whisper runtime assets (including silero_vad_v6.onnx)
datas += collect_data_files("faster_whisper")
hiddenimports += collect_submodules("faster_whisper")
binaries += collect_dynamic_libs("onnxruntime")

# App TTS assets
tts_root = ROOT / "tts"
if tts_root.exists():
    for file_path in tts_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_parent = file_path.relative_to(tts_root).parent
        dest_dir = (Path("tts") / rel_parent).as_posix()
        datas.append((str(file_path), dest_dir))

# Installer runtime assets (node.exe, prebuilt ru_tts.exe, samjs)
runtime_root = ROOT / "installer" / "runtime"
if runtime_root.exists():
    for file_path in runtime_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_parent = file_path.relative_to(runtime_root).parent
        dest_dir = (Path("runtime") / rel_parent).as_posix()
        datas.append((str(file_path), dest_dir))


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

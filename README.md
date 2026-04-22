# V2TTS

## EN

Desktop GUI application for real-time `speech -> text -> speech`.

- STT: `faster-whisper`
- TTS: `ru_tts` (Russian) and `sam` (English)
- GUI: `tkinter`

### Features

- Start/stop loop from GUI (no CLI arguments required).
- Select STT device: `cpu` or `cuda`.
- Select STT model size: `tiny`, `base`, `small`, `medium`, `large-v3`.
- Select audio input/output devices.
- Auto TTS model selection by text language (Cyrillic -> `ru_tts`, Latin -> `sam`).
- Manual TTS model override (`ru_tts`, `sam`).
- Unified TTS root path for Linux/Windows.

### Project structure

- `main.py` - app entry point and controller wiring.
- `gui.py` - GUI only (widgets/events).
- `audio_stream.py` - microphone stream and phrase segmentation.
- `stt.py` - Whisper transcription logic.
- `tts.py` - TTS routing and path resolution.
- `audio_queue.py` - runtime loop (`STT -> TTS -> playback`).
- `devices.py` - audio device discovery helpers.
- `audio_backend.py` - lazy audio backend imports (`sounddevice`, `soundfile`).
- `installer/` - Windows build scripts (`PyInstaller` + `Inno Setup`).

### Requirements

- Python 3.12+
- Node.js (for `sam` synthesis)
- PortAudio system library
- `ru_tts` binary available in the project TTS root or in `PATH`

Python packages are listed in `pyproject.toml` and `requirements.txt`.

### Install

System dependencies:

Ubuntu/Debian:
```bash
sudo apt install libportaudio2 portaudio19-dev
```

Fedora:
```bash
sudo dnf install portaudio portaudio-devel
```

Python dependencies:

Using uv:
```bash
uv sync
```

Or pip:
```bash
pip install -r requirements.txt
```

Optional (after installing PortAudio):

```bash
python -m pip install --force-reinstall sounddevice
```

### TTS paths

Default TTS root:
- `<project>/tts`

Expected layout:
- `tts/sam/dist/samjs.min.js`
- `tts/ru_tts/...` with compiled `ru_tts` binary

Environment overrides:

- `V2TTS_TTS_ROOT`
- `V2TTS_SAM_JS`
- `V2TTS_RU_TTS_BIN`

### Run

```bash
python main.py
```

### Windows build (onefile + installer)

From PowerShell:

```powershell
cd installer
.\build_windows.ps1
```

Result:
- `dist\V2TTS.exe`

`build_windows.ps1` automatically downloads missing `sam` and `ru_tts` sources and prepares runtime assets.
If local `ru_tts` runtime is not found, it tries prebuilt package download first (`ru_tts.exe` or `ru_tts.dll` + `rulex.dll`), and only then falls back to source build.
Node runtime is also auto-staged (`node.exe` from local PATH or official Node.js ZIP download).
Optional overrides:
- `V2TTS_RU_TTS_EXE_URL`
- `V2TTS_RU_TTS_PACKAGE_URL`
- `V2TTS_NODE_EXE_URL`
- `V2TTS_NODE_ZIP_URL`
- `V2TTS_NODE_VERSION` (single version or comma-separated list)

To build installer:

```powershell
cd installer
.\build_installer.ps1
```

If Inno Setup is installed in a custom location, set `V2TTS_ISCC_PATH` to the full `ISCC.exe` path.
If Inno Setup is missing, `build_installer.ps1` tries to install a local compiler copy into `<project>\.tools\InnoSetup6`.

Result:
- `dist-installer\V2TTS-Setup.exe`

### Troubleshooting

- `OSError: PortAudio library not found`:
  install PortAudio system package and reinstall `sounddevice`.
- `ru_tts` not found:
  ensure binary exists in `tts` root or set `V2TTS_RU_TTS_BIN`.
- `sam` not found:
  ensure `samjs.min.js` exists or set `V2TTS_SAM_JS`.
- `WinError 2` on TTS:
  ensure runtime files exist: `runtime/node/node.exe`, `runtime/tts/sam/dist/samjs.min.js`, and either `runtime/tts/ru_tts/bin/ru_tts.exe` or `runtime/tts/ru_tts/bin/ru_tts.dll` (+ `rulex.dll`).

### Notes

- Language split is heuristic (Cyrillic/Latin detection).
- Bottom GUI log is read-only.

---

## RU

Десктопное GUI-приложение для real-time конвейера `speech -> text -> speech`.

- STT: `faster-whisper`
- TTS: `ru_tts` (русский) и `sam` (английский)
- GUI: `tkinter`

### Возможности

- Запуск/остановка из GUI (без аргументов командной строки).
- Выбор устройства STT: `cpu` или `cuda`.
- Выбор размера STT-модели: `tiny`, `base`, `small`, `medium`, `large-v3`.
- Выбор устройств ввода/вывода аудио.
- Автовыбор TTS-модели по языку текста (кириллица -> `ru_tts`, латиница -> `sam`).
- Ручной выбор TTS-модели (`ru_tts`, `sam`).
- Единый кроссплатформенный путь к TTS-моделям (Linux/Windows).

### Структура проекта

- `main.py` - точка входа и связывание компонентов.
- `gui.py` - только GUI (виджеты/события).
- `audio_stream.py` - поток микрофона и разбиение на фразы.
- `stt.py` - логика Whisper.
- `tts.py` - роутинг TTS и резолв путей.
- `audio_queue.py` - runtime-цикл (`STT -> TTS -> playback`).
- `devices.py` - функции получения аудиоустройств.
- `audio_backend.py` - ленивые импорты аудио-бэкенда (`sounddevice`, `soundfile`).
- `installer/` - скрипты сборки Windows (`PyInstaller` + `Inno Setup`).

### Требования

- Python 3.12+
- Node.js (для `sam`)
- Системная библиотека PortAudio
- Бинарник `ru_tts` в TTS-каталоге проекта или в `PATH`

Python-зависимости указаны в `pyproject.toml` и `requirements.txt`.

### Установка

Системные зависимости:

Ubuntu/Debian:
```bash
sudo apt install libportaudio2 portaudio19-dev
```

Fedora:
```bash
sudo dnf install portaudio portaudio-devel
```

Python-зависимости:

Через uv:
```bash
uv sync
```

Или через pip:
```bash
pip install -r requirements.txt
```

Опционально (после установки PortAudio):

```bash
python -m pip install --force-reinstall sounddevice
```

### Пути к TTS

Путь по умолчанию:
- `<project>/tts`

Ожидаемая структура:
- `tts/sam/dist/samjs.min.js`
- `tts/ru_tts/...` с собранным бинарником `ru_tts`

Переопределение через переменные окружения:

- `V2TTS_TTS_ROOT`
- `V2TTS_SAM_JS`
- `V2TTS_RU_TTS_BIN`

### Запуск

```bash
python main.py
```

### Сборка Windows (onefile + установщик)

Из PowerShell:

```powershell
cd installer
.\build_windows.ps1
```

Результат:
- `dist\V2TTS.exe`

`build_windows.ps1` автоматически скачивает недостающие исходники `sam` и `ru_tts` и готовит runtime-ассеты.
Если локальный runtime `ru_tts` не найден, скрипт сначала пробует скачать готовый пакет (`ru_tts.exe` или `ru_tts.dll` + `rulex.dll`), и только потом переходит к сборке из исходников.
Runtime для Node тоже подготавливается автоматически (`node.exe` из PATH или из официального ZIP Node.js).
Опциональные переменные:
- `V2TTS_RU_TTS_EXE_URL`
- `V2TTS_RU_TTS_PACKAGE_URL`
- `V2TTS_NODE_EXE_URL`
- `V2TTS_NODE_ZIP_URL`
- `V2TTS_NODE_VERSION` (одна версия или список через запятую)

Сборка установщика:

```powershell
cd installer
.\build_installer.ps1
```

Если Inno Setup установлен в нестандартную директорию, задай `V2TTS_ISCC_PATH` (полный путь к `ISCC.exe`).
Если Inno Setup не установлен, `build_installer.ps1` попробует поставить локальную копию компилятора в `<project>\.tools\InnoSetup6`.

Результат:
- `dist-installer\V2TTS-Setup.exe`

### Решение проблем

- `OSError: PortAudio library not found`:
  установи системный пакет PortAudio и переустанови `sounddevice`.
- Не найден `ru_tts`:
  проверь, что бинарник есть в `tts` или задай `V2TTS_RU_TTS_BIN`.
- Не найден `sam`:
  проверь, что `samjs.min.js` существует, или задай `V2TTS_SAM_JS`.
- `WinError 2` при TTS:
  проверь runtime-файлы: `runtime/node/node.exe`, `runtime/tts/sam/dist/samjs.min.js`, и либо `runtime/tts/ru_tts/bin/ru_tts.exe`, либо `runtime/tts/ru_tts/bin/ru_tts.dll` (+ `rulex.dll`).

### Примечания

- Выбор языка для TTS эвристический (кириллица/латиница).
- Нижний лог в GUI только для чтения.

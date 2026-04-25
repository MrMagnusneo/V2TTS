# V2TTS

## EN

Desktop GUI app for a real-time `speech -> text -> speech` loop.

- STT: `faster-whisper`
- TTS: vendored Python `ru_tts` for Russian and vendored Python `sam` for English
- GUI: `tkinter`
- Packaging: one cross-platform Python build script

### Features

- Start/stop from GUI, no command-line arguments required.
- Select STT device: `cpu` or `cuda`.
- Select STT model size: `tiny`, `base`, `small`, `medium`, `large-v3`.
- Select audio input/output devices.
- Auto TTS model selection by text language: Cyrillic -> `ru_tts`, Latin -> `sam`.
- Manual TTS model override: `ru_tts` or `sam`.

### Project Structure

- `main.py` - app entry point and controller wiring.
- `gui.py` - GUI widgets and UI events.
- `audio_stream.py` - microphone stream and phrase segmentation.
- `audio_queue.py` - runtime loop: `STT -> TTS -> playback`.
- `stt.py` - Whisper transcription logic.
- `tts.py` - TTS routing through the vendored Python engines.
- `devices.py` - audio device discovery helpers.
- `audio_backend.py` - lazy audio backend imports.
- `installer/build.py` - Linux/Windows PyInstaller build script.
- `installer/V2TTS.spec` - PyInstaller spec.
- `installer/V2TTS.iss` - optional Windows Inno Setup installer script.

### Requirements

- Python 3.12+
- Git submodules initialized
- C compiler available as `gcc` to build the native `ru_tts` backend
- PortAudio system library for `sounddevice`

Python dependencies are listed in `pyproject.toml` and `requirements.txt`.

### Install

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/MrMagnusneo/V2TTS.git
```

Or initialize submodules in an existing clone:

```bash
git submodule update --init --recursive
```

System dependencies:

Ubuntu/Debian:

```bash
sudo apt install gcc libportaudio2 portaudio19-dev
```

Fedora:

```bash
sudo dnf install gcc portaudio portaudio-devel
```

Windows:

- Install Python 3.12+
- Install a GCC toolchain such as MSYS2 MinGW-w64
- Optional installer build: install Inno Setup 6

Python dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Build

Build on the OS you want to distribute for. PyInstaller does not cross-compile.

```bash
python installer/build.py
```

Results:

- Windows: `dist/V2TTS.exe`
- Linux: `dist/V2TTS`

Build the Windows installer:

```bash
python installer/build.py --installer
```

Result:

- `dist-installer/V2TTS-Setup.exe`

### Troubleshooting

- `OSError: PortAudio library not found`: install PortAudio and reinstall `sounddevice`.
- CUDA DLL errors: select `cpu` in the GUI or install the CUDA runtime required by your `ctranslate2` build.
- `gcc` not found during build: install GCC/MSYS2 MinGW-w64 and make sure it is in `PATH`.
- Slow STT: use `tiny` or `base`, select `cpu` if CUDA is not configured correctly.

---

## RU

Десктопное GUI-приложение для real-time конвейера `speech -> text -> speech`.

- STT: `faster-whisper`
- TTS: vendored Python `ru_tts` для русского и vendored Python `sam` для английского
- GUI: `tkinter`
- Сборка: один кроссплатформенный Python-скрипт

### Возможности

- Запуск/остановка из GUI, без аргументов командной строки.
- Выбор устройства STT: `cpu` или `cuda`.
- Выбор размера STT-модели: `tiny`, `base`, `small`, `medium`, `large-v3`.
- Выбор устройств ввода/вывода аудио.
- Автовыбор TTS по языку текста: кириллица -> `ru_tts`, латиница -> `sam`.
- Ручной выбор TTS: `ru_tts` или `sam`.

### Структура Проекта

- `main.py` - точка входа и связывание компонентов.
- `gui.py` - виджеты GUI и события UI.
- `audio_stream.py` - поток микрофона и разбиение на фразы.
- `audio_queue.py` - runtime-цикл: `STT -> TTS -> playback`.
- `stt.py` - логика Whisper.
- `tts.py` - роутинг TTS через vendored Python-движки.
- `devices.py` - поиск аудиоустройств.
- `audio_backend.py` - ленивые импорты аудио-бэкенда.
- `installer/build.py` - скрипт сборки для Linux/Windows через PyInstaller.
- `installer/V2TTS.spec` - spec-файл PyInstaller.
- `installer/V2TTS.iss` - опциональный установщик Windows через Inno Setup.

### Требования

- Python 3.12+
- Инициализированные git submodules
- C-компилятор `gcc` для сборки native backend `ru_tts`
- Системная библиотека PortAudio для `sounddevice`

Python-зависимости указаны в `pyproject.toml` и `requirements.txt`.

### Установка

Клонирование с submodules:

```bash
git clone --recurse-submodules https://github.com/MrMagnusneo/V2TTS.git
```

Или инициализация submodules в уже существующем клоне:

```bash
git submodule update --init --recursive
```

Системные зависимости:

Ubuntu/Debian:

```bash
sudo apt install gcc libportaudio2 portaudio19-dev
```

Fedora:

```bash
sudo dnf install gcc portaudio portaudio-devel
```

Windows:

- Установи Python 3.12+
- Установи GCC toolchain, например MSYS2 MinGW-w64
- Для сборки установщика установи Inno Setup 6

Python-зависимости:

```bash
python -m pip install -r requirements.txt
```

### Запуск

```bash
python main.py
```

### Сборка

Собирать нужно на той ОС, под которую нужен бинарник. PyInstaller не делает cross-compile.

```bash
python installer/build.py
```

Результаты:

- Windows: `dist/V2TTS.exe`
- Linux: `dist/V2TTS`

Сборка Windows-установщика:

```bash
python installer/build.py --installer
```

Результат:

- `dist-installer/V2TTS-Setup.exe`

### Решение Проблем

- `OSError: PortAudio library not found`: установи PortAudio и переустанови `sounddevice`.
- Ошибки CUDA DLL: выбери `cpu` в GUI или установи CUDA runtime, который нужен твоей сборке `ctranslate2`.
- `gcc` не найден при сборке: установи GCC/MSYS2 MinGW-w64 и добавь его в `PATH`.
- STT работает медленно: используй `tiny` или `base`, выбирай `cpu`, если CUDA настроена некорректно.

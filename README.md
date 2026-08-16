# V2TTS

## EN

Desktop GUI app for a real-time `speech -> text -> speech` loop.

- STT: local GigaAM v3 for Russian and `faster-whisper` for Russian/English
- TTS: vendored Python `ru_tts` for Russian and vendored Python `sam` for English
- GUI: `tkinter`
- Packaging: one cross-platform Python build script

### Features

- Start/stop from GUI, no command-line arguments required. `Stop` interrupts the
  pipeline worker without closing the window.
- Select an explicit `Russian` or `English` recognition profile.
- Select STT device: `cpu` or `cuda`.
- Russian default: GigaAM v3 E2E RNN-T. Russian alternatives: GigaAM v3 E2E
  CTC and Whisper `small`, `medium`, or `large-v3`.
- English: explicit-English Whisper `small`, `medium`, or `large-v3`.
- Select audio input/output devices.
- Auto TTS model selection by text language: Cyrillic -> `ru_tts`, Latin -> `sam`.
- Manual TTS model override: `ru_tts` or `sam`.

### Project Structure

- `main.py` - app entry point and controller wiring.
- `gui.py` - GUI widgets and UI events.
- `audio_stream.py` - microphone stream and phrase segmentation.
- `audio_queue.py` - parent-side worker lifecycle and safe Stop escalation.
- `pipeline.py` - child-owned `capture -> STT -> TTS -> playback` pipeline.
- `stt_profiles.py` - language/engine/model catalog and model paths.
- `stt.py` - GigaAM and Whisper transcription adapters.
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
- Install MSYS2 and its UCRT64 GCC toolchain:

```powershell
winget install --id MSYS2.MSYS2 -e
C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc"
```

The build script detects GCC in standard MSYS2 directories even when it is
not yet present in `PATH`.

- Optional installer build: install Inno Setup 6

Python dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run

```bash
python main.py
```

The first use of a selected STT model downloads its weights. Later recognition
can run offline. On Windows the files are stored outside the executable under:

```text
%LOCALAPPDATA%\V2TTS\models
```

The executable contains the inference runtimes, but no GigaAM or Whisper model
weights. If CUDA initialization fails, the log shows the reason and the actual
CPU fallback instead of silently changing devices.

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

### Tests

Run the automated suite:

```bash
python -m pytest -q
```

The Windows workflow builds the frozen executable and runs the packaged TTS plus
multiprocessing smoke checks automatically. The smoke does not download an STT
model. You can run the same check locally after a build:

```bash
dist/V2TTS.exe --smoke-test
```

The hardware test is deliberately opt-in because CI runners do not expose a
real microphone:

```bash
V2TTS_REAL_AUDIO_TEST=1 python -m pytest -m integration
```

### Troubleshooting

- `OSError: PortAudio library not found`: install PortAudio and reinstall `sounddevice`.
- CUDA DLL errors: select `cpu` in the GUI or install the CUDA runtime required by your `ctranslate2` build.
- `gcc` not found during build: install GCC/MSYS2 MinGW-w64 and make sure it is in `PATH`.
- Slow STT: use GigaAM v3 E2E CTC for Russian or Whisper `small`; check the log
  if requested CUDA fell back to CPU.

---

## RU

Десктопное GUI-приложение для real-time конвейера `speech -> text -> speech`.

- STT: локальный GigaAM v3 для русского и `faster-whisper` для русского/английского
- TTS: vendored Python `ru_tts` для русского и vendored Python `sam` для английского
- GUI: `tkinter`
- Сборка: один кроссплатформенный Python-скрипт

### Возможности

- Запуск/остановка из GUI, без аргументов командной строки. `Stop` прерывает
  дочерний конвейер и не закрывает окно.
- Отдельные профили распознавания `Russian` и `English`.
- Выбор устройства STT: `cpu` или `cuda`.
- Русский профиль по умолчанию: GigaAM v3 E2E RNN-T. Также доступны GigaAM v3
  E2E CTC и Whisper `small`, `medium`, `large-v3`.
- Английский профиль: Whisper `small`, `medium`, `large-v3` с явно заданным
  английским языком.
- Выбор устройств ввода/вывода аудио.
- Автовыбор TTS по языку текста: кириллица -> `ru_tts`, латиница -> `sam`.
- Ручной выбор TTS: `ru_tts` или `sam`.

### Структура Проекта

- `main.py` - точка входа и связывание компонентов.
- `gui.py` - виджеты GUI и события UI.
- `audio_stream.py` - поток микрофона и разбиение на фразы.
- `audio_queue.py` - управление дочерним процессом и безопасным Stop.
- `pipeline.py` - дочерний конвейер `capture -> STT -> TTS -> playback`.
- `stt_profiles.py` - каталог языков, движков, моделей и пути к весам.
- `stt.py` - адаптеры GigaAM и Whisper.
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
- Установи MSYS2 и GCC toolchain UCRT64:

```powershell
winget install --id MSYS2.MSYS2 -e
C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc"
```

Скрипт сборки сам находит GCC в стандартных каталогах MSYS2, даже если путь
к нему ещё не добавлен в `PATH`.

- Для сборки установщика установи Inno Setup 6

Python-зависимости:

```bash
python -m pip install -r requirements.txt
```

### Запуск

```bash
python main.py
```

При первом выборе STT-модели её веса скачиваются автоматически. После загрузки
распознавание может работать без интернета. В Windows веса хранятся отдельно от
`.exe`:

```text
%LOCALAPPDATA%\V2TTS\models
```

В `.exe` находятся только runtime-компоненты, но не веса GigaAM/Whisper. Если
CUDA не запускается, причина и фактически выбранный CPU отображаются в журнале.

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

### Тесты

Запуск автоматических тестов:

```bash
python -m pytest -q
```

Windows workflow собирает frozen executable и автоматически запускает smoke-тест
упакованных TTS-движков и дочернего процесса. STT-модели при этом не скачиваются.
Тот же тест можно запустить локально после сборки:

```bash
dist/V2TTS.exe --smoke-test
```

Тест реального аудиоустройства включается явно, потому что у CI runner нет
настоящего микрофона:

```bash
V2TTS_REAL_AUDIO_TEST=1 python -m pytest -m integration
```

### Решение Проблем

- `OSError: PortAudio library not found`: установи PortAudio и переустанови `sounddevice`.
- Ошибки CUDA DLL: выбери `cpu` в GUI или установи CUDA runtime, который нужен твоей сборке `ctranslate2`.
- `gcc` не найден при сборке: установи GCC/MSYS2 MinGW-w64 и добавь его в `PATH`.
- STT работает медленно: для русского выбери GigaAM v3 E2E CTC или Whisper
  `small`; проверь журнал — возможно, CUDA переключилась на CPU.

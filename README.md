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
- Manual TTS model override.
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

### Troubleshooting

- `OSError: PortAudio library not found`:
  install PortAudio system package and reinstall `sounddevice`.
- `ru_tts` not found:
  ensure binary exists in `tts` root or set `V2TTS_RU_TTS_BIN`.
- `sam` not found:
  ensure `samjs.min.js` exists or set `V2TTS_SAM_JS`.

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
- Ручной выбор TTS-модели.
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

### Решение проблем

- `OSError: PortAudio library not found`:
  установи системный пакет PortAudio и переустанови `sounddevice`.
- Не найден `ru_tts`:
  проверь, что бинарник есть в `tts` или задай `V2TTS_RU_TTS_BIN`.
- Не найден `sam`:
  проверь, что `samjs.min.js` существует, или задай `V2TTS_SAM_JS`.

### Примечания

- Выбор языка для TTS эвристический (кириллица/латиница).
- Нижний лог в GUI только для чтения.

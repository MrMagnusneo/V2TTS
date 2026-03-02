import speech_recognition as sr
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import time
import tempfile
from silero_tts.silero_tts import SileroTTS

# --- КОНФИГУРАЦИЯ ---
# ID вашего виртуального кабеля (CABLE Input).
# Убедитесь, что ID верен (проверьте через sd.query_devices())
VIRTUAL_MIC_ID = 7

# Язык распознавания (для Google Speech Recognition)
RECOGNIZE_LANGUAGE = "ru-RU"

# --- НАСТРОЙКИ SILERO ---
# Модель 'v4_ru' часто более стабильна для простых задач, но можно пробовать 'v5_ru'
SILERO_MODEL = 'v4_ru'
SILERO_SPEAKER = 'eugene'  # Доступные: aidar, baya, kseniya, xenia, eugene
SILERO_SAMPLE_RATE = 24000  # Лучше использовать 24000 или 48000 для качества
SILERO_DEVICE = 'cuda'  # 'cuda' если есть GPU

# Инициализация модели Silero (делаем один раз при старте, чтобы не грузить каждый раз)
print("Загрузка модели Silero TTS...")
try:
    tts_engine = SileroTTS(
        model_id=SILERO_MODEL,
        language='ru',
        speaker=SILERO_SPEAKER,
        sample_rate=SILERO_SAMPLE_RATE,
        device=SILERO_DEVICE
    )
    print(f"Модель {SILERO_MODEL} (спикер: {SILERO_SPEAKER}) успешно загружена.")
except Exception as e:
    print(f"Ошибка загрузки Silero TTS: {e}")
    exit(1)


def generate_silero_audio(text):
    """
    Генерирует аудио файл с помощью Silero TTS.
    Возвращает путь к временному файлу .wav
    """
    # Создаем имя временного файла.
    # Важно: silero-tts сама сохраняет файл, нам нужно дать ей путь.
    temp_wav = tempfile.mktemp(suffix=".wav")

    try:
        print(f"[Silero] Генерирую голос для: {text}")
        # Метод tts сохраняет файл по указанному пути
        tts_engine.tts(text, temp_wav)

        if os.path.exists(temp_wav):
            return temp_wav
        else:
            print("Ошибка: Файл не был создан.")
            return None

    except Exception as e:
        print(f"Ошибка генерации голоса: {e}")
        return None


def play_audio_to_mic(filename, device_id):
    """
    Читает WAV файл, добавляет тишину в конец и играет в виртуальный кабель.
    """
    try:
        # Читаем файл
        data, fs = sf.read(filename, dtype='float32')

        # --- ИСПРАВЛЕНИЕ ---
        # Добавляем 0.5 - 0.8 секунды тишины в конец массива.
        # Это создаст буфер, который "съестся" при остановке, а полезный звук останется.
        silence_duration = 0.6  # Секунды
        silence_length = int(fs * silence_duration)

        # Создаем массив нулей (тишины)
        silence = np.zeros(silence_length, dtype='float32')

        # Если вдруг звук стерео (2 канала), тишина тоже должна быть стерео
        if len(data.shape) > 1:
            # Превращаем одномерный массив тишины в двумерный (N, 2)
            silence = np.tile(silence.reshape(-1, 1), (1, data.shape[1]))

        # Склеиваем голос и тишину
        data_with_silence = np.concatenate((data, silence))

        # Играем
        sd.play(data_with_silence, fs, device=device_id)
        sd.wait()  # Ждем конца воспроизведения (включая тишину)

        # Дополнительная страховка
        time.sleep(0.1)

    except Exception as e:
        print(f"Ошибка воспроизведения в устройство {device_id}: {e}")


def main():
    recognizer = sr.Recognizer()

    # Используем микрофон по умолчанию как источник (ваш реальный микрофон)
    with sr.Microphone() as source:
        print("Калибровка шума... (помолчите 1 секунду)")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print(f"Готов! Слушаю... (Вывод пойдет на устройство ID: {VIRTUAL_MIC_ID})")

        while True:
            try:
                print("Говорите...")
                # timeout - сколько ждем начала речи
                # phrase_time_limit - максимальная длина фразы
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

                print("Распознаю...")
                # Используем Google Web Speech API
                text = recognizer.recognize_google(audio, language=RECOGNIZE_LANGUAGE)
                print(f"Вы сказали: {text}")

                if text:
                    # 1. Генерируем аудио файл голосом Silero
                    wav_file = generate_silero_audio(text)

                    # 2. Если файл создан, играем его в виртуальный микрофон
                    if wav_file:
                        play_audio_to_mic(wav_file, VIRTUAL_MIC_ID)
                        try:
                            os.remove(wav_file)  # Удаляем временный файл
                        except PermissionError:
                            pass  # Иногда файл занят, не критично
                    else:
                        print("Аудио не было сгенерировано.")

            except sr.WaitTimeoutError:
                pass  # Просто продолжаем слушать, если тишина
            except sr.UnknownValueError:
                print("Не удалось распознать речь")  # Это нормально, если был шум
            except sr.RequestError as e:
                print(f"Ошибка сервиса распознавания Google: {e}")
            except KeyboardInterrupt:
                print("\nВыход...")
                break
            except Exception as e:
                print(f"Непредвиденная ошибка: {e}")


if __name__ == "__main__":
    main()
"""
transcribe_reference.py
Автоматическая транскрипция референсного аудио для config
"""

from faster_whisper import WhisperModel
import soundfile as sf
import os


def transcribe_audio(audio_path, model_size="base.en"):
    """
    Транскрибирует аудио файл и возвращает текст
    
    Args:
        audio_path: Путь к аудио файлу
        model_size: Размер модели Whisper (tiny.en, base.en, small.en, medium.en, large)
    """
    
    print("\n" + "="*60)
    print("🎤 ТРАНСКРИПЦИЯ РЕФЕРЕНСНОГО АУДИО")
    print("="*60)
    
    if not os.path.exists(audio_path):
        print(f"❌ Файл не найден: {audio_path}")
        return None
    
    # Проверка формата
    try:
        data, sr = sf.read(audio_path)
        duration = len(data) / sr
        print(f"📁 Файл: {audio_path}")
        print(f"⏱️ Длительность: {duration:.2f} секунд")
        print(f"📊 Частота: {sr} Hz")
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return None
    
    # Загрузка модели Whisper
    print(f"\n⏳ Загрузка модели Whisper ({model_size})...")
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    except:
        print("⚠️ CUDA недоступна, используется CPU (медленнее)")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Транскрипция
    print("🎯 Транскрибирование...")
    segments, info = model.transcribe(audio_path, language="en")
    
    # Сборка текста
    full_text = ""
    print("\n📝 РЕЗУЛЬТАТ ТРАНСКРИПЦИИ:")
    print("-" * 60)
    
    for segment in segments:
        text = segment.text.strip()
        full_text += text + " "
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")
    
    full_text = full_text.strip()
    
    print("-" * 60)
    print("\n✅ ПОЛНЫЙ ТЕКСТ:")
    print(f'"{full_text}"')
    
    # Генерация кода для config
    print("\n" + "="*60)
    print("📋 СКОПИРУЙТЕ В config_gpt_sovits.py:")
    print("="*60)
    print(f"""
GPT_SOVITS_CONFIG = {{
    'api_url': 'http://127.0.0.1:19880',
    'reference_audio': '{audio_path}',
    'reference_text': '{full_text}',
    'language': 'en',
    'speed': 1.0,
}}
""")
    print("="*60)
    
    return full_text


def batch_transcribe(audio_folder="voice"):
    """Транскрибирует все WAV файлы в папке"""
    
    print("\n🔄 ПАКЕТНАЯ ТРАНСКРИПЦИЯ")
    print(f"📁 Папка: {audio_folder}")
    
    if not os.path.exists(audio_folder):
        print(f"❌ Папка не найдена: {audio_folder}")
        return
    
    wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    
    if not wav_files:
        print("❌ WAV файлы не найдены")
        return
    
    print(f"✅ Найдено файлов: {len(wav_files)}")
    
    results = {}
    
    for i, filename in enumerate(wav_files, 1):
        file_path = os.path.join(audio_folder, filename)
        print(f"\n[{i}/{len(wav_files)}] Обрабатывается: {filename}")
        
        text = transcribe_audio(file_path)
        if text:
            results[filename] = text
        
        if i < len(wav_files):
            print("\n⏸️ Нажмите Enter для продолжения...")
            input()
    
    # Сводка
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ СВОДКА")
    print("="*60)
    
    for filename, text in results.items():
        print(f"\n{filename}:")
        print(f'  "{text}"')


if __name__ == "__main__":
    print("\n🎤 УТИЛИТА ТРАНСКРИПЦИИ")
    print("\nВыберите режим:")
    print("1. Транскрибировать один файл")
    print("2. Транскрибировать все WAV в папке voice/")
    
    choice = input("\nВыбор (1-2): ").strip()
    
    if choice == "1":
        # Один файл
        default_path = "voice/VO_Jane_Doe_Obtain_Agent_01.wav"
        print(f"\nПуть по умолчанию: {default_path}")
        custom_path = input("Или введите свой путь (Enter для использования по умолчанию): ").strip()
        
        audio_path = custom_path if custom_path else default_path
        transcribe_audio(audio_path)
        
    elif choice == "2":
        # Пакетная обработка
        folder = input("Папка (Enter для 'voice/'): ").strip() or "voice"
        batch_transcribe(folder)
    else:
        print("❌ Некорректный выбор")

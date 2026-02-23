import torch
import sounddevice as sd
import time
import speech_recognition as sr
import ollama
import sys
import numpy as np

# --- НАСТРОЙКИ ---
OLLAMA_MODEL = "gemma"   # Ваша модель в Ollama
SILERO_DEVICE = "cpu"     # Используем процессор (или 'cuda' если есть видеокарта NVIDIA)
SAMPLE_RATE = 48000       # Качество звука
SPEAKER = "xenia"         # Голоса: 'aidar', 'baya', 'kseniya', 'xenia', 'eugene'

print(">>> Загрузка модели голоса Silero TTS...")

# Загрузка модели Silero с torch hub
# v4_ru - самая актуальная русская модель
local_file = 'model.pt'
model_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='ru',
                              speaker='v4_ru')

model_tts.to(torch.device(SILERO_DEVICE))

print(">>> Голос загружен. Ассистент готов!")

# --- ПАМЯТЬ ДИАЛОГА ---
# Храним историю переписки здесь
# system message задает характер
messages_history = [
    {
        'role': 'system', 
        'content': 'Ты умный голосовой помощник. Ты говоришь на русском языке. Твои ответы должны быть разговорными, но не слишком длинными (максимум 2-3 предложения), так как их нужно озвучивать.'
    }
]

def speak_silero(text):
    """Озвучивает текст с помощью нейросети Silero"""
    if not text:
        return
    
    print(f"🔊 Ассистент: {text}")
    
    # Генерация аудио
    audio = model_tts.apply_tts(text=text,
                                speaker=SPEAKER,
                                sample_rate=SAMPLE_RATE)
    
    # Воспроизведение (sounddevice требует numpy array)
    sd.play(audio.numpy(), SAMPLE_RATE)
    sd.wait() # Ждем, пока договорит
    sd.stop()

def listen():
    """Слушает микрофон"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Слушаю...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            # Слушаем
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            # Распознаем (используем Google для простоты, можно заменить на Whisper)
            query = r.recognize_google(audio, language="ru-RU")
            print(f"👤 Вы: {query}")
            return query
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"Ошибка слуха: {e}")
            return None
        
def input_keyboard():
    """Альтернативный способ ввода через клавиатуру (для тестов)"""
    return input("\n👤 Введите текст (или 'стоп' для выхода): ")

def ask_ollama_with_memory(user_input):
    """Отправляет запрос с учетом истории диалога"""
    global messages_history
    
    # 1. Добавляем вопрос пользователя в историю
    messages_history.append({'role': 'user', 'content': user_input})
    
    # 2. Ограничиваем память (чтобы не перегрузить контекст)
    # Оставляем системный промпт [0] и последние 10 сообщений
    if len(messages_history) > 11:
        messages_history = [messages_history[0]] + messages_history[-10:]

    # 3. Отправляем ВСЮ историю в Ollama
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages_history)
        ai_answer = response['message']['content']
        
        # 4. Добавляем ответ ассистента в историю
        messages_history.append({'role': 'assistant', 'content': ai_answer})
        
        return ai_answer
    except Exception as e:
        return f"Произошла ошибка мозга: {e}"

# --- ГЛАВНЫЙ ЦИКЛ ---
def main():
    speak_silero("Привет! Я на связи и помню контекст беседы.")
    
    # Выбор режима ввода
    print("\n=== ВЫБЕРИТЕ РЕЖИМ ВВОДА ===")
    print("1. Голосовой ввод (микрофон)")
    print("2. Ввод с клавиатуры")
    
    while True:
        mode_choice = input("\nВыберите режим (1 или 2): ").strip()
        if mode_choice in ['1', '2']:
            break
        print("Пожалуйста, введите 1 или 2")
    
    use_keyboard = mode_choice == '2'
    
    while True:
        if use_keyboard:
            query = input_keyboard()
        else:
            query = listen()
        
        if query:
            # Проверка на выход
            if any(cmd in query.lower() for cmd in ["стоп", "хватит", "выключись", "пока"]):
                speak_silero("Хорошо, отключаюсь. До связи.")
                sys.exit()
            
            # Проверка на переключение режима
            if any(cmd in query.lower() for cmd in ["переключи режим", "голос", "клавиатура", "микрофон"]):
                use_keyboard = not use_keyboard
                mode = "клавиатуры" if use_keyboard else "голоса"
                speak_silero(f"Переключилась на режим {mode}")
                print(f"\n>>> Режим изменен на: {mode}")
                continue
            
            # Получение ответа от "умного" мозга
            answer = ask_ollama_with_memory(query)
            
            # Озвучивание
            speak_silero(answer)

if __name__ == "__main__":
    main()
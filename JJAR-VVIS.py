import speech_recognition as sr
import pyttsx3
import ollama
import sys
import whisper

# 1. Настройка Голоса (TTS)
engine = pyttsx3.init()
stt_model = whisper.load_model("base")
# Настройка скорости и голоса (по желанию)
rate = engine.getProperty('rate')
engine.setProperty('rate', 180) # Скорость речи

voices = engine.getProperty('voices')
# Попытка найти русский голос. Если не найдет - будет говорить стандартным
for voice in voices:
    if 'ru' in voice.id or 'Russian' in voice.name:
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    """Функция для озвучивания текста"""
    print(f"Ассистент: {text}")
    engine.say(text)
    engine.runAndWait()

# 2. Настройка Слуха (STT)
def listen_offline():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Слушаю...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        
        # Сохраняем аудио во временный файл
        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())
            
        # Распознаем через локальный Whisper
        result = stt_model.transcribe("temp.wav", fp16=False, language='ru')
        text = result['text']
        print(f"Вы: {text}")
        return text

# 3. Настройка Мозга (Ollama)
def ask_ollama(prompt):
    """Отправляет запрос в локальную Ollama"""
    try:
        response = ollama.chat(model='gemma', messages=[
            {
                'role': 'system',
                'content': 'Ты полезный голосовой ассистент. Отвечай кратко, емко и по-русски (не более 2-3 предложений).'
            },
            {
                'role': 'user',
                'content': prompt
            },
        ])
        return response['message']['content']
    except Exception as e:
        return f"Ошибка связи с Ollama: {e}"

# Основной цикл
def main():
    speak("Привет! Я готов к работе.")
    
    while True:
        query = listen_offline()
        
        if query:
            # Команды для выхода
            if query in ["стоп", "выход", "пока", "отключись"]:
                speak("До свидания!")
                sys.exit()
            
            # Генерация ответа
            answer = ask_ollama(query)
            speak(answer)

if __name__ == "__main__":
    main()
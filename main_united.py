import sys
import os
import platform
import webbrowser
import threading

import torch
import sounddevice as sd
import speech_recognition as sr
import ollama
import pyautogui
import keyboard

# --- НАСТРОЙКИ ---
OLLAMA_MODEL = "gemma"
SILERO_DEVICE = "cpu"
SAMPLE_RATE = 48000
SPEAKER = "xenia"

# --- МОДЕЛЬ ТТС ---
print(">>> Загрузка модели голоса Silero TTS... (если уже загружено в другом модуле, загрузка будет выполнена повторно)")
model_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='ru',
                              speaker='v4_ru')
model_tts.to(torch.device(SILERO_DEVICE))
print(">>> Голос загружен. Ассистент готов!")

# --- ПАМЯТЬ И ПРОМПТ ---
system_prompt = """
Ты — голосовой ассистент. Ты управляешь компьютером и отвечаешь по-русски.
Если пользователь просит выполнить действие из списка ниже, возвращай ТОЛЬКО специальный тег команды:
- Открыть браузер -> {{OPEN_BROWSER}}
- Открыть блокнот -> {{OPEN_NOTEPAD}}
- Сделать громче -> {{VOLUME_UP}}
- Сделать тише -> {{VOLUME_DOWN}}
Если действия нет — отвечай кратко (не больше 2 предложений).
"""

messages_history = [
    {'role': 'system', 'content': system_prompt}
]

# --- КОМАНДЫ (интеграция управления ОС) ---
def open_browser():
    webbrowser.open("https://www.google.com")
    return "Открываю браузер."

def open_notepad():
    if platform.system() == "Windows":
        os.system("start notepad")
    elif platform.system() == "Darwin":
        os.system("open -a TextEdit")
    else:
        os.system("gedit")
    return "Запускаю блокнот."

def volume_up():
    for _ in range(5):
        pyautogui.press("volumeup")
    return "Сделала громче."

def volume_down():
    for _ in range(5):
        pyautogui.press("volumedown")
    return "Сделала тише."

commands = {
    "{{OPEN_BROWSER}}": open_browser,
    "{{OPEN_NOTEPAD}}": open_notepad,
    "{{VOLUME_UP}}": volume_up,
    "{{VOLUME_DOWN}}": volume_down
}

is_running = True

# --- УТИЛИТЫ КЛАВИАТУРЫ ---
def toggle_exit():
    global is_running
    print("\n⌨️ Выход по Ctrl+Q...")
    is_running = False

def setup_keyboard_shortcuts():
    keyboard.add_hotkey('ctrl+q', lambda: toggle_exit())
    print("   Ctrl+Q - выйти")

# --- TTS / ASR ---
def speak_silero(text):
    if not text:
        return
    print(f"🔊 Ассистент: {text}")
    audio = model_tts.apply_tts(text=text, speaker=SPEAKER, sample_rate=SAMPLE_RATE)
    sd.play(audio.numpy(), SAMPLE_RATE)
    sd.wait()
    sd.stop()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Слушаю...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            query = r.recognize_google(audio, language="ru-RU")
            print(f"👤 Вы: {query}")
            return query
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"Ошибка слуха: {e}")
            return None

def input_keyboard():
    return input("\n👤 Введите текст (или 'стоп' для выхода): ")

# --- ВЗАИМОДЕЙСТВИЕ С OLLAMA ---
def ask_ollama_with_memory(user_input):
    global messages_history
    messages_history.append({'role': 'user', 'content': user_input})
    if len(messages_history) > 11:
        messages_history = [messages_history[0]] + messages_history[-10:]
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages_history)
        ai_answer = response['message']['content']
        # Не добавляем командные теги в историю
        if "{{" not in ai_answer:
            messages_history.append({'role': 'assistant', 'content': ai_answer})
        return ai_answer
    except Exception as e:
        return f"Произошла ошибка мозга: {e}"

def process_ai_command(response_text):
    clean_text = response_text.strip()
    if clean_text in commands:
        func = commands[clean_text]
        result_message = func()
        return result_message
    return response_text

# --- MAIN ---
def main():
    global is_running
    setup_keyboard_shortcuts()

    speak_silero("Системы управления активны. Жду приказов.")

    # Выбор режима ввода (голос/клавиатура)
    print("\n=== ВЫБЕРИТЕ РЕЖИМ ВВОДА ===")
    print("1. Голосовой ввод (микрофон)")
    print("2. Ввод с клавиатуры")
    use_keyboard = False
    while True:
        choice = input("Выберите режим (1 или 2): ").strip()
        if choice in ['1', '2']:
            use_keyboard = choice == '2'
            break
        print("Пожалуйста, введите 1 или 2")

    while is_running:
        if use_keyboard:
            query = input_keyboard()
        else:
            query = listen()

        if query:
            if any(cmd in query.lower() for cmd in ["стоп", "выход", "пока", "хватит"]):
                speak_silero("Отключаюсь. До связи.")
                is_running = False
                break

            # Переключение режима
            if any(cmd in query.lower() for cmd in ["переключи режим", "голос", "клавиатура", "микрофон"]):
                use_keyboard = not use_keyboard
                mode = "клавиатуры" if use_keyboard else "голоса"
                speak_silero(f"Переключилась на режим {mode}")
                print(f"\n>>> Режим изменен на: {mode}")
                continue

            # Общение с Ollama
            llm_response = ask_ollama_with_memory(query)

            # Проверяем, командный тег или обычный ответ
            final_answer = process_ai_command(llm_response)

            # Озвучиваем
            speak_silero(final_answer)

    print("Завершение работы.")


if __name__ == "__main__":
    main()

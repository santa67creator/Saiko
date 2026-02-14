import torch
import sounddevice as sd
import speech_recognition as sr
import ollama
import sys
import webbrowser       # Для браузера
import pyautogui        # Для клавиш (громкость и т.д.)
import os               # Для запуска программ
import platform         # Чтобы понять, Windows это или Mac/Linux
import keyboard         # Для контроля с клавиатуры
import threading        # Для параллельного отслеживания клавиш

# --- НАСТРОЙКИ ---
OLLAMA_MODEL = "gemma"
SILERO_DEVICE = "cpu"
SAMPLE_RATE = 48000
SPEAKER = "xenia"

# Загрузка Silero (как в прошлом шаге)
print(">>> Загрузка голоса...")
model_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='ru',
                              speaker='v4_ru')
model_tts.to(torch.device(SILERO_DEVICE))

# --- ФУНКЦИИ УПРАВЛЕНИЯ (РУКИ) ---
def open_browser():
    webbrowser.open("https://www.google.com")
    return "Открываю браузер."

def open_notepad():
    if platform.system() == "Windows":
        os.system("start notepad")
    elif platform.system() == "Darwin": # Mac
        os.system("open -a TextEdit")
    else: # Linux
        os.system("gedit") # или другой редактор
    return "Запускаю блокнот."

def volume_up():
    for _ in range(5): # Нажать 5 раз
        pyautogui.press("volumeup")
    return "Сделала громче."

def volume_down():
    for _ in range(5):
        pyautogui.press("volumedown")
    return "Сделала тише."

# Словарь доступных команд
# Ключ - это то, что вернет LLM. Значение - функция Python.
commands = {
    "{{OPEN_BROWSER}}": open_browser,
    "{{OPEN_NOTEPAD}}": open_notepad,
    "{{VOLUME_UP}}": volume_up,
    "{{VOLUME_DOWN}}": volume_down
}

# Флаг для остановки программы
is_running = True

# --- КЛАВИАТУРНЫЙ КОНТРОЛЬ ---
def execute_keyboard_command(func):
    """Выполняет команду с клавиатуры"""
    result = func()
    print(f"⌨️ Команда с клавиатуры: {result}")

def setup_keyboard_shortcuts():
    """Регистрирует горячие клавиши"""
    global is_running
    # Ctrl+B - открыть браузер    
    # Ctrl+Q - выйти из приложения
    keyboard.add_hotkey('ctrl+q', lambda: toggle_exit())
    
    print("   Ctrl+Q - выйти")

def toggle_exit():
    """Безопасно выходит из приложения"""
    global is_running
    print("\n⌨️ Выход по Ctrl+Q...")
    is_running = False

# --- ПАМЯТЬ И ПРОМПТ ---
# Самое важное: Инструкция для LLM
system_prompt = """
Ты — голосовой ассистент Джарвис. Ты управляешь компьютером.
Твоя задача: определять намерение пользователя.

ЕСЛИ пользователь просит выполнить действие из списка ниже, ты ДОЛЖЕН вернуть ТОЛЬКО специальный тег команды и ничего больше:
- Открыть браузер/интернет -> {{OPEN_BROWSER}}
- Открыть блокнот/заметки -> {{OPEN_NOTEPAD}}
- Сделать громче/увеличить звук -> {{VOLUME_UP}}
- Сделать тише/уменьшить звук -> {{VOLUME_DOWN}}

ЕСЛИ запроса на действие нет, просто отвечай на вопрос кратко и по-русски (максимум 2 предложения).
"""

messages_history = [
    {'role': 'system', 'content': system_prompt}
]

# --- ФУНКЦИИ ЯДРА ---
def speak_silero(text):
    if not text: return
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
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            query = r.recognize_google(audio, language="ru-RU")
            print(f"👤 Вы: {query}")
            return query
        except:
            return None

def process_ai_command(response_text):
    """Проверяет, есть ли команда в ответе AI"""
    clean_text = response_text.strip()
    
    # Проверяем, есть ли текст в нашем словаре команд
    if clean_text in commands:
        func = commands[clean_text] # Берем функцию
        result_message = func()     # Выполняем её
        return result_message       # Возвращаем фразу "Открываю..."
    
    # Если команды нет, возвращаем обычный текст ответа
    return response_text

def ask_ollama(user_input):
    global messages_history
    messages_history.append({'role': 'user', 'content': user_input})
    
    if len(messages_history) > 10:
        messages_history = [messages_history[0]] + messages_history[-9:]

    response = ollama.chat(model=OLLAMA_MODEL, messages=messages_history)
    ai_answer = response['message']['content']
    
    # Не добавляем команды в историю диалога, чтобы не сбивать модель в будущем
    if "{{" not in ai_answer:
        messages_history.append({'role': 'assistant', 'content': ai_answer})
        
    return ai_answer

# --- MAIN ---
def main():
    global is_running
    
    # Активируем контроль с клавиатуры
    setup_keyboard_shortcuts()
    
    speak_silero("Системы управления активны. Жду приказов.")
    
    while is_running:
        query = listen()
        
        if query:
            if any(cmd in query.lower() for cmd in ["стоп", "выход", "пока"]):
                speak_silero("Отключаюсь.")
                is_running = False
                break
            
            # 1. Спрашиваем LLM
            llm_response = ask_ollama(query)
            
            # 2. Проверяем, команда это или текст
            # Если команда -> выполняется действие и возвращается фраза о выполнении
            # Если текст -> возвращается сам текст
            final_answer = process_ai_command(llm_response)
            
            # 3. Озвучиваем результат
            speak_silero(final_answer)

if __name__ == "__main__":
    main()
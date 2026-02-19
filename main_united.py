import sys
import os
import platform
import webbrowser
import threading
import queue

import torch
import sounddevice as sd
import speech_recognition as sr
import ollama
import pyautogui
import keyboard
import numpy as np

# --- SETTINGS ---
OLLAMA_MODEL = "gemma"
SILERO_DEVICE = "cpu"
SAMPLE_RATE = 48000
SPEAKER = "en_0"

# --- TTS MODEL ---
print(">>> Loading Silero TTS voice model... (if already loaded in another module, it will be reloaded)")
model_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker='v3_en')
model_tts.to(torch.device(SILERO_DEVICE))
print(">>> Voice loaded. Assistant ready!")

# --- AUDIO STREAMING CLASS ---
class AudioStreamer:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_playing = threading.Event()
        
    def audio_callback(self, outdata, frames, time, status):
        """Callback function for continuous audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        try:
            # Try to get audio chunk from queue
            data = self.audio_queue.get_nowait()
            
            # Handle data size
            if len(data) < len(outdata):
                # Pad with zeros if not enough data
                outdata[:len(data), 0] = data
                outdata[len(data):, 0] = 0
                self.is_playing.clear()  # Mark as finished
            else:
                # Fill output buffer
                outdata[:, 0] = data[:len(outdata)]
                # Put remaining data back in queue
                remaining = data[len(outdata):]
                if len(remaining) > 0:
                    self.audio_queue.put(remaining)
                else:
                    self.is_playing.clear()
                    
        except queue.Empty:
            # No data available, output silence
            outdata[:, 0] = 0
            self.is_playing.clear()
    
    def start(self):
        """Start the continuous audio stream"""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=2048,
            dtype='float32'
        )
        self.stream.start()
        print("🔊 Audio stream started")
    
    def speak(self, text):
        """Queue audio for playback"""
        if not text:
            return
        
        print(f"🔊 Assistant: {text}")
        
        # Generate audio
        audio = model_tts.apply_tts(text=text, speaker=SPEAKER, sample_rate=self.sample_rate)
        audio_data = audio.numpy().astype(np.float32)
        
        # Add to queue
        self.is_playing.set()
        self.audio_queue.put(audio_data)
        
   
    def wait_until_done(self):
        """Wait until all audio has been played"""
        while self.is_playing.is_set() or not self.audio_queue.empty():
            threading.Event().wait(0.1)
    
    def stop(self):
        """Stop the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("🔊 Audio stream stopped")

# Initialize global audio streamer
audio_streamer = AudioStreamer(SAMPLE_RATE)

# --- MEMORY AND PROMPT ---
system_prompt = """
You are a voice assistant. You control the computer and respond in English.
If the user asks to perform an action from the list below, return ONLY the special command tag:
- Open browser -> {{OPEN_BROWSER}}
- Open notepad -> {{OPEN_NOTEPAD}}
- Increase volume -> {{VOLUME_UP}}
- Decrease volume -> {{VOLUME_DOWN}}
If there is no action to perform — reply briefly (no more than 2 sentences).
"""

messages_history = [
    {'role': 'system', 'content': system_prompt}
]

# --- COMMANDS (OS control integration) ---
def open_browser():
    webbrowser.open("https://www.google.com")
    return "Opening browser."

def open_notepad():
    if platform.system() == "Windows":
        os.system("start notepad")
    elif platform.system() == "Darwin":
        os.system("open -a TextEdit")
    else:
        os.system("gedit")
    return "Launching notepad."

def volume_up():
    for _ in range(5):
        pyautogui.press("volumeup")
    return "Increased volume."

def volume_down():
    for _ in range(5):
        pyautogui.press("volumedown")
    return "Decreased volume."

commands = {
    "{{OPEN_BROWSER}}": open_browser,
    "{{OPEN_NOTEPAD}}": open_notepad,
    "{{VOLUME_UP}}": volume_up,
    "{{VOLUME_DOWN}}": volume_down
}

is_running = True

# --- KEYBOARD UTILITIES ---
def toggle_exit():
    global is_running
    print("\n⌨️ Exit with Ctrl+Q...")
    is_running = False

def setup_keyboard_shortcuts():
    keyboard.add_hotkey('ctrl+q', lambda: toggle_exit())
    print("   Ctrl+Q - exit")

# --- TTS / ASR ---
def speak_silero(text):
    """Use the audio streamer instead of direct sd.play"""
    audio_streamer.speak(text)
    audio_streamer.wait_until_done()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            query = r.recognize_google(audio, language="en-US")
            print(f" You: {query}")
            return query
        except sr.UnknownValueError:
            return None
        except Exception as e:
            print(f"Listening error: {e}")
            return None

def input_keyboard():
    return input("\n👤 Enter text (or 'stop' to exit): ")

# --- INTERACTION WITH OLLAMA ---
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
        return f"An error occurred: {e}"


#--- MAIN MATH LOGIC ---
def pronounce_number_in_text(text):
    """Convert all digits in text to spoken words, including decimals.

    Examples:
        12 -> one two
        3.5 -> three point five
        -4.2 -> minus four point two
    """
    number_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
    }
    result_chars = []
    for ch in text:
        if ch in number_words:
            result_chars.append(number_words[ch])
        elif ch == ".":
            result_chars.append("point")
        elif ch == "-":
            result_chars.append("minus")
        else:
            # keep other characters (e.g. spaces)
            result_chars.append(ch)
    # join and collapse any duplicate spaces that may have been introduced
    result = " ".join(result_chars).replace("  ", " ")
    return result.strip()

def calculate_math(query):
    """Detect and calculate math expressions with pronounced numbers"""
    import re
    # Pattern for simple math: "what is X + Y", "how much is X - Y", "X + Y", etc.
    math_patterns = [
        r'(?:what is|how much is|calculate|compute)\s+([\d+\-*/ ().]+)(?:\s*[?])?',
        r'^([\d+\-*/ ().]+)$',
        r'([\d+\-*/ ().]+)\s*[=]?$'
    ]
    
    for pattern in math_patterns:
        match = re.search(pattern, query.lower())
        if match:
            expression = match.group(1).strip()
            # Basic validation - only allow digits, operators, and spaces
            if re.match(r'^[\d+\-*/.() ]+$', expression):
                try:
                    result = eval(expression)
                    # Format the answer
                    if isinstance(result, float):
                        # Round to avoid long floating-point representations
                        result = round(result, 6)
                        if result.is_integer():
                            result = int(result)
                    # Pronounce the result
                    pronounced_result = pronounce_number_in_text(str(result))
                    return f"The answer is {pronounced_result}"
                except:
                    return None
    return None

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
    
    # Start the audio stream
    audio_streamer.start()

    speak_silero("Control systems active. Awaiting commands.")

    # Выбор режима ввода (голос/клавиатура)
    print("\n=== SELECT INPUT MODE ===")
    print("1. Voice input (microphone)")
    print("2. Keyboard input")
    use_keyboard = False
    while True:
        choice = input("Choose mode (1 or 2): ").strip()
        if choice in ['1', '2']:
            use_keyboard = choice == '2'
            break
        print("Please enter 1 or 2")

    try:
        while is_running:
            if use_keyboard:
                query = input_keyboard()
            else:
                query = listen()

            if query:
                if any(cmd in query.lower() for cmd in ["stop", "exit", "bye", "enough"]):
                    speak_silero("Shutting down. Goodbye.")
                    is_running = False
                    break

                # Переключение режима
                if any(cmd in query.lower() for cmd in ["switch mode", "voice", "keyboard", "microphone"]):
                    use_keyboard = not use_keyboard
                    mode = "keyboard" if use_keyboard else "voice"
                    speak_silero(f"Switched to {mode} mode")
                    print(f"\n>>> Mode changed to: {mode}")
                    continue

                # Check for math expressions first
                math_result = calculate_math(query)
                if math_result:
                    final_answer = math_result
                    print(f"🧮 Math: {query} = {math_result}")
                else:
                    # Общение с Ollama
                    llm_response = ask_ollama_with_memory(query)

                    # Проверяем, командный тег или обычный ответ
                    final_answer = process_ai_command(llm_response)

                # Speak the response
                speak_silero(final_answer)

    finally:
        # Clean shutdown
        audio_streamer.stop()
        print("Shutting down.")


if __name__ == "__main__":
    main()
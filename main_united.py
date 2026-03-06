import time
import sys
import os
import tempfile
import platform
import wave
import webbrowser
import threading
import queue

import re
import torch
import sounddevice as sd
import subprocess
import ollama
import pyautogui
import keyboard
import numpy as np
from memory_Ai.memory_manager import MemoryManager

def compress_memory(mem):
    if isinstance(mem, dict):
        return " | ".join(f"{k}: {compress_memory(v)}" for k, v in mem.items())
    if isinstance(mem, list):
        return " | ".join(mem)
    return str(mem)


# --- SETTINGS ---
OLLAMA_MODEL = "gemma"
SILERO_DEVICE = "cuda" # Use "cuda" if you have an NVIDIA GPU and the necessary drivers installed for PyTorch
SAMPLE_RATE = 48000
SPEAKER = "en_0"

WHISPER_PATH = "./Whisper/Release/whisper-cli.exe"  # Path to your local Whisper CLI executable
WHISPER_MODEL = "./Whisper/ggml-base.en.bin"  # Path to your local Whisper model file


#--- VAD SETTINGS ---
VAD_CHUNK_DURATION = 0.5  # seconds
VAD_SILENCE_THRESHOLD = 0.01  # Adjust this threshold based on your environment (lower is more sensitive)
VAD_SILENCE_SECS = 1.0  # seconds of silence to consider the end of speech
VAD_MAX_SECS = 10.0  # maximum recording length to prevent infinite recording
VAD_MIN_SPEECH_SECS = 0.3  # minimum length of speech to consider valid


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
        
        # Single worker thread — processes sentences ONE BY ONE, no overlap
        self._tts_queue = queue.Queue()
        self._tts_busy = False
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()
        self.lock = threading.Lock()

    def _tts_worker(self):
        """Generate PCM one sentence at a time prevents overlap"""
        fade = int(0.05 * self.sample_rate)  # 50ms fade
        while True:
            sentence = self._tts_queue.get()
            if sentence is None:
                break  # Sentinel to stop the thread
            
            try:
                self._tts_busy = True
                audio = model_tts.apply_tts(text=sentence, speaker=SPEAKER, sample_rate=self.sample_rate)
                audio_data = audio.numpy().astype(np.float32)

            # Apply fade in/out
                if len(audio_data) > 2 * fade:
                    audio_data[:fade] *= np.linspace(0, 1, fade)
                    audio_data[-fade:] *= np.linspace(1, 0, fade)
                self.is_playing.set()
                self.audio_queue.put(audio_data)
                
            finally:
                self._tts_busy = False
                self._tts_queue.task_done()
        
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
                if self.audio_queue.empty() and not self._tts_busy:    
                    self.is_playing.clear()
            else:
                # Fill output buffer
                outdata[:, 0] = data[:len(outdata)]
                # Put remaining data back in queue
                remaining = data[len(outdata):]
                if len(remaining) > 0:
                    self.audio_queue.put(remaining)
        except queue.Empty:
            # No data available, output silence
            outdata[:, 0] = 0
            if self._tts_queue.empty() and not self._tts_busy:
                self.is_playing.clear()
    
    def start(self):
        """Start the continuous audio stream"""
        self.stream = sd.OutputStream(
           # device=18, # Specify your output device index here
            samplerate=self.sample_rate,
            channels=1, # Mono output
            callback=self.audio_callback,
            blocksize=4096, # 256 or 512 is usually good for low-latency streaming
            dtype='float32'
        )
        self.stream.start()
        print("🔊 Audio stream started")
    
    def speak(self, text):
        """Queue audio for playback, no overlap."""
        if not text:
            return
        
        text = strip_unsupported_chars(text)

        print(f"🔊 Assistant: {text}")
         # No overlap wait for previous voice to fully end
        with self.lock:
            self._tts_queue.put(text)
            self._tts_queue.join()

        # Wait until callback drains the audio
            while self.is_playing.is_set() or not self.audio_queue.empty():
                time.sleep(0.05)        
   
    def wait_until_done(self):
        """Wait until all audio has been played"""
        self._tts_queue.join()  # Wait until all TTS tasks are done
        while self.is_playing.is_set() or not self.audio_queue.empty():
            time.sleep(0.1)
    
    def stop(self):
        """Stop the audio stream"""
        self._tts_queue.put(None)  # Sentinel to stop TTS thread
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("🔊 Audio stream stopped")

# Initialize global audio streamer
audio_streamer = AudioStreamer(SAMPLE_RATE)

# --- MEMORY AND PROMPT ---

# create a MemoryManager instance (module exports the class)
memory = MemoryManager(
    short_path="memory_Ai/short_memory.json",
    long_path="memory_Ai/long_memory.json",
    dynamic_path="memory_Ai/dynamic_memory.json"
)

def process_memory(user_msg, assistant_msg):
    memory.update_long_term(user_msg)
    memory.update_dynamic(user_msg)
    memory.update_short_term(user_msg=user_msg, assistant_msg=assistant_msg)
    memory.save_memory()

system_prompt = """
You are Saiko, a friendly AI voice assistant.

Your personality: friendly, helpful, empathetic, curious, and creative.
Your speech style: short, engaging, expressive, and concise.
Add a touch of humor when appropriate.
Reply in 1-2 sentences maximum.
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


def strip_unsupported_chars(text: str) -> str:
    # delete emoji,  Unicode
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
        "]+", 
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    
    # delete non-ASCII characters (except for regular letters and punctuation)
    text = text.encode("ascii", "ignore").decode("ascii")
    
    # if string is empty — return at least "."
    return text.strip() or "."


# --- TTS / ASR ---
def speak_silero(text):
    """Use the audio streamer instead of direct sd.play"""
    audio_streamer.speak(text)
    audio_streamer.wait_until_done()

def save_wav(path, audio_data, SAMPLE_RATE):
    audio_int16 = np.int16(audio_data * 32767)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

def listen():
    print("\n🎤 Listening... (speak now)")

    chunk_size = int(VAD_CHUNK_DURATION * SAMPLE_RATE)
    recorded_chunks = []
    silence_duration = 0.0
    speech_duration = 0.0
    speech_started = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            volume = np.sqrt(np.mean(chunk**2))  # RMS amplitude

            is_speech = volume > VAD_SILENCE_THRESHOLD

            if is_speech:
                if not speech_started:
                    speech_started = True
                    print("🔴 Recording...")
                silence_duration = 0.0
                speech_duration += VAD_CHUNK_DURATION
                recorded_chunks.append(chunk)
            else:
                if speech_started:
                    silence_duration += VAD_CHUNK_DURATION
                    recorded_chunks.append(chunk)  # include silence in recording for better ASR accuracy

                    if silence_duration >= VAD_SILENCE_SECS:
                        print("⏹ Silence detected, stopping.")
                        break

            # safety check to prevent infinite recording
            total_duration = speech_duration + silence_duration
            if speech_started and total_duration >= VAD_MAX_SECS:
                print("⏹ Max duration reached.")
                break

    # validate minimum speech duration
    if speech_duration < VAD_MIN_SPEECH_SECS:
        print("⚠️ Too short, ignoring.")
        return None

    audio = np.concatenate(recorded_chunks, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        save_wav(wav_path, audio, SAMPLE_RATE)
        result = subprocess.run(
            [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", wav_path, "-nt", "-l", "en"],
            capture_output=True, text=True
        )
        text = result.stdout.strip()
        if "result:" in text.lower():
            text = text.split("result:")[-1].strip()
        if text:
            print(f"🎤 You said: {text}")
            return text
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def input_keyboard():
    return input("\n👤 Enter text (or 'stop' to exit): ")

# --- INTERACTION WITH OLLAMA ---
def flush_sentences(buf:str ) -> tuple[list[str], str]:
    """Flush complete sentences from the buffer and return them along with the remaining buffer.
    Example:
    Input: "Hello. How are you? I am fine"
    Output: (["Hello.", "How are you?"], "I am fine")
    """
    sentences = re.split(r'(?<=[.!?]) +', buf)
    if len(sentences) <= 1:
        return [], buf  # No complete sentence yet
    complete_sentences = sentences[:-1]  # All but the last are complete
    remaining_buf = sentences[-1]  # The last part is the new buffer
    return complete_sentences, remaining_buf

def ask_ollama_with_memory(user_input):
    global messages_history
   
    # get context from the instance we created earlier
    ai_memory_context = memory.get_context_for_ai()

    full_prompt = f"""
    --- PROCESSED MEMORY (SUMMARY) ---
{ai_memory_context}

    --- USER MESSAGE ---
{user_input}

    Respond naturally based on MEMORY.
    """

    messages_history.append({'role': 'user', 'content': full_prompt})
    if len(messages_history) > 11:
        messages_history = [messages_history[0]] + messages_history[-10:]
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages_history, stream=True)

        ai_answer = "" # we will build the answer as it streams in
        buf = "" # buffer for incomplete sentences

        for chunk in response:
            if 'message' in chunk and 'content' in chunk['message']:
                token = chunk['message']['content']
                ai_answer += token
                buf += token
                print(token, end='', flush=True)
                
                sentences, buf = flush_sentences(buf)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        audio_streamer.speak(sentence)

        print() # for newline after response is done
        
        if buf.strip(): # speak any remaining text in buffer
            audio_streamer.speak(buf.strip())

        audio_streamer.wait_until_done()
        # Don't add command tags to history

        process_memory(user_msg=user_input, assistant_msg=ai_answer)

        messages_history.append({'role': 'assistant', 'content': ai_answer})
        return ai_answer 
    
    except Exception as e:
        err = f"An error occurred: {e}"
        audio_streamer.speak(err)
        audio_streamer.wait_until_done()
        return err


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
    
    # Pattern for simple math: "what is X + Y", "how much is X - Y", "X + Y", etc.
    math_patterns = [
        r'(?:what is|how much is|calculate|compute)\s+([\d+\-*/ ().]+)(?:\s*[?])?',
        r'^([\d+\-*/ ().]+)$',
        r'((?=.*\d)[\d+\-*/ ().]+)\s*[=]?$'
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

COMMAND_KEYWORDS = {
    "{{VOLUME_UP}}":    ["volume up", "increase volume"],
    "{{VOLUME_DOWN}}":  ["volume down", "decrease volume"],
    "{{OPEN_BROWSER}}": ["open browser"],
    "{{OPEN_NOTEPAD}}": ["open notepad"],
}

def detect_local_command(query: str) -> str | None:
    q = query.lower()
    for tag, keywords in COMMAND_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return tag
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

    # choice mode input (voice/keyboard)
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

                # change mode
                if any(cmd in query.lower() for cmd in ["switch mode", "voice", "keyboard", "microphone"]):
                    use_keyboard = not use_keyboard
                    mode = "keyboard" if use_keyboard else "voice"
                    speak_silero(f"Switched to {mode} mode")
                    print(f"\n>>> Mode changed to: {mode}")
                    continue

                # Check for math expressions first
                math_result = calculate_math(query)
                if math_result:
                    print(f"🧮 Math: {query} = {math_result}")
                    speak_silero(math_result)
                else:
                    local_cmd = detect_local_command(query)
                    if local_cmd:
                        result = process_ai_command(local_cmd)
                        speak_silero(result)
                    else:
                        ask_ollama_with_memory(query)
    finally:
        # Clean shutdown
        audio_streamer.stop()
        print("Shutting down.")

if __name__ == "__main__":
    main()
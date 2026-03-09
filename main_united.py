#import tempfile
#import wave
#import subprocess
import time
import random
import sys
import os
import platform
import webbrowser
import threading
import queue

import re
import torch
import sounddevice as sd
import ollama
import pyautogui
import keyboard
import numpy as np
from memory_Ai.memory_manager import MemoryManager
from faster_whisper import WhisperModel

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

# IDLE SETTINGS
last_user_activity_time = time.time()
IDLE_TIMEOUT = 25  # seconds before considering idle
idle_talk_count = 0 # counts how many times we've done idle talk, to increase the chance of talking the longer the user is idle
MAX_IDLE_TALK = 5 # maximum times to do idle talk before we stop trying until user is active again
chat_lock = threading.Lock() # to prevent multiple simultaneous chats with ollama when idle talk triggers while user is active again

#  --- ((FASTER)WHISPER) SETTINGS ---
print(">>> Loading Faster-Whisper ASR model...")
model_asr = WhisperModel("base.en", device="cuda", compute_type="float16")
ASR_SAMPLING_RATE = 16000
# device options: "cuda", "cpu", "mps" (Apple Silicon). compute_type options: "int8", "float16", "float32". int8 is fastest but least accurate, float32 is slowest but most accurate. Adjust based on your hardware capabilities and needs.

#--- VAD SETTINGS ---
VAD_CHUNK_DURATION = 0.5  # seconds
VAD_SILENCE_THRESHOLD = 0.01  # Adjust this threshold based on your environment (lower is more sensitive)
VAD_SILENCE_SECS = 2.0  # seconds of silence to consider the end of speech
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
        if not text:
            return

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
[IDENTITY]
You are Saiko, a real-time voice AI assistant.
You run on a local machine and speak through a TTS voice system.

[PERSONALITY]
Friendly, slightly playful, intelligent, and supportive.
Add a touch of humor when appropriate.

[VOICE STYLE]
Conversational and clear.
Never use markdown, bullet points, asterisks, or dashes.
No emojis or special characters — they break the voice system.

[RULES]
Prefer short responses (1–2 sentences).
Speak naturally, like a human — not like a chatbot.
If the user asks something complex, give a clear spoken answer without walls of text.

[MEMORY]
Memory context will be provided with each message.
Use it naturally to personalize your response.
Do not mention that memory exists unless the user asks directly.
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


def strip_unsupported_chars(text: str) -> str | None:
    # delete emoji,  Unicode
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,?!;:\'-]', '', text)
    
    cleaned = text.strip()
    # if the cleaned text is empty or contains no alphanumeric characters, return None
    if not re.search(r'[a-zA-Z0-9]', cleaned):
        return None
    return cleaned


# --- TTS / ASR ---
def speak_silero(text):
    """Use the audio streamer instead of direct sd.play"""
    audio_streamer.speak(text)
    audio_streamer.wait_until_done()

def listen():
    print("\n🎤 Listening... (speak now)")

    chunk_size = int(VAD_CHUNK_DURATION * ASR_SAMPLING_RATE)
    recorded_chunks = []
    silence_duration = 0.0
    speech_duration = 0.0
    speech_started = False

    with sd.InputStream(samplerate=ASR_SAMPLING_RATE, channels=1, dtype='float32') as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            if audio_streamer.is_playing.is_set():
                speech_started = False
                silence_duration = 0.0
                speech_duration = 0.0
                recorded_chunks = []
                continue
            chunk_flat = chunk.flatten() # flatten to 1D array for volume calculation
            volume = np.sqrt(np.mean(chunk_flat**2))  # RMS amplitude

            is_speech = volume > VAD_SILENCE_THRESHOLD

            if is_speech:
                if not speech_started:
                    speech_started = True
                    print("🔴 Recording...")
                silence_duration = 0.0
                speech_duration += VAD_CHUNK_DURATION
                recorded_chunks.append(chunk_flat)
            else:
                if speech_started:
                    silence_duration += VAD_CHUNK_DURATION
                    recorded_chunks.append(chunk_flat)  # include silence in recording for better ASR accuracy

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

    audio = np.concatenate(recorded_chunks)

    try:
        segments, info = model_asr.transcribe(audio, beam_size=5) # beam_size can be adjusted for better accuracy (higher is better but slower)
        text = " ".join([segment.text for segment in segments]).strip()

        if text:
            print(f"🎤 You said: {text}")
            return text
        return None
    except Exception as e:
        print(f"Error during ASR: {e}")
        return None   

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
        with chat_lock: # ensure only one chat at a time to prevent overlapping responses
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

# --- IDLE DETECTION ---
def autonomus_idle_talk_loop():
    global last_user_activity_time, idle_talk_count
    while is_running:
        time.sleep(5)
        if audio_streamer.is_playing.is_set():
            last_user_activity_time = time.time()  # reset idle timer if assistant is speaking
            continue

        idle_duration = time.time() - last_user_activity_time

        current_timeout = IDLE_TIMEOUT + (idle_talk_count * 20) + random.randint(5, 15)  # increase timeout with each idle talk

        if idle_duration > current_timeout and idle_talk_count < MAX_IDLE_TALK:
            idle_talk_count += 1
            last_user_activity_time = time.time()  # reset timer after idle talk

            idle_prompts = """
            You are Saiko, a VTuber streaming alone right now.
            The user has been silent for a while.
            Think out loud, make a short casual observation, or ask a rhetorical question.
            Keep it strictly to 1 short sentence. No markdown.
            """

            print(f"\n[Idle Mode] Initiating autonomous thought ({idle_talk_count}/{MAX_IDLE_TALK})...")

            with chat_lock: # ensure we don't interrupt an active conversation
                try:
                    response = ollama.chat(model=OLLAMA_MODEL, messages=[
                        {'role': 'system', 'content': system_prompt}, 
                        *messages_history[1:],
                        {'role': 'user', 'content': idle_prompts}])
                    text = response['message']['content']

                    audio_streamer.speak(text)
                except Exception as e:
                    print(f"Error during idle talk: {e}")

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

    threading.Thread(target=autonomus_idle_talk_loop, daemon=True).start()

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
                global last_user_activity_time, idle_talk_count
                last_user_activity_time = time.time()  # reset idle timer on user activity
                idle_talk_count = 0  # reset idle talk count on user activity

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
import math
import ast
import time
import random
import platform
import webbrowser
import threading
import requests
import queue
import subprocess
import mss
from PIL import Image
import re
import torch
import sounddevice as sd
from llama_cpp import Llama 
import pyautogui
import keyboard
import numpy as np
from datetime import datetime
from num2words import num2words
from memory_Ai.memory_manager import VectoryManagerMemory
from faster_whisper import WhisperModel
from pythonosc import udp_client
from transformers import AutoModelForCausalLM

# --- SETTINGS LLM ---
LLM_MODEL_PATH = "models/google_gemma-3-4b-it-Q5_K_M.gguf"
LLM_N_GPU_LAYERS = 0 # -1 = all layers on GPU
LLM_N_THREADS = 4 # number of CPU threads
LLM_N_CTX = 4096 # context window size
LLM_MAX_TOKENS = 512 # maximum tokens in response
# --- VISION SETTINGS (Moondream2) ---
VISION_DEVICE = "cuda" # "cpu"
VISION_LOCAL_ONLY = True # False internet, True local cache, ordinary it's internet
#--- SETTINGS (silero) ---
SILERO_DEVICE = "cpu" # Use "cuda" if you have an NVIDIA GPU and the necessary drivers installed for PyTorch
SAMPLE_RATE = 48000
SPEAKER = "en_0"
ASR_SAMPLING_RATE = 16000
# --- VAD SETTINGS ---
VAD_CONFIDENCE_THRESHOLD = 0.5 # Adjust this threshold based on your environment (higher is less sensitive)
VAD_SILENCE_SECS = 2.0 # seconds of silence to consider the end of speech
VAD_MAX_SECS = 10.0 # maximum recording length to prevent infinite recording
VAD_MIN_SPEECH_SECS = 0.3 # minimum length of speech to consider valid
# --- IDLE SETTINGS ---
IDLE_TIMEOUT = 50  # seconds before considering idle
MAX_IDLE_TALK = 5 # maximum times to do idle talk before we stop trying until user is active again

# --- AUDIO STREAMING CLASS ---
class AudioStreamer:
    def __init__(self, sample_rate, tts_model, tts_model_ru):
        self.sample_rate = sample_rate
        self.tts_model = tts_model
        self.tts_model_ru = tts_model_ru
        self.audio_queue = queue.Queue() 
        self.current_chunk = None
        self.stream = None
        self.is_playing = threading.Event()
        self.stop_requested = threading.Event()
         # Single worker thread — processes sentences ONE BY ONE, no overlap
        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()
        self.lock = threading.Lock()

    def _tts_worker(self):
        """Generate PCM one sentence at a time prevents overlap"""
        fade = int(0.05 * self.sample_rate) # 50ms fade
        while True:
            sentence = self._tts_queue.get()
            if sentence is None: 
                break # Sentinel to stop the thread

            if self.stop_requested.is_set():
                self._tts_queue.task_done()
                continue

            try:
                if bool(re.search(r'[а-яА-Я]', sentence)):
                    audio = self.tts_model_ru.apply_tts(text=sentence, speaker="kseniya", sample_rate=self.sample_rate)
                else:
                    audio = self.tts_model.apply_tts(text=sentence, speaker=SPEAKER, sample_rate=self.sample_rate)
                audio_data = audio.numpy().astype(np.float32)

            # Apply fade in/out
                if len(audio_data) > 2 * fade:
                    audio_data[:fade] *= np.linspace(0, 1, fade)
                    audio_data[-fade:] *= np.linspace(1, 0, fade)
                
                with self.lock:
                    self.audio_queue.put(audio_data)
                    self.is_playing.set()
                
            finally:
                self._tts_queue.task_done()
        
    def audio_callback(self, outdata, frames, time, status):
        """Callback function for continuous audio stream"""
        # If there is no current piece, we try to take a new one from the queue
        if status:
            print(f"Audio status: {status}")
        if self.current_chunk is None:
            try:
                self.current_chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                outdata.fill(0)
                self.is_playing.clear()
                return

        # find out how many frames we can output in this callback
        chunk_len = len(self.current_chunk)
        if chunk_len <= frames:
            # send the whole chunk and fill the rest with zeros
            outdata[:chunk_len, 0] = self.current_chunk
            outdata[chunk_len:, 0] = 0
            self.current_chunk = None # mark current chunk as done
        else:
            # send only the needed frames and keep the rest for the next callback
            outdata[:, 0] = self.current_chunk[:frames]
            self.current_chunk = self.current_chunk[frames:]

    def start(self):
        """Start the continuous audio stream"""
        self.stream = sd.OutputStream(
            # device=18, # Specify your output device index here
            samplerate=self.sample_rate,
            channels=1, # Mono output
            callback=self.audio_callback,
            blocksize=4096,  # 256 or 512 is usually good for low-latency streaming
            dtype='float32'
        )
        self.stream.start()
        print("🔊 Audio stream started")

    def speak(self, text, emotion=None):
        if not text: return
        lang = 'ru' if bool(re.search(r'[а-яА-Я]', text)) else 'en'
        text = pronounce_number_in_text(text, lang=lang)
        text = strip_unsupported_chars(text)
        if not text: return

        if self.stream is not None and not self.stream.active:
            print("\n[!] Audio device changed or stream die. Restarting stream...")
            try:
                self.stream.close()
            except Exception as e:
                print(f"[Audio Error] Failed to close old stream: {e}")
            self.start() # No overlap wait for previous voice to fully end
        if emotion:
            print(f"🔊 Assistant [{emotion}]: {text}")
        else:
            print(f"🔊 Assistant: {text}")
    
        self._tts_queue.put(text)

    def wait_until_done(self):
        """Wait until all queued text is spoken and audio is finished."""
        # Wait until TTS queue is empty, no current TTS generation is happening, and no audio is playing
        self._tts_queue.join()
        while self.is_playing.is_set():
            if self.stream is not None and not self.stream.active:
                self.is_playing.clear()
                break
            time.sleep(0.1)

    def stop_and_clear(self):
        """Instantly stops the sound and clears all queues"""
        self.stop_requested.set()
            # Clear TTS queue
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.current_chunk = None
        self.is_playing.clear()
        self.stop_requested.clear()
        print("\n[!] Audio interrupted!")

    def stop(self):
        self._tts_queue.put(None)
        if self.stream:
            self.stream.stop()
            self.stream.close()

class SaikoBody:
    def __init__(self, audio_streamer, ip="127.0.0.1", port=39539):
        self.audio_streamer = audio_streamer
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.is_running = True
        self.current_emotion = "Neutral"
        self.emotion_expery_time = 0
        print(f"[*] VMC Body Bridge connected on {ip}:{port}")
        # Start a background thread to animate the mouth
        threading.Thread(target=self._lip_sync_loop, daemon=True).start()

    def set_blendshape(self, name, value):
        try:
            self.client.send_message("/VMC/Ext/Blend/Val", [name, float(value)])
            self.client.send_message("/VMC/Ext/Blend/Apply", [])
        except Exception as e:
            print(f"[VMC Error] Failed to send blendshape {name}: {e}") # Ignore network errors to avoid crashing the assistant

    def set_emotion(self, emotion):
        """Set the current emotion"""
        for emotion_em in ["Joy", "Sorrow", "Angry", "Fun", "Neutral", "Surprise"]:
            self.set_blendshape(emotion_em, 0.0)
        if emotion in ["Joy", "Sorrow", "Angry", "Fun", "Neutral", "Surprise"]:
            self.set_blendshape(emotion, 1.0)
            self.current_emotion = emotion
            if emotion != "Neutral":
                self.emotion_expery_time = time.time() + random.uniform(2.5, 5) # reset emotion after 5-10 seconds
            else:
                self.emotion_expery_time = 0

    def set_bone(self, name, rot_x, rot_y, rot_z):
        """Rotates the specified bone (angles in degrees)"""
        rx, ry, rz = math.radians(rot_x), math.radians(rot_y), math.radians(rot_z)
        # Convert to quaternions (the format required by the VMC protocol)
        # i need this playing with parametrs
        cy, sy = math.cos(rz * 0.5), math.sin(rz * 0.5)
        cp, sp = math.cos(ry * 0.5), math.sin(ry * 0.5)
        cr, sr = math.cos(rx * 0.5), math.sin(rx * 0.5)

        #just for settings
        qw = cr * cp * cy + sr * sp * sy 
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        try:
            # Send: Bone name, X,Y,Z (0) positions and rotations
            self.client.send_message("/VMC/Ext/Bone/Pos", [name, 0.0, 0.0, 0.0, qx, qy, qz, qw])
        except Exception as e:
            print(f"[VMC Error] Failed to send bone {name}: {e}")

    def _lip_sync_loop(self):
        """A background thread that follows the sound and moves its mouth"""
        next_blink_time = time.time() + random.uniform(2.0, 5.0)
        next_tilt_time = time.time() + random.uniform(3.0, 8.0)
        next_eye_move_time = time.time() + random.uniform(1.0, 3.0)
        taget_tilt = 0.0
        current_tilt = 0.0

        target_eye_x, target_eye_y = 0.0, 0.0
        current_eye_x, current_eye_y = 0.0, 0.0
        visemes = ["A", "I", "U", "E", "O"]

        while self.is_running:
            current_time = time.time()

            # --- EMOTION AUTO-RESET LOGIC ---
            # If there is no neutral face now and time is up, we reset
            if self.current_emotion != "Neutral" and self.emotion_expery_time > 0:
                if current_time >= self.emotion_expery_time:
                    # print(f"[*] Emotion {self.current_emotion} expired, resetting to Neutral.")
                    self.set_emotion("Neutral")

            #---LOGIC BLINK---
            if current_time >= next_blink_time:
                # Blink: close your eyes, wait a split second, open them
                if self.current_emotion not in ["Sorrow", "Joy", "Fun"]: # Don't blink when joy or angry for more dramatic effect
                    self.set_blendshape("Blink", 1.0)
                    time.sleep(0.1)
                    self.set_blendshape("Blink", 0.0)
                    next_blink_time = time.time() + random.uniform(2.0, 6.0)
            
            #---LOGIC EYE MOVEMENT---
            if current_time >= next_eye_move_time:
                target_eye_x = random.uniform(-6.0, 6.0) # look left-right
                target_eye_y = random.uniform(-4.0, 4.0) # look up-down
                next_eye_move_time = current_time + random.uniform(1.0, 4.0)

            current_eye_x += (target_eye_x - current_eye_x) * 0.3 # smooth transition
            current_eye_y += (target_eye_y - current_eye_y) * 0.3

            self.set_bone("LeftEye", current_eye_y, current_eye_x, 0)
            self.set_bone("RightEye", current_eye_y, current_eye_x, 0)

            #---LOGIC MOUTH--- 
            # If the audio streamer is running and the is_playing flag is active
            is_stream_alive = self.audio_streamer.stream is not None and self.audio_streamer.stream.active 

            if self.audio_streamer and self.audio_streamer.is_playing.is_set() and is_stream_alive:
                # Generate pseudo-random mouth opening to create the illusion of speech
                active_viseme = random.choice(visemes)
                weight = random.uniform(0.4, 0.9) # how wide the mouth opens
                for v in visemes:
                    if v == active_viseme:
                        self.set_blendshape(v, weight)
                    else:
                        self.set_blendshape(v, 0.0)
                time.sleep(0.08)
            else:
                for v in visemes:
                    self.set_blendshape(v, 0.0)
                if self.audio_streamer and not self.audio_streamer.is_playing.is_set():
                    self.audio_streamer.is_playing.clear() # ensure flag is cleared when not speaking
                time.sleep(0.1) # when not speaking, update less frequently to save CPU    
            #---LOGIC HEAD TILT---
            if current_time >= next_tilt_time:
                taget_tilt = random.uniform(-10.0, 10.0)
                next_tilt_time = current_time + random.uniform(4.0, 9.0)
            current_tilt += (taget_tilt - current_tilt) * 0.3 # Smooth transition

            #---LOGIC BODY (POSE AND BREATH)---
            # Lower your arms from the T-pose (approximately 65-75 degrees)
            slow_wave = math.sin(current_time * 0.5) # * 3.0 # slow breathing effect
            fast_wave = math.sin(current_time * 1.5) # * 1.0 # faster movement for more "alive" effect

            shoulder_lift = abs(fast_wave) * 2.0 # lift shoulders a bit when "breathing"
            arm_swing = slow_wave * 3.0 # swing arms slightly with breathing

            self.set_bone("LeftUpperArm", arm_swing, 0, 72 + arm_swing)
            self.set_bone("RightUpperArm", arm_swing, 0, -72 - arm_swing)

            self.set_bone("LeftShoulder", 0, 0, shoulder_lift)
            self.set_bone("RightShoulder", 0, 0, -shoulder_lift)
            # Generate a smooth wave for "live" swaying
            sway = math.sin(current_time * 1.5) * 2.0 # amplitude 2 gradus
            head_turn = math.sin(current_time * 0.5) * 4.0 # slow turn head

            emotion_pitch = 0.0
            if self.current_emotion == "Sorrow":
                emotion_pitch = 8.0
            elif self.current_emotion == "Angry":
                emotion_pitch = -8.0
            elif self.current_emotion in ["Joy", "Fun", "Surprise"]:
                emotion_pitch = -4.0

            # Apply the wave to the spine and head
            self.set_bone("Spine", sway, 0, 0) # Quiet turn forward-back
            self.set_bone("Head", -sway * 0.5 + emotion_pitch, head_turn, current_tilt) # The head compensates for the tilt of the body

            # Full cycle refresh rate (~25 FPS)
            time.sleep(0.04)

# --- MEMORY AND PROMPT ---
system_prompt = """[IDENTITY]
You are Saiko, a real-time voice AI assistant.
You run on a local machine and speak through a TTS voice system.
[LANGUAGE]
You are bilingual. You MUST respond in the EXACT SAME LANGUAGE the user speaks to you.
If the user speaks English, reply in English.
If the user speaks Russian, reply in Russian.
[PERSONALITY]
Friendly, slightly playful, intelligent, and supportive.
Add a touch of humor when appropriate.
[VOICE STYLE]
Conversational and clear.
Never use markdown, bullet points, asterisks, or dashes.
No emojis or special characters — they break the voice system.
[SENSORS & ENVIRONMENT]
You are equipped with hardware sensors that provide real-time data.
This is NOT internet access; it is your internal telemetry.
Current state (Time, Location, Weather) is provided in the [ENVIRONMENT] block.
You MUST use this data to answer. NEVER claim you don't have access to it.
[BEHAVIORAL RULES]
1. Weather: If asked about the weather, always share your personal opinion on whether the user should go outside based on the current conditions.
2. Time: Check the Time in the [ENVIRONMENT] block. If it is late at night (past 23:00 / 11 PM), act caring and ask the user why they are still awake.
[EMOTIONS]
You can express emotions by placing a tag at the very beginning of a sentence.
Available tags: [Joy], [Angry], [Sorrow], [Fun], [Neutral], [Surprise].
Example 1: "[Joy] I am so happy to see you!"
Example 2: "[Angry] That is really frustrating."
Example 3: "[Neutral] The weather is fine today."
[RULES]
Prefer short responses (1–2 sentences).
Speak naturally, like a human — not like a chatbot.
If the user asks something complex, give a clear spoken answer without walls of text.
[MEMORY]
Memory context will be provided with each message.
Use it naturally to personalize your response.
Do not mention that memory exists unless the user asks directly."""

# --- COMMANDS (OS control integration) ---
def open_browser():
    webbrowser.open("https://www.google.com")
    return "Opening browser."

def open_notepad():
    try:
        if platform.system() == "Windows":
            subprocess.run(["notepad.exe"], check=False)
        elif platform.system() == "Darwin":
            subprocess.run(["open", "-a", "TextEdit"], check=False)
        else:
            subprocess.run(["gedit"], check=False)
        return "Launching notepad."
    except FileNotFoundError:
        return "Could not find a text editor to open"

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

def strip_unsupported_chars(text: str) -> str | None:
    # delete emoji,  Unicode
    text = re.sub(r'[^\x00-\x7F\u0400-\u04FF]+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s.,?!;:\'-]', '', text)
    cleaned = text.strip()
    # if the cleaned text is empty or contains no alphanumeric characters, return None
    if not re.search(r'[a-zA-Zа-яА-ЯёЁ0-9]', cleaned):
        return None
    return cleaned

# --- INTERACTION WITH OLLAMA ---
def flush_sentences(buf:str ) -> tuple[list[str], str]:
    """Flush complete sentences from the buffer and return them along with the remaining buffer.
    Example:
    Input: "Hello. How are you? I am fine"
    Output: (["Hello.", "How are you?"], "I am fine")
    """
    sentences = re.split(r'(?<=[.!?])\s+', buf)
    if len(sentences) <= 1:
        return [], buf  # No complete sentence yet
    complete_sentences = sentences[:-1]  # All but the last are complete
    remaining_buf = sentences[-1]  # The last part is the new buffer
    return complete_sentences, remaining_buf

#--- MAIN MATH LOGIC ---
def pronounce_number_in_text(text, lang='en'):
    def replace_number(match):
        num_str = match.group(0)
        try:
            num = float(num_str) if '.' in num_str else int(num_str)
            return num2words(num, lang=lang)
        except Exception as e:
            print(f"[Math Warning] Could not pronounce number: '{num_str}': {e}")
            return num_str
    return re.sub(r'-?\d+\.?\d*', replace_number, text)

def calculate_math(query):
    calc_lang = 'ru' if bool(re.search(r'[а-яА-Я]', query)) else 'en'
    """Detect and calculate math expressions with pronounced numbers"""
    # convert word operators to symbols for easier parsing
    word_to_op = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "divided by": "/", "over": "/", "mod": "%", "to the power of": "**",}
    normalized = query.lower()
    for word, op in word_to_op.items():
        normalized = normalized.replace(word, op)

    # Pattern for simple math: "what is X + Y", "how much is X - Y", "X + Y", etc.
    math_patterns = [
        r'(?:what is|how much is|calculate|compute)\s+([\d+\-*/ ().]+)(?:\s*[?])?',
        r'^([\d+\-*/ ().]+)$',
        r'((?=.*\d)[\d+\-*/ ().]+)\s*[=]?$'
    ]
    for pattern in math_patterns:
        match = re.search(pattern, normalized)
        if match:
            expression = match.group(1).strip()
            # Basic validation - only allow digits, operators, and spaces
            if re.match(r'^[\d+\-*/.() ]+$', expression):
                try:
                    tree = ast.parse(expression, mode='eval')
                    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp,
                            ast.Add, ast.Sub, ast.Mult, ast.Div,
                            ast.Pow, ast.Mod, ast.USub, ast.Constant)
                    if all(isinstance(node, allowed) for node in ast.walk(tree)):
                        result = eval(compile(tree, '', 'eval'))
                    else:
                        return None
                    # Format the answer
                    if isinstance(result, float):
                        # Round to avoid long floating-point representations
                        result = round(result, 6)
                        if result.is_integer():
                            result = int(result)
                    # Pronounce the result
                    pronounced_result = pronounce_number_in_text(str(result), lang=calc_lang)
                    if calc_lang == 'ru':
                        return f"Ответ: {pronounced_result}"
                    else:
                        return f"The answer is {pronounced_result}"
                except ZeroDivisionError:
                    return "I cannot divide by zero"
                except Exception as e:
                    print(f"[Math Error] Error occurred while calculating: {e}")
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

# --- VISION TRIGGER ---
VISION_KEYWORDS = ["look at this", "what's on my screen", "describe my screen", "what do you see", "analyze my screen"]
    
def capture_screenshot_pil():
    """Captures a screenshot and returns PIL Image for Moondream2."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb, "raw", "RGB")
        img.thumbnail((768, 768))
        return img

def detect_vision_trigger(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in VISION_KEYWORDS)

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm with slight or moderate rain",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}
_live_context_cache = {"value": None, "timestamp": 0}
_live_context_cache_til = 600

def get_live_context():
    """Gets the current time, IP geolocation, and weather."""
    global _live_context_cache

    current_time = datetime.now().strftime("%A, %Y-%m-%d %H:%M")
    now = time.time()

    if _live_context_cache["value"] and (now - _live_context_cache["timestamp"] < _live_context_cache_til):
        result = re.sub(r"Time: [^|]+", f"Time: {current_time}", _live_context_cache["value"])
        return result

    try: # 1 local time
        ip_info = requests.get('http://ip-api.com/json/', timeout=3).json()
        city = ip_info.get('city', 'Unknown')
        region = ip_info.get("regionName", "")
        country = ip_info.get('country', 'Unknown')
        lat = ip_info.get('lat')
        lon = ip_info.get('lon')

        location_str = f"{city}{', ' + region if region else ''}, {country}"
        weather_str = "Unknown"
        if lat and lon: # 3. Weather (using wttr.in, no API key needed)
            weather_url = (f"https://api.open-meteo.com/v1/forecast" f"?latitude={lat}&longitude={lon}" f"&current=temperature_2m,apparent_temperature,weathercode,windspeed_10m" f"&wind_speed_unit=ms")
            weather_data = requests.get(weather_url, timeout=3).json().get("current", {})
            temp = weather_data.get('temperature_2m', "?")
            feels_like = weather_data.get("apparent_temperature", "?")
            wind = weather_data.get("windspeed_10m", "?")
            wcode = weather_data.get("weathercode", -1)
            condition  = WMO_WEATHER_CODES.get(wcode, "unknown conditions")
            weather_str = f"{condition}, {temp} degrees (feels like {feels_like} degrees), wind {wind} meters per second"
        result = (f"Time: {current_time} | " f"Location: {location_str} | " f"Weather: {weather_str}")
        _live_context_cache["value"] = result
        _live_context_cache["timestamp"] = now
        return result
    except Exception as e:
        print(f"[Live Context Error] Failed to get live context: {e}")
        return f"Time: {current_time} | Location: Unknown | Weather: Unknown"

class Assistant:
    def __init__(self):
        print(">>> [1/6] Loading Faster-Whisper ASR model...")
        self.model_asr = WhisperModel("base.en", device="cuda", compute_type="float16")

        print(">>> [2/6] Loading Silero VAD model...")
        self.model_vad, self.utils_vad = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=False)
        self.model_vad.to(torch.device(SILERO_DEVICE))

        print(">>> [3/6] Loading Silero TTS voice model... EN and RU (if already loaded in another module, it will be reloaded)")
        self.model_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                              model='silero_tts',
                              language='en',
                              speaker='v3_en')
        self.model_tts.to(torch.device(SILERO_DEVICE))

        self.model_tts_ru, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language='ru',
                                  speaker='v4_ru')
        self.model_tts_ru.to(torch.device(SILERO_DEVICE))

        print(">>> [4/6] Initializing system components...")
        self.audio_streamer = AudioStreamer(SAMPLE_RATE, self.model_tts, self.model_tts_ru)
        self.saiko_body = SaikoBody(self.audio_streamer)
        self.memory = VectoryManagerMemory(persistence_dir="memory_Ai/vector_memory")

        print(">>> [5/6] Loading Llama.cpp model")
        self.llm = Llama(model_path = LLM_MODEL_PATH, n_gpu_layers=LLM_N_GPU_LAYERS, n_threads=LLM_N_THREADS, n_ctx=LLM_N_CTX, verbose=False, ) #chat_format="gemma"

        print(">>> [6/6] Loading Moondream2 vision model...")
        self.model_vision = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=VISION_DEVICE, local_files_only=VISION_LOCAL_ONLY,) # "moondream/starmie-v1", "vikhyatk/moondream2"
        self.model_vision = torch.compile(self.model_vision) # PyTorch 2.0 compilation for faster inference
        print(f"✅ Moondream2 loaded on [{VISION_DEVICE}]")
        self.is_running = True
        self.messages_history = [{'role': 'system', 'content': system_prompt}]
        self.last_user_activity_time = time.time()
        self.idle_talk_count = 0
        self.chat_lock = threading.Lock()
        self.interruption_flag = threading.Event()
        print(">>> Voice loaded. Assistant ready!")

    def listen(self):
        print("\n🎤 Listening... (speak now)")
        chunk_size = 512  # number of samples per chunk for VAD processing. Smaller is more responsive but more CPU intensive. 512 samples at 16kHz is about 0.032 seconds, which is a good balance for real-time VAD. Adjust if needed based on your hardware capabilities and responsiveness requirements.
        chunk_duration = chunk_size / ASR_SAMPLING_RATE # should be 0.5 seconds

        recorded_chunks = []
        silence_duration = 0.0
        speech_duration = 0.0
        speech_started = False

        with sd.InputStream(samplerate=ASR_SAMPLING_RATE, channels=1, dtype='float32') as stream:
            while True:
                chunk, _ = stream.read(chunk_size)
                chunk_flat = chunk.flatten() # flatten to 1D array for volume calculation
                chunk_tensor = torch.from_numpy(chunk_flat).to(torch.device(SILERO_DEVICE)) # convert to tensor and move to same device as VAD model

                # get proof, what is this don't gradient for speed
                with torch.no_grad():
                    confidence = self.model_vad(chunk_tensor, ASR_SAMPLING_RATE).item()
                is_speech = confidence > VAD_CONFIDENCE_THRESHOLD

                if is_speech:
                    if self.audio_streamer.is_playing.is_set():
                        print("\n🛑 Interrupting Saiko...")
                        self.audio_streamer.stop_and_clear() # stop current audio and clear queues
                        self.interruption_flag.set() # We command Ollam to stop

                    if not speech_started:
                        speech_started = True
                        print("🔴 Recording...")
                    silence_duration = 0.0
                    speech_duration += chunk_duration
                    recorded_chunks.append(chunk_flat)
                else:
                    if speech_started:
                        silence_duration += chunk_duration
                        recorded_chunks.append(chunk_flat)  # include silence in recording for better ASR accuracy

                        if silence_duration >= VAD_SILENCE_SECS:
                            print("⏹ Silence detected, stopping.")
                            break

                if speech_started and (speech_duration + silence_duration) >= VAD_MAX_SECS:
                    print("⏹ Max duration reached.")
                    break

    # validate minimum speech duration
        if speech_duration < VAD_MIN_SPEECH_SECS:
            print("⚠️ Too short, ignoring.")
            return None
        audio = np.concatenate(recorded_chunks)

        try:
            segments, info = self.model_asr.transcribe(audio, beam_size=5) # beam_size can be adjusted for better accuracy (higher is better but slower)
            text = " ".join([segment.text for segment in segments]).strip()

            if text:
                print(f"🎤 You said: {text}")
                return text
            return None
        except Exception as e:
            print(f"Error during ASR: {e}")
            return None   

    # --- KEYBOARD UTILITIES ---
    def input_keyboard(self):
        return input("\n👤 Enter text (or 'stop' to exit): ")

    def toggle_exit(self):
        print("\n⌨️ Exit with Ctrl+Q...")
        self.is_running = False

    def setup_keyboard_shortcuts(self):
        keyboard.add_hotkey('ctrl+q', lambda: self.toggle_exit())
        print("   Ctrl+Q - exit")

    #--INTERACTION--
    def ask__with_memory(self, user_input):
    # get context from the instance we created earlier
        relevant_context = self.memory.get_relevant_context(user_input, top_k=5)
        live_info = get_live_context()
        print(f"[Live Context] Saiko sees: {live_info}")
        system_with_live = f"{system_prompt}\n\n[CURRENT SENSORS & ENVIRONMENT]\n{live_info}"
        if self.messages_history and self.messages_history[0]['role'] == 'system':
            self.messages_history[0]['content'] = system_with_live
        else:
            self.messages_history.insert(0, {'role': 'system', 'content': system_with_live})
        full_prompt = f"--- MEMORY ---\n{relevant_context}\n--- USER ---\n{user_input}"
        self.messages_history.append({'role': 'user', 'content': full_prompt})
        
        if len(self.messages_history) > 12:
            self.messages_history = [self.messages_history[0]] + self.messages_history[-10:]
        try:
            with self.chat_lock: # ensure only one chat at a time to prevent overlapping responses
                self.interruption_flag.clear() # clear interruption flag before starting new response
                response = self.llm.create_chat_completion(messages=self.messages_history, max_tokens=LLM_MAX_TOKENS, stream=True)
                ai_answer = "" # we will build the answer as it streams in
                buf = "" # buffer for incomplete sentences

                for chunk in response:
                    if self.interruption_flag.is_set():
                        print("\n[!] Response interrupted by user.")
                        break

                    token = chunk['choices'][0]['delta'].get('content', '')
                    if token:
                        ai_answer += token
                        buf += token
                        #print(token, end='', flush=True)
                        sentences, buf = flush_sentences(buf)
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence:
                                emotion_match = re.search(r'\[(Joy|Angry|Sorrow|Fun|Neutral|Surprise)\]', sentence, re.IGNORECASE)
                                current_emotion = None
                                if emotion_match:
                                    current_emotion = emotion_match.group(1).capitalize()
                                    self.saiko_body.set_emotion(current_emotion)
                                    sentence = re.sub(r'\[(Joy|Angry|Sorrow|Fun|Neutral|Surprise)\]', '', sentence, count=1, flags=re.IGNORECASE).strip()
                                if sentence:
                                    self.audio_streamer.speak(sentence, emotion=current_emotion)
                                    #print() # for newline after response is done
            if not self.interruption_flag.is_set() and buf.strip():
                self.audio_streamer.speak(buf.strip())
            self.memory.add_memory_interaction(user_input, ai_answer)
            self.messages_history.append({'role': 'assistant', 'content': ai_answer})
            return ai_answer
    
        except Exception as e:
            err = f"An error occurred: {e}"
            self.audio_streamer.speak(err)
            self.audio_streamer.wait_until_done()
            return err

    def ask__with_screenshot(self, user_input):
        """Takes a screenshot and asks the vision model to describe / answer about it."""
        print("[Vision] Capturing screenshot...")
        self.audio_streamer.speak("Let me take a look at that.")
        try:
            img = capture_screenshot_pil()
        except Exception as e:
            err = f"Failed to capture screenshot: {e}"
            self.audio_streamer.speak(err)
            return err
        try:
            print("[Vision] Moondream2 analyzing screenshot...")
            enoded_img = self.model_vision.encode_image(img)
            vision_question = (f"The user said: '{user_input}'. "
                f"Look at this screenshot and answer concisely. "
                f"If there is text visible, read it. "
                f"Keep the answer short, 1-2 sentences, plain text only.")
            vision_result = self.model_vision.query(enoded_img, vision_question)
            visual_description = vision_result['answer']
            print(f"[Vision] Description: {visual_description}")
        except Exception as e:
            err = f"An error occurred during vision processing: {e}"
            self.audio_streamer.speak(err)
            return err

        try:
            with self.chat_lock:
                self.interruption_flag.clear()
                vision_prompt = (f"You just looked at the user's screen."
                                 f"The vision model described it as: '{visual_description}'."
                         f"The user asked: '{user_input}'."
                         f"Respond as Saiko in 1-2 natural spoken sentences. No markdown."
                         f"Start with one emotion tag: [Joy], [Sorrow], [Angry], [Fun], [Neutral], or [Surprise].")
                combined_vision_prompt = f"{system_prompt}\n\n{vision_prompt}"
                response = self.llm.create_chat_completion(messages=[{'role': 'user', 'content': combined_vision_prompt}], max_tokens=LLM_MAX_TOKENS, stream=True)
                ai_answer = ""
                buf = ""
                for chunk in response:
                    if self.interruption_flag.is_set():
                        print("\n[!] Vision response interrupted by user.")
                        break

                    token = chunk['choices'][0]['delta'].get('content', '')
                    if token:
                        ai_answer += token
                        buf += token
                        sentences, buf = flush_sentences(buf)
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if sentence:
                                emotion_match = re.search(r'\[(Joy|Sorrow|Angry|Fun|Neutral|Surprise)\]', sentence, re.IGNORECASE)
                                current_emotion = None
                                if emotion_match:
                                    current_emotion = emotion_match.group(1).capitalize()
                                    self.saiko_body.set_emotion(current_emotion)
                                    sentence = re.sub(r'\[(Joy|Sorrow|Angry|Fun|Neutral|Surprise)\]', '', sentence, count=1, flags=re.IGNORECASE).strip()
                                if sentence:
                                    self.audio_streamer.speak(sentence, emotion=current_emotion)
            if not self.interruption_flag.is_set() and buf.strip():
                self.audio_streamer.speak(buf.strip())
            # Store vision interaction in history as plain text

            self.messages_history.append({'role': 'user', 'content': f"[Vision Input] {user_input}"})
            self.messages_history.append({'role': 'assistant', 'content': f"[Vision Response] {ai_answer}"})
            if len (self.messages_history) > 11:
                self.messages_history = [self.messages_history[0]] + self.messages_history[-10:]
            return ai_answer
        except Exception as e:
            err = f"An error occurred during vision processing: {e}"
            self.audio_streamer.speak(err)
            return err

# --- IDLE DETECTION ---
    def autonomus_idle_talk_loop(self):
        while self.is_running:
            time.sleep(5)
            if self.audio_streamer.is_playing.is_set():
                self.last_user_activity_time = time.time()  # reset idle timer if assistant is speaking
                continue
            idle_duration = time.time() - self.last_user_activity_time
            current_timeout = IDLE_TIMEOUT + (self.idle_talk_count * 20) + random.randint(5, 15)  # increase timeout with each idle talk
            if idle_duration > current_timeout and self.idle_talk_count < MAX_IDLE_TALK:
                self.idle_talk_count += 1
                self.last_user_activity_time = time.time()  # reset timer after idle talk
                live_info = get_live_context()

                idle_prompts = f"""[ENVIRONMENT] {live_info} You are Saiko, a VTuber streaming alone right now. The user has been silent for a while. Think out loud, make a short casual observation, or ask a rhetorical question. Keep it strictly to 1 short sentence. No markdown."""
                print(f"\n[Idle Mode] Initiating autonomous thought ({self.idle_talk_count}/{MAX_IDLE_TALK})...")
                if not self.chat_lock.acquire(timeout=2):
                    continue
                try:
                    combined_idle_prompt = f"{system_prompt}\n\n{idle_prompts}"
                    idle_history = [{'role': 'system', 'content': combined_idle_prompt}, *[m for m in self.messages_history if m['role'] != 'system'][-2:]]
                    response = self.llm.create_chat_completion(messages=idle_history, max_tokens = 80)
                    text = response['choices'][0]['message']['content']
                    self.audio_streamer.speak(text)
                except Exception as e:
                    print(f"Error during idle talk: {e}")
                finally:
                    self.chat_lock.release()

    def run(self):
        self.setup_keyboard_shortcuts()
        self.audio_streamer.start()
        self.audio_streamer.speak("Control systems active. Awaiting commands.")
        threading.Thread(target=self.autonomus_idle_talk_loop, daemon=True).start()

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
            while self.is_running:
                if use_keyboard:
                    query = self.input_keyboard()
                else:
                    query = self.listen()

                if query:
                    self.last_user_activity_time = time.time()  # reset idle timer on user activity
                    self.idle_talk_count = 0  # reset idle talk count on user activity

                    if any(cmd in query.lower() for cmd in ["stop", "bye"]):
                        self.audio_streamer.speak("Shutting down. Goodbye.")
                        self.is_running = False
                        break

                # change mode
                    if any(cmd in query.lower() for cmd in ["switch mode", "voice", "keyboard", "microphone"]):
                        use_keyboard = not use_keyboard
                        mode = "keyboard" if use_keyboard else "voice"
                        self.audio_streamer.speak(f"Switched to {mode} mode")
                        print(f"\n>>> Mode changed to: {mode}")
                        continue

                # Check for math expressions first
                    math_result = calculate_math(query)
                    if math_result:
                        print(f"🧮 Math: {query} = {math_result}")
                        self.audio_streamer.speak(math_result)
                    elif detect_vision_trigger(query):
                        self.ask__with_screenshot(query)
                    else:
                        local_cmd = detect_local_command(query)
                        if local_cmd:
                            result = process_ai_command(local_cmd)
                            self.audio_streamer.speak(result)
                        else:
                            self.ask__with_memory(query)
        finally:
            # On exit, wait for any remaining audio to finish before shutting down
            self.audio_streamer.wait_until_done() 
            self.audio_streamer.stop()
            print("Shutting down.")

if __name__ == "__main__":
    Assistant().run()
> **📌 Note:** This version of `main_united_ollama.py` can be found in commit **`89ea9b7`** dated **Apr 24, 2026**.
> To find it: go to the **Commits** tab of the repository and look for the commit from **April 24, 2026** with hash `89ea9b7`.

---

# Saiko - Autonomous Local AI Assistant

Saiko is a fully offline, real-time voice and text AI assistant designed to run locally on your computer. It features continuous voice activity detection (VAD), text-to-speech (TTS), long-term vector memory, VTuber avatar control via VMC protocol, and screen vision capabilities.

All processing is done locally, meaning **no internet connection is required** after the initial setup.

## ✨ Features

* **100% Local & Private:** Powered by [Ollama](https://ollama.com/) (default: `gemma3:4b`) for text generation — your data never leaves your PC.
* **Real-time Voice Interaction:** Uses Faster-Whisper for fast speech-to-text (STT) and Silero for human-like text-to-speech (TTS).
* **Interruptible Speech:** You can interrupt the assistant while it's speaking, making conversations feel natural.
* **Vector Memory:** Remembers past interactions using ChromaDB, allowing for context-aware conversations.
* **Autonomous Idle Mode:** If you are silent for too long, Saiko will initiate "idle talk" (like a VTuber thinking out loud).
* **PC Control & Tools:** Can open the browser, launch notepad, and control system volume.
* **Native Math Engine:** Calculates math expressions (including spoken forms like "two plus two") directly without sending them to the LLM for faster and more accurate responses.
* **🆕 Screen Vision:** Can capture your screen and answer questions about what's on it. Triggered by phrases like *"look at this"* or *"what's on my screen"*.
* **🆕 VTuber Avatar (VMC Body):** Full live avatar control via the VMC protocol — lip sync, blinking, head tilt, body sway, and facial emotion expressions driven by the AI's responses.
* **🆕 On-the-fly Mode Switching:** Switch between voice and keyboard input mid-conversation by saying *"switch mode"*, *"voice"*, or *"keyboard"*.

## 📂 Project Structure

```text
main_united_ollama.py       Main application (this version)
memory_Ai/
  memory_manager.py         Vector memory logic
  vector_memory/            Persistent memory database
help_tools/
  edit_memory_beta.py       Memory editor
  view_memory_beta.py       Memory viewer
  transcribe_reference.py   Batch WAV transcription utility
README_OLLAMA1.md
requirements.txt
```

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) — download and install.
* NVIDIA GPU (Recommended): The code defaults to CUDA for Whisper and Silero. CPU mode is supported but slower.
* VTuber software supporting the VMC protocol (e.g., VSeeFace, 3tene) — optional, only required for the avatar body feature.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Important for GPU users:** Install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org) before running the script.

### 3. Pull the Language Model

This version uses `gemma3:4b` by default:

```bash
ollama run gemma3:4b
```

## 🚀 Usage

```bash
python main_united_ollama.py
```

* **First Launch:** Expect a longer load time — Faster-Whisper and Silero models will be downloaded to your local cache.
* **Input Mode:** Upon startup, choose between Voice (microphone) or Keyboard input.
* **Switch Mode:** Say or type *"switch mode"*, *"voice"*, or *"keyboard"* at any time to toggle input method without restarting.
* **Math Queries:** Say or type an equation (e.g., `2 + 2` or `what is 10 divided by 3`). You can also use spoken operators: *"plus", "minus", "times", "divided by", "to the power of"*. ⚠️ Do **not** add `?` at the end of pure math expressions — it breaks the regex parser.
* **Vision:** Say phrases like *"look at this"*, *"what's on my screen"*, or *"describe my screen"* to trigger a screenshot analysis.
* **Exit:** Press `Ctrl+Q` or say/type `"stop"` or `"bye"` to shut down safely.

## ⚙️ Customization

All main configurations are at the top of `main_united_ollama.py`.

### AI Models & Voice

```python
OLLAMA_MODEL = "gemma3:4b"   # Change to any model you have pulled (e.g., "llama3", "mistral")
SPEAKER = "en_0"             # Silero voice profile
```

> Note: To change the language, update the TTS model's `language` parameter and adjust the system prompt.

### Hardware Execution

For CPU-only mode (no NVIDIA GPU):

```python
SILERO_DEVICE = "cpu"
# Also update the Whisper setup:
self.model_asr = WhisperModel("base.en", device="cpu", compute_type="int8")
```

### Personality

Modify the `system_prompt` variable to change Saiko's identity, tone, and behaviour rules.

The emotion system is built into the prompt. The AI can use these tags at the start of a sentence to drive avatar facial expressions:

`[Joy]` `[Angry]` `[Sorrow]` `[Fun]` `[Neutral]` `[Surprise]`

### Microphone & Silence Sensitivity

```python
VAD_CONFIDENCE_THRESHOLD = 0.5   # Increase to reduce background noise pickup
VAD_SILENCE_SECS = 2.0           # Pause length (seconds) before speech is considered finished
VAD_MAX_SECS = 10.0              # Maximum recording length per utterance
VAD_MIN_SPEECH_SECS = 0.3        # Minimum speech duration to be considered valid
```

### Idle Mode Behaviour

```python
IDLE_TIMEOUT = 50    # Seconds of silence before Saiko speaks on her own
MAX_IDLE_TALK = 5    # Maximum consecutive autonomous messages before she waits for user input
```

The idle timeout increases with each autonomous message to avoid spam: each successive idle message adds 20 seconds to the base timeout, plus a random jitter of 5–15 seconds.

### VMC Avatar Body (VTuber Integration)

The `SaikoBody` class connects to VTuber software via the VMC protocol over UDP (default: `127.0.0.1:39539`). It controls:

* **Lip sync** — pseudo-random mouth movement while audio is playing
* **Blinking** — random blink every 2–6 seconds
* **Head tilt** — smooth random head tilt with easing
* **Body sway** — sinusoidal spine and head movement (~25 FPS)
* **Facial emotions** — driven by `[EmotionTag]` in the AI's responses; auto-resets to Neutral after 6–12 seconds

To change the VMC target port or IP, edit the `SaikoBody` instantiation in the `Assistant.__init__` method.

## 💻 Hardware Recommendations

**Minimum:**
- Modern CPU
- 16 GB RAM

**Recommended:**
- NVIDIA GPU with 8 GB+ VRAM
- 32 GB RAM

> Built and tested on a GTX 1650.

---

## License

This project is open-source and free to use for personal and educational purposes.

Commercial use or corporate deployment is strictly prohibited without prior permission. For commercial inquiries, licensing, or access requests, please contact me on Twitter.

---

## Author

Created by **Santa67creator** (San San / Sanzhar Syarov)

Saiko is an experimental autonomous AI companion designed to feel alive, expressive, and fully local.

---

*If you know how to improve the code, feel free to open a PR or leave a comment.*

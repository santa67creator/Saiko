# Saiko - Autonomous Local AI Assistant

Saiko is a fully local, real-time AI assistant with voice interaction, screen understanding, long-term memory, and VTuber avatar control.

All core AI processing runs on your own computer:
- Speech recognition with Faster-Whisper
- Language generation with llama.cpp + Google Gemma 3 GGUF
- Text-to-speech with Silero TTS
- Vision with Moondream 2
- Long-term vector memory
- Real-time VTuber animation through VMC Protocol

After the initial model download, the assistant can work offline.

---

## Features

- 100% Local and Private
- Real-time voice conversations
- Interruptible speech
- English and Russian support
- Screen analysis ("look at my screen")
- Long-term vector memory
- Autonomous idle talk
- Built-in math engine
- Basic PC control (browser, notepad, volume)
- VTuber avatar lip sync, emotions, blinking, and body movement

---

## Core Technologies

- Python 3.10+
- llama.cpp
- Google Gemma 3 4B IT (GGUF)
- Faster-Whisper
- Silero VAD
- Silero TTS
- Moondream 2
- ChromaDB
- VMC Protocol

---

## Project Structure

```text
main_united.py          Main application
memory_Ai/
  memory_manager.py     Vector memory logic
  vector_memory/        Persistent memory database
help_tools/
  edit_memory_beta.py   Memory editor
  view_memory_beta.py   Memory viewer
  transcribe_reference.py
models/
  google_gemma-3-4b-it-Q5_K_M.gguf
README.md
requirements.txt
```

---

## Installation

### 1. Requirements

- Python 3.10 or newer
- NVIDIA GPU recommended
- CUDA-enabled PyTorch for best performance

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### For NVIDIA GPU acceleration (strongly recommended):
### Install CUDA-enabled PyTorch first (see pytorch.org)
### Then install llama-cpp-python with CUDA support:
```
set CMAKE_ARGS="-DLLAMA_CUDA=on"

pip install -r requirements.txt --force-reinstall --no-cache-dir
```
### 3. Download a GGUF Model

Place your model in the `models/` folder. (you need create folder myself)

Default model:

```text
models/google_gemma-3-4b-it-Q5_K_M.gguf
```

### 4. First Launch

The first launch will download:
- Faster-Whisper models
- Silero models
- Moondream 2 weights

---

## Usage

```bash
python main_united.py
```

Choose:
1. Voice input
2. Keyboard input

Exit with:
- Ctrl+Q
- "stop"
- "bye"


VTuber Integration

Saiko sends animation data via the VMC Protocol on port 39539. 

To see the avatar move, make sure you have a compatible software (like VSeeFace or VNyan) running and configured to listen to VMC data on 127.0.0.1:39539

---

## Voice Commands

### Vision
- "Look at my screen"
- "What's on my screen?"
- "Describe my screen"

### System Control
- "Open browser"
- "Open notepad"
- "Volume up"
- "Volume down"

### Math
- "2 + 2"
- "What is 15 * 7"

---

## Configuration

All settings are located at the top of `main_united.py`.

### LLM

```python
LLM_MODEL_PATH = "models/google_gemma-3-4b-it-Q5_K_M.gguf"
LLM_N_GPU_LAYERS = 0
LLM_N_THREADS = 4
LLM_N_CTX = 4096
LLM_MAX_TOKENS = 512
```

### Vision

```python
VISION_DEVICE = "cuda"
VISION_LOCAL_ONLY = True
```

### Audio

```python
SAMPLE_RATE = 48000
ASR_SAMPLING_RATE = 16000
```

### VAD

```python
VAD_CONFIDENCE_THRESHOLD = 0.5
VAD_SILENCE_SECS = 2.0
VAD_MAX_SECS = 10.0
VAD_MIN_SPEECH_SECS = 0.3
```

### Idle Mode

```python
IDLE_TIMEOUT = 50
MAX_IDLE_TALK = 5
```

---

## Personality Customization

Edit the `system_prompt` variable in `main_united.py` to change:
- Personality
- Tone of voice
- Behavioral rules
- Emotion usage

---

## Hardware Recommendations

Minimum:
- 16 GB RAM (4GB)
- Modern CPU

Recommended:
- NVIDIA GPU with 8 GB+ VRAM
- 32 GB RAM

i make on gtx 1650

---

## Current Capabilities

- Streaming LLM responses
- Real-time speech interruption
- English and Russian TTS
- Live weather, time, and location context
- Screenshot understanding
- Persistent vector memory
- VTuber avatar animation

---

## License

This project is open-source and free to use for personal and educational purposes

Commercial use or corporate deployment is strictly prohibited without prior permission. For commercial inquiries, licensing, or access requests, please contact me on Twitter

---

## Author

Created by Santa67creator (San San) (Sanzhar Syarov)

Saiko is an experimental autonomous AI companion designed to feel alive, expressive, and fully local.

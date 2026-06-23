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

<img width="1280" height="664" alt="5188371402275361982" src="https://github.com/user-attachments/assets/0af3ce98-f88d-4809-a697-8cf03eb070a0" />

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

```bash
git clone https://github.com/santa67creator/cool-yea-jarvis.git
```

Activate env (or create your own):
```bash
# Windows
./venv4/Scripts/Activate

# Linux / macOS
source venv4/bin/activate
```

---

### Step 1 — Requirements

- Python 3.10 or newer
- NVIDIA GPU recommended (tested on GTX 1650)
- CUDA Toolkit installed on your system

---

### Step 2 — Install CUDA Toolkit (NVIDIA only)

Download and install the CUDA Toolkit for your OS from the official page:

> https://developer.nvidia.com/cuda-downloads

After installation, verify it works:
```bash
nvcc --version
```

Make sure the version matches what you'll use for PyTorch (e.g. CUDA 11.8 or 12.1).

---

### Step 3 — Install PyTorch with CUDA support

Go to the official PyTorch install page and generate the exact command for your system:

> https://pytorch.org/get-started/locally/

Select: `Stable` → `Pip` → `Python` → your CUDA version.



---

### Step 4 — Install llama-cpp-python with CUDA support

`llama-cpp-python` must be compiled with CUDA flags. Install it **before** running `requirements.txt`.

**Windows:**
```bash
set CMAKE_ARGS="-DGGML_CUDA=on"
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Linux / macOS:**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --no-cache-dir
```

> Full documentation and troubleshooting for llama-cpp-python:
> https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration

Verify the install:
```bash
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
```

---

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

### Step 6 — Download a GGUF Model

Create the `models/` folder and place your model inside:

```bash
mkdir models
```

Default model path:
```text
models/google_gemma-3-4b-it-Q5_K_M.gguf
```

---

### Step 7 — First Launch

The first launch will automatically download:
- Faster-Whisper models
- Silero VAD / TTS models
- Moondream 2 weights

```bash
python main_united.py
```

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

---

## VTuber Integration

Saiko sends animation data via the VMC Protocol on port `39539`.

To see the avatar move, run a compatible app (e.g. VSeeFace or VNyan) configured to receive VMC data on `127.0.0.1:39539`.

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

All settings are at the top of `main_united.py`.

### LLM

```python
LLM_MODEL_PATH = "models/google_gemma-3-4b-it-Q5_K_M.gguf"
LLM_N_GPU_LAYERS = 0       # Set higher (e.g. 20–35) to offload layers to GPU
LLM_N_THREADS = 4
LLM_N_CTX = 4096
LLM_MAX_TOKENS = 512
```

> Tip: increase `LLM_N_GPU_LAYERS` to speed up inference on NVIDIA GPUs.
> Start with `20` and increase until you hit VRAM limits.

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
- 16 GB RAM
- Modern CPU

Recommended:
- NVIDIA GPU with 8 GB+ VRAM
- 32 GB RAM

Tested on GTX 1650.

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

This project is open-source and free to use for personal and educational purposes.

Commercial use or corporate deployment is strictly prohibited without prior permission.
For commercial inquiries, licensing, or access requests, please contact me on Twitter.

Licensed under AGPL-3.0 with Commons Clause. See LICENSE and COMMONS-CLAUSE.md

---

## Author

Created by Santa67creator (San San) (Sanzhar Syarov)

Saiko is an experimental autonomous AI companion designed to feel alive, expressive, and fully local.

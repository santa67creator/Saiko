ALL CHANGES YOU CAN SEE IN COMMIT, AND ALL DESCRIBE ABOUT CHANGES YOU ALSO SEE IN COMMIT

# Saiko - Autonomous Local AI Assistant

Saiko is a fully offline, real-time voice and text AI assistant designed to run locally on your computer. It features continuous voice activity detection (VAD), text-to-speech (TTS), long-term vector memory, and basic PC control capabilities.

All processing is done locally, meaning **no internet connection is required** after the initial setup.

## ✨ Features

* 100% Local & Private:** Powered by [Ollama](https://ollama.com/) (default: Gemma) for text generation, meaning your data never leaves your PC.
* Real-time Voice Interaction:** Uses Faster-Whisper for fast speech-to-text (STT) and Silero for human-like text-to-speech (TTS).
* Interruptible Speech:** You can interrupt the assistant while it's speaking, making conversations feel natural.
* Vector Memory:** Remembers past interactions using ChromaDB, allowing for context-aware conversations.
* Autonomous Idle Mode:** If you are silent for too long, Saiko will initiate "idle talk" (like a VTuber thinking out loud) to keep the interaction alive.
* PC Control & Tools:** Can open the browser, launch notepad, and control system volume.
* Native Math Engine:** Calculates math expressions directly without sending them to the LLM for faster and more accurate responses. 

## 📂 Project Structure

The repository is organized to separate the core logic from memory management and utility tools:

* `main_united.py`: The core application. Contains the logic for audio streaming, LLM routing, VAD, and OS commands.
* `memory_Ai/`: Contains the ChromaDB vector database (`vector_memory/`) and `memory_manager.py` which handles saving and retrieving contextual dialogue.
* `help tools/`: A suite of utilities to manage the assistant's brain:
  * `edit_memory_beta.py`: Console menu to search, delete, or manually inject facts into the vector memory.
  * `view_memory_beta.py`: Displays all raw records currently stored in the database.
  * `transcribe_reference.py`: A utility to batch transcribe `.wav` files using Faster-Whisper.

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10+
* Ollama: Download and install from [ollama.com](https://ollama.com/).
* NVIDIA GPU (Recommended): The code is optimized for CUDA (`device="cuda"`).

### 2. Install Dependencies
Clone the repository, create a virtual environment, and install the required packages.

```bash
pip install -r requirements.txt

Important Note for GPU Users: To utilize your NVIDIA GPU, you must install the CUDA-enabled version of PyTorch. Visit the PyTorch website to get the correct installation command for your system before running the script.

3. Pull the Language Model
By default, Saiko uses the gemma model. You need to pull it via Ollama:
ollama run gemma

🚀 Usage
Run the main script:
python main_united.py

First Launch: There will be a long loading time and delay when you first launch it. The script needs to download the Faster-Whisper and Silero models to your local cache.

Input Mode: Upon startup, you can choose between Voice (microphone) or Keyboard input.

Math Queries: When asking math problems, simply type/say the equation (e.g., 2 + 2). WARNING: Do not use "?" for math queries (e.g., do not use 2 + 2?), as it breaks the regex parser.

Exit: Press Ctrl+Q or say/type "stop" or "bye" to shut down the assistant safely and close audio streams.

⚙️ Customization (Where to change things)
All main configurations are located at the top of main_united.py. You can easily tweak the assistant to your liking:

AI Models & Voice

Change the LLM: Change OLLAMA_MODEL = "gemma" to any model you have pulled (e.g., llama3, mistral).

Change the Voice: Change SPEAKER = "en_0" to another Silero voice profile. Note: If you want to change the language, you must update the TTS model language parameters and the prompt.

Hardware Execution: If you don't have an NVIDIA GPU, change SILERO_DEVICE = "cuda" to "cpu", and update the Whisper setup to device="cpu", compute_type="int8".

Personality
Modify the system_prompt variable in main_united.py to change Saiko's identity, tone, and behavior rules.

Microphone & Silence Sensitivity
Adjust the VAD (Voice Activity Detection) settings in main_united.py:

VAD_CONFIDENCE_THRESHOLD = 0.5 (Increase if it picks up too much background noise).

VAD_SILENCE_SECS = 2.0 (How long you need to pause before it considers you finished speaking).

Idle Mode Behavior
IDLE_TIMEOUT = 50: Seconds of silence before Saiko speaks on her own.
MAX_IDLE_TALK = 5: Maximum consecutive autonomous messages before she waits for your input.

There will be a long loading time and delay when you first launch it 

so this is file will "readme" can be changed then time
if you know how change code better, you can write in comments or fork idk


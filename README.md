# Fast Wisper

Voice dictation tool for French using faster-whisper with GPU acceleration.

## Features

- F9 hotkey to start/stop recording
- Real-time audio capture at 16kHz mono (required by faster-whisper)
- Voice Activity Detection (VAD) for better speech recognition
- GPU acceleration (CUDA) with automatic CPU fallback
- Automatic text typing to active window via pyautogui

## Installation

```bash
# Clone the repo
git clone https://github.com/Grow1nly/fast-wisper.git
cd fast-wisper

# Install dependencies
pip install faster-whisper sounddevice numpy pyautogui keyboard torch

# Download the model (or use the provided one in models/)
# The model will be loaded from models/faster-whisper-large-v3/
```

## Usage

```bash
python voice_to_claude.py
```

- Press **F9** to start recording
- Speak in French
- Press **F9** again to stop and transcribe
- Text is automatically typed to the active window

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (optional, falls back to CPU)
- Microphone

## Model

Uses `faster-whisper-large-v3` model. Place it in `models/faster-whisper-large-v3/` or let it download automatically.

## License

MIT
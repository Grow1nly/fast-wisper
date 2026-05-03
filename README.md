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

# Create conda env from environment.yml
conda env create -f environment.yml

# Activate
conda activate fast-wisper
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

- Python 3.10
- NVIDIA GPU with CUDA (optional, falls back to CPU)
- Microphone
- Conda (for environment management)

## Key Dependencies

The environment is based on `whisper12` conda env with these key packages:
- faster-whisper==1.2.1
- torch==2.5.1+cu121
- sounddevice==0.5.5
- pyautogui==0.9.54
- keyboard==0.13.5
- numpy==2.2.6

## Model

Uses `faster-whisper-large-v3` model. Place it in `models/faster-whisper-large-v3/` or let it download automatically.

## License

MIT
#!/usr/bin/env python3
"""
Voice to Claude - Voice Dictation
Press F9 to start/stop recording
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import pyautogui
import keyboard
import threading
import time

print("🚀 Voice to Claude")
print("=" * 40)

# ======================
# CONFIGURATION
# ======================

# IMPORTANT: faster-whisper requires 16kHz mono float32
FS = 16000  # Sample rate required: 16kHz
CHUNK_DURATION = 1  # Seconds per chunk
SILENCE_THRESHOLD = 0.005  # Lower silence threshold to capture soft speech

# Check microphone
try:
    device_info = sd.query_devices(kind='input')
    print(f"🎙️ Microphone: {device_info['name']}")
except Exception as e:
    print(f"❌ Device error: {e}")

print(f"🎙️ Sample rate: {FS} Hz (required for Whisper)")
print("⌨️ F9 = Start/Stop dictation")

# ======================
# LOAD MODEL
# ======================
print("\n🔄 Loading model...")

import torch
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    compute_type = "int8"
    print("⚠️ CPU")

# Load local model (no HF download)
model_path = os.path.join(os.path.dirname(__file__), "models", "faster-whisper-large-v3")
model = WhisperModel(model_path, device=device, compute_type=compute_type)
print("✅ Model loaded")

# ======================
# STATE VARIABLES
# ======================
is_recording = False
audio_buffer = []
stop_flag = threading.Event()

def record_audio():
    """Record audio continuously (16kHz mono)"""
    global audio_buffer, is_recording

    print("\n🎙️ Recording... speak now!")
    print("⏹️ Press F9 to stop")

    audio_buffer = []
    stop_flag.clear()

    chunk_size = int(CHUNK_DURATION * FS)  # 1s * 16000 = 16000 samples

    while not stop_flag.is_set():
        try:
            # Official doc: sd.rec() returns (samples, channels)
            # channels=1 for mono, dtype=float32
            chunk = sd.rec(
                frames=chunk_size,
                samplerate=FS,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Block until done

            # IMPORTANT: .squeeze() to convert 2D → 1D
            chunk = chunk.squeeze()

            # Add only if not silence (lower threshold)
            if np.max(np.abs(chunk)) > SILENCE_THRESHOLD:
                audio_buffer.append(chunk)
                print(f"   📊 {len(audio_buffer) * CHUNK_DURATION}s", end='\r')

        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

    print("\n✅ Stopped")

def transcribe():
    """Transcribe recorded audio"""
    global audio_buffer

    if not audio_buffer:
        print("⚠️ No audio")
        return

    # Combine all audio
    audio = np.concatenate(audio_buffer)
    audio_buffer = []

    duration = len(audio) / FS
    print(f"📊 Audio: {duration:.1f}s")

    if duration < 0.5:
        print("⚠️ Too short")
        return

    # Transcription with VAD filter (official doc)
    print("🔄 Transcribing...")

    # Ensure audio is 1D
    if audio.ndim > 1:
        audio = audio.squeeze()

    segments, info = model.transcribe(
        audio,
        language="en",  # English - change to "fr" for French
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,  # Recommended for faster-whisper
        vad_filter=True,  # Enable Voice Activity Detection
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            threshold=0.5,
            speech_pad_ms=400,
        ),
    )

    # Collect text
    text = " ".join(seg.text.strip() for seg in segments)

    if text:
        print(f"📝: {text}")
        print("⌨️ Typing...")

        time.sleep(0.3)
        pyautogui.write(text, interval=0.01)

        print("✅ Done!")
    else:
        print("⚠️ No text detected")

def on_f9():
    """F9 handler"""
    global is_recording

    if not is_recording:
        is_recording = True
        threading.Thread(target=record_audio, daemon=True).start()
    else:
        is_recording = False
        stop_flag.set()
        time.sleep(0.5)
        threading.Thread(target=transcribe, daemon=True).start()

# ======================
# MAIN LOOP
# ======================
print("\n" + "=" * 40)
print("✅ Ready! Press F9 to dictate, Ctrl+C to quit")
print("=" * 40)

keyboard.add_hotkey('f9', on_f9)

try:
    keyboard.wait()
except KeyboardInterrupt:
    print("\n👋")
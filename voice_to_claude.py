#!/usr/bin/env python3
"""
Voice to Claude - Dictée vocale
Appuie sur F9 pour démarrer/arrêter l'enregistrement
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

# IMPORTANT: faster-whisper nécessite 16kHz mono float32
FS = 16000  # Sample rate obligatoire: 16kHz
CHUNK_DURATION = 1  # Secondes par chunk
SILENCE_THRESHOLD = 0.005  # Seuil de silence plus bas pour éviter de couper la parole

# Vérifier le micro
try:
    device_info = sd.query_devices(kind='input')
    print(f"🎙️ Micro: {device_info['name']}")
except Exception as e:
    print(f"❌ Erreur périphérique: {e}")

print(f"🎙️ Fréquence: {FS} Hz (obligatoire pour Whisper)")
print("⌨️ F9 = Start/Stop dictée")

# ======================
# CHARGER LE MODÈLE
# ======================
print("\n🔄 Chargement du modèle...")

import torch
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    compute_type = "int8"
    print("⚠️ CPU")

# Charger le modèle local (pas de download HF)
model_path = os.path.join(os.path.dirname(__file__), "models", "faster-whisper-large-v3")
model = WhisperModel(model_path, device=device, compute_type=compute_type)
print("✅ Modèle chargé")

# ======================
# VARIABLES D'ÉTAT
# ======================
is_recording = False
audio_buffer = []
stop_flag = threading.Event()

def record_audio():
    """Enregistre l'audio en continu (16kHz mono)"""
    global audio_buffer, is_recording

    print("\n🎙️ Enregistrement... parle maintenant!")
    print("⏹️ F9 pour arrêter")

    audio_buffer = []
    stop_flag.clear()

    chunk_size = int(CHUNK_DURATION * FS)  # 3s * 16000 = 48000 samples

    while not stop_flag.is_set():
        try:
            # Doc officielle: sd.rec() retourne (samples, channels)
            # channels=1 pour mono, dtype=float32
            chunk = sd.rec(
                frames=chunk_size,
                samplerate=FS,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Bloquer jusqu'à fin

            # IMPORTANT: .squeeze() pour convertir 2D → 1D
            chunk = chunk.squeeze()

            # Ajouter seulement si pas de silence (seuil plus bas)
            if np.max(np.abs(chunk)) > SILENCE_THRESHOLD:
                audio_buffer.append(chunk)
                print(f"   📊 {len(audio_buffer) * CHUNK_DURATION}s", end='\r')

        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            break

    print("\n✅ Arrêté")

def transcribe():
    """Transcrit l'audio enregistré"""
    global audio_buffer

    if not audio_buffer:
        print("⚠️ Pas d'audio")
        return

    # Combiner tout l'audio
    audio = np.concatenate(audio_buffer)
    audio_buffer = []

    duration = len(audio) / FS
    print(f"📊 Audio: {duration:.1f}s")

    if duration < 0.5:
        print("⚠️ Trop court")
        return

    # Transcription avec VAD filter (doc officielle)
    print("🔄 Transcription...")

    # Assurer que l'audio est bien 1D
    if audio.ndim > 1:
        audio = audio.squeeze()

    segments, info = model.transcribe(
        audio,
        language="fr",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,  # Recommandé pour faster-whisper
        vad_filter=True,  # Active le Voice Activity Detection
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            threshold=0.5,
            speech_pad_ms=400,
        ),
    )

    # Collecter le texte
    text = " ".join(seg.text.strip() for seg in segments)

    if text:
        print(f"📝: {text}")
        print("⌨️ Écriture...")

        time.sleep(0.3)
        pyautogui.write(text, interval=0.01)

        print("✅ Terminé!")
    else:
        print("⚠️ Pas de texte détecté")

def on_f9():
    """Gestionnaire F9"""
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
# BOUCLE PRINCIPALE
# ======================
print("\n" + "=" * 40)
print("✅ Prêt! F9 pour dicter, Ctrl+C pour quitter")
print("=" * 40)

keyboard.add_hotkey('f9', on_f9)

try:
    keyboard.wait()
except KeyboardInterrupt:
    print("\n👋")
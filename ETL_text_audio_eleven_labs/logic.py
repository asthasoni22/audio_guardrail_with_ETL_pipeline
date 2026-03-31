# logic.py

import os
import random
from pathlib import Path
import pandas as pd
from pydub import AudioSegment, effects
import numpy as np
import librosa

import config
import utils

# ---------------- Voice Generation ---------------- #
class VoiceGenerator:
    def __init__(self, api_key=config.ELEVEN_API_KEY):
        self.api_key = api_key
        # (Here you'd set up Eleven Labs API client)

    def synthesize(self, text: str, voice_id: str, out_path: str):
        """Call Eleven Labs API to generate audio and save to file."""
        # Stub: replace with real Eleven API call
        with open(out_path, "wb") as f:
            f.write(b"FAKE_AUDIO")  

# ---------------- Augmentation Layer ---------------- #
class Augmentor:
    def modulation(self, audio: AudioSegment):
        """Apply pitch/speed modulation."""
        rate = random.uniform(0.9, 1.1)
        return audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * rate)
        }).set_frame_rate(audio.frame_rate)

    def noise_overlay(self, audio: AudioSegment, noise_path: str):
        """Overlay noise audio."""
        noise = AudioSegment.from_file(noise_path).apply_gain(-15)
        return audio.overlay(noise)

    def echo(self, audio: AudioSegment):
        """Apply echo effect."""
        return audio.overlay(audio, delay=150)

    def hidden_attack(self, human_audio: AudioSegment, attack_audio: AudioSegment):
        """Mix human-audible label0 with ultrasonic label1 attack."""
        attack_high = attack_audio.set_frame_rate(96000)  # shift frequency
        return human_audio.overlay(attack_high - 30)

# ---------------- Pipeline ---------------- #
class Pipeline:
    def __init__(self):
        self.voice_gen = VoiceGenerator()
        self.augmentor = Augmentor()

    def process_texts(self, df: pd.DataFrame, label: int, category: str, out_dir: str):
        """Convert all texts to audio for a category."""
        voice_ids = utils.load_voice_ids(category, config.VOICE_IDS_DIR)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for i, row in df.iterrows():
            text = row["text"]
            voice_id = random.choice(voice_ids)
            out_path = Path(out_dir) / f"{label}_{category}_{i}.wav"
            self.voice_gen.synthesize(text, voice_id, str(out_path))

    def apply_modulations(self, audio_dir: str):
        """Step 1: apply pitch/speed changes."""
        for file in Path(audio_dir).glob("*.wav"):
            audio = AudioSegment.from_file(file)
            modulated = self.augmentor.modulation(audio)
            modulated.export(file, format="wav")

    def apply_noise(self, audio_dir: str):
        """Step 2: overlay noise."""
        noise_files = list(Path(config.NOISE_DIR).glob("*.wav"))
        for file in Path(audio_dir).glob("*.wav"):
            audio = AudioSegment.from_file(file)
            noise_path = random.choice(noise_files)
            noisy = self.augmentor.noise_overlay(audio, str(noise_path))
            noisy.export(file, format="wav")

    def apply_attacks(self, hidden_df: pd.DataFrame, echo_df: pd.DataFrame, out_dir: str):
        """Step 3: hidden + echo attack layer."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for i, row in hidden_df.iterrows():
            human_audio = AudioSegment.silent(duration=1000)  # stub
            attack_audio = AudioSegment.silent(duration=1000)
            out_path = Path(out_dir) / f"hidden_{i}.wav"
            result = self.augmentor.hidden_attack(human_audio, attack_audio)
            result.export(out_path, format="wav")

        for i, row in echo_df.iterrows():
            human_audio = AudioSegment.silent(duration=1000)  # stub
            out_path = Path(out_dir) / f"echo_{i}.wav"
            result = self.augmentor.echo(human_audio)
            result.export(out_path, format="wav")

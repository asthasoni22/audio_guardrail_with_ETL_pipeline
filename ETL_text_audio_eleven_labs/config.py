# config.py

import os

# Paths
LABEL1_FILE = "data/label1.xlsx"
LABEL0_FILE = "data/label0.xlsx"
HIDDEN_ATTACK_FILE = "data/hidden_attack.xlsx"  # equal label1/label0 data
SJCC_ATTACK_FILE = "data/sjcc_attack.xlsx"

VOICE_IDS_DIR = "voice_ids"   # contains txt files with voice IDs for each category
OUTPUT_DIR = "outputs"       # base dir for generated audio
NOISE_DIR = "noise_inputs"   # folder with background noise wav files
HF_REPO = "username/dataset" # huggingface repo name

# Percent distribution
DISTRIBUTION = {
    "male": {"share": 0.30, "split": {"male_adult": 0.7, "male_young": 0.2, "male_old": 0.1}},
    "female": {"share": 0.30, "split": {"female_adult": 0.7, "female_young": 0.2, "female_old": 0.1}},
    "emotions": {"share": 0.30, "split": {
        "joyful": 1/6, "depressed": 1/6, "angry": 1/6, 
        "crying": 1/6, "giggle": 1/6, "whisper": 1/6
    }},
    "neutral": {"share": 0.10}
}

# API Keys
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# Audio parameters
SAMPLE_RATE = 44100

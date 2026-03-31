# utils.py

import os
import random
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi, HfFolder

# -------- Data Loaders -------- #
def load_texts(file_path: str):
    """Load text data from Excel file."""
    return pd.read_excel(file_path)

def load_voice_ids(category: str, voice_ids_dir: str):
    """Load list of voice IDs for a given category from txt file."""
    path = Path(voice_ids_dir) / f"{category}.txt"
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# -------- Samplers -------- #
def sample_texts(df, fraction: float):
    """Sample a fraction of texts from dataframe."""
    return df.sample(frac=fraction, random_state=random.randint(1, 9999))

def get_balanced_batches(df1, df0, batch_size: int):
    """Yield balanced batches of label1 and label0 texts."""
    min_len = min(len(df1), len(df0))
    df1, df0 = df1.sample(min_len), df0.sample(min_len)
    for i in range(0, min_len, batch_size):
        yield df1.iloc[i:i+batch_size], df0.iloc[i:i+batch_size]

# -------- HuggingFace Upload -------- #
def upload_to_huggingface(local_dir: str, repo_name: str, path_in_repo: str = ""):
    """Upload directory of files to HuggingFace repo."""
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_name,
        path_in_repo=path_in_repo,
        repo_type="dataset"
    )

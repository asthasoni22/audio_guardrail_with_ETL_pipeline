# main.py

import config
import utils
from logic import Pipeline

def run_pipeline():
    pipe = Pipeline()

    # --- Load datasets ---
    df1 = utils.load_texts(config.LABEL1_FILE)
    df0 = utils.load_texts(config.LABEL0_FILE)

    # Run pipeline separately for label1 and label0
    for label, df in [(1, df1), (0, df0)]:
        for category, rules in config.DISTRIBUTION.items():
            if "split" in rules:  # nested categories
                for subcat, frac in rules["split"].items():
                    subset = df.sample(frac=rules["share"] * frac)
                    pipe.process_texts(subset, label, subcat, f"{config.OUTPUT_DIR}/{label}/{subcat}")
            else:
                subset = df.sample(frac=rules["share"])
                pipe.process_texts(subset, label, category, f"{config.OUTPUT_DIR}/{label}/{category}")

        # Apply augmentations
        pipe.apply_modulations(f"{config.OUTPUT_DIR}/{label}")
        pipe.apply_noise(f"{config.OUTPUT_DIR}/{label}")

    # --- Attacks ---
    hidden_df = utils.load_texts(config.HIDDEN_ATTACK_FILE)
    sjcc_df = utils.load_texts(config.SJCC_ATTACK_FILE)

    pipe.apply_attacks(hidden_df, sjcc_df, f"{config.OUTPUT_DIR}/attacks")

    # --- Upload to HuggingFace ---
    utils.upload_to_huggingface(config.OUTPUT_DIR, config.HF_REPO)

if __name__ == "__main__":
    run_pipeline()

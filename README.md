# Audio Guardrail Prototype

A prototype developed during an internship that builds audio-based guardrails and jailbreak detectors. This repository contains notebooks and supporting materials for dataset synthesis, ETL, model fine-tuning, and evaluation.

## Key Results
- Fine-tuned Wav2Vec2 and HuBERT audio classification models (Hugging Face Transformers) on GPU-backed AWS EC2 instances.
- Achieved 80–86% jailbreak-detection accuracy across four benchmark datasets.
- Synthesized datasets using Google Cloud TTS, Coqui XTTS, and ElevenLabs (voice cloning).
- Created an ETL pipeline with `pydub`-based augmentations to generate diverse audio samples for 25 guardrail categories.

## Repository Structure
- `audio_guardrail_prototype_code/` — this folder with analysis and model notebooks.
- `ETL_text_audio_eleven_labs/` — ETL code, TTS integration, and augmentation scripts (primary pipeline).

Key files to inspect
- Notebooks: [data_analysis.ipynb](data_analysis.ipynb), [model_impli.ipynb](model_impli.ipynb), [eleven_labs_api.ipynb](eleven_labs_api.ipynb)
- ETL entry point: [ETL_text_audio_eleven_labs/main.py](ETL_text_audio_eleven_labs/main.py)
- ETL dependencies: [ETL_text_audio_eleven_labs/requirements.txt](ETL_text_audio_eleven_labs/requirements.txt)

## Approach (for GitHub readers)
This section explains the end-to-end approach so contributors and reviewers can quickly understand and reproduce the work.

1) Data synthesis and augmentation
	- Source text datasets (company policies, prompt sets) are converted to audio using multiple TTS providers: Google Cloud TTS, Coqui XTTS, and ElevenLabs API for voice cloning.
	- Each TTS voice is combined with `pydub` augmentations (gain, speed, pitch shift, background noise mixing, clipping) to simulate real-world variability.
	- The pipeline emits labeled samples across 25 predefined guardrail categories to support multi-class and binary detection tasks.

2) ETL pipeline (text → audio)
	- The ETL code lives in `ETL_text_audio_eleven_labs/`. It is modular: text ingestion → TTS adapters → augmentation → normalization → export.
	- Adapters hide provider specifics; adding a new TTS provider requires implementing a small adapter class.
	- Outputs are stored as WAV/FLAC files with a CSV manifest containing labels, source text, TTS voice, augmentation metadata, and file paths.

3) Model fine-tuning
	- Fine-tuning was performed using Hugging Face Transformers and `datasets` for data streaming and batching.
	- Pretrained audio encoders used: Wav2Vec2 and HuBERT. We appended a lightweight classification head for guardrail detection.
	- Training scripts accept config flags for batch size, learning rate, number of epochs, mixed precision (AMP), and distributed/GPU settings.

4) Evaluation
	- Evaluations use held-out synthetic test sets and external benchmark datasets where available.
	- Metrics reported: accuracy, precision, recall, F1, and confusion matrices per guardrail class.
	- Experiments were run on GPU-backed EC2 instances; training logs and checkpoints were saved to cloud storage during runs.

5) Reproducibility & recommended environment
	- Use a Python virtual environment and install dependencies from `ETL_text_audio_eleven_labs/requirements.txt`.
	- For model training, a GPU instance (>=16GB GPU memory) is recommended. Use CUDA-compatible drivers and the appropriate `transformers`/`torchaudio` builds.

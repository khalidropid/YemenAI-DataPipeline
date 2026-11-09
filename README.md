# YemenAI — Audio Emotion Recognition Pipeline

A PyTorch-based pipeline for classifying speech recordings into five emotion categories: angry, fearful, happy, neutral, and sad.

## Project structure

- `configs/default.yaml` — training and audio feature configuration.
- `scripts/prepare_data.py` — validates metadata and speaker-independent splits.
- `scripts/train.py` — model training entry point.
- `scripts/evaluate.py` — held-out test evaluation.
- `scripts/test_pipeline.py` — data/model smoke test.
- `src/audio.py` — audio loading, Log-Mel and MFCC extraction.
- `src/data.py` — metadata validation, dataset and batch padding.

## Dataset setup

Place the dataset under `data/` and provide `data/metadata_group_split.csv` with at least `file_name`, `emotion`, and `new_split` columns. The split values must be `train`, `validation`, or `test`. The loader enforces speaker-independent splits to avoid speaker leakage.

The dataset is not included. Ensure you have permission to use and redistribute the audio and metadata.

## Environment

Python, PyTorch, NumPy, librosa, pandas, PyYAML, and matplotlib are used.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy librosa pandas pyyaml matplotlib
```

Install the PyTorch build appropriate for your CUDA environment if using a GPU.

## Run

From the repository root:

```bash
python scripts/prepare_data.py --data-dir data
python scripts/test_pipeline.py --data-dir data --batch-size 4
python scripts/train.py --data-dir data --model cnn
python scripts/evaluate.py --checkpoint artifacts/cnn/best.pt --model cnn --data-dir data
```

Supported model names referenced by the scripts: `mlp`, `cnn`, `lstm`, `gru`, and `cnn_bilstm_attention`.

## Important completeness note

The supplied archive contains seven Python/configuration files. The scripts import `src.models` and `src.engine`, but `src/models.py` and `src/engine.py` are absent from the archive. The project therefore needs those modules before training/evaluation can run. A dependency lock/requirements file, dataset, and split CSV are also not included.

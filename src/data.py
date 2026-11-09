from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .audio import load_audio, log_mel, mfcc

LABELS = ["angry", "fearful", "happy", "neutral", "sad"]
ALIASES = {
    "anger": "angry", "angry": "angry",
    "fear": "fearful", "fearful": "fearful",
    "happiness": "happy", "happy": "happy",
    "sadness": "sad", "sad": "sad",
    "neutrality": "neutral", "neutral": "neutral",
}


def speaker_from_name(name: str):
    """YSED source filenames encode participant ID in the final numeric field."""
    match = re.search(r"-(\d+)\.wav$", Path(str(name)).name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_metadata(data_dir: str | Path, split_csv: str | Path | None = None) -> pd.DataFrame:
    """Load YSED metadata and the verified speaker-independent split.

    The split CSV is required by default to prevent accidentally using the
    original speaker-overlapping train/validation/test assignment.
    """
    root = Path(data_dir).resolve()
    split_path = Path(split_csv) if split_csv else root / "metadata_group_split.csv"
    if not split_path.is_absolute():
        split_path = Path.cwd() / split_path

    if split_path.exists():
        df = pd.read_csv(split_path)
        split_col = "new_split"
    else:
        raise FileNotFoundError(
            f"Speaker-independent split file not found: {split_path}\n"
            "Create data/metadata_group_split.csv first. Do not train using the "
            "original metadata.csv split because speakers overlap across splits."
        )

    required = {"file_name", "emotion", split_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split metadata is missing columns: {sorted(missing)}")

    source_col = "source_filename" if "source_filename" in df.columns else "file_name"
    if "speaker_id" not in df.columns:
        df["speaker_id"] = df[source_col].map(speaker_from_name)

    df["emotion"] = df["emotion"].astype(str).str.lower().map(ALIASES)
    df = df.dropna(subset=["emotion", "speaker_id"]).copy()
    df["speaker_id"] = df["speaker_id"].astype(int)
    df["path"] = df["file_name"].astype(str).map(lambda p: str((root / p).resolve()))
    exists = df["path"].map(lambda p: Path(p).is_file())
    missing_files = int((~exists).sum())
    if missing_files:
        raise FileNotFoundError(f"{missing_files} audio paths from split metadata do not exist.")
    df = df.rename(columns={split_col: "dataset_split"})
    df = df[df["dataset_split"].isin(["train", "validation", "test"])].reset_index(drop=True)

    speaker_sets = {
        name: set(group["speaker_id"])
        for name, group in df.groupby("dataset_split")
    }
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = speaker_sets.get(a, set()) & speaker_sets.get(b, set())
        if overlap:
            raise ValueError(f"Speaker leakage detected between {a} and {b}: {sorted(overlap)}")

    return df


class AudioEmotionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature: str = "mel",
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 256,
        augment: bool = False,
    ):
        self.frame = frame.reset_index(drop=True)
        self.feature = feature
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        self.label_to_id = {name: i for i, name in enumerate(LABELS)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        y = load_audio(row["path"], self.sample_rate)

        if self.augment:
            if np.random.rand() < 0.5:
                y = y * np.random.uniform(0.85, 1.15)
            if np.random.rand() < 0.20:
                noise = np.random.normal(0, 0.0015, size=y.shape).astype(np.float32)
                y = y + noise

        if self.feature == "mel":
            x = log_mel(y, self.sample_rate, self.n_mels, self.n_fft, self.hop_length)
            x = torch.from_numpy(x).unsqueeze(0)
        elif self.feature == "mfcc":
            x = mfcc(y, self.sample_rate, self.n_mfcc, self.n_fft, self.hop_length).T
            x = torch.from_numpy(x)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature}")

        x = (x - x.mean()) / x.std().clamp_min(1e-6)
        label = self.label_to_id[row["emotion"]]
        return x.float(), torch.tensor(label, dtype=torch.long)


def pad_audio_batch(batch):
    """Pad variable-length features to the longest item in this batch only."""
    features, labels = zip(*batch)
    max_time = max(x.shape[-1] if x.ndim == 3 else x.shape[0] for x in features)

    padded = []
    for x in features:
        if x.ndim == 3:
            padded.append(F.pad(x, (0, max_time - x.shape[-1])))
        else:
            padded.append(F.pad(x, (0, 0, 0, max_time - x.shape[0])))

    return torch.stack(padded), torch.stack(labels)

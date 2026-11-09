from __future__ import annotations

import numpy as np
import librosa


def load_audio(path_or_file, sample_rate: int = 16000) -> np.ndarray:
    """Load the complete recording as mono float32 audio at the target rate.

    No fixed-duration crop or padding is applied here. Batch padding is handled
    later by pad_audio_batch, so no recording content is discarded.
    """
    y, _ = librosa.load(path_or_file, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path_or_file}")
    return y.astype(np.float32, copy=False)


def log_mel(
    y: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """Return a [n_mels, time] log-Mel spectrogram."""
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    return librosa.power_to_db(spec, ref=np.max).astype(np.float32)


def mfcc(
    y: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    return librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    ).astype(np.float32)


def normalize_feature(x: np.ndarray) -> np.ndarray:
    """Per-recording normalization; computed independently for each sample."""
    mean = float(x.mean())
    std = float(x.std())
    return ((x - mean) / max(std, 1e-6)).astype(np.float32)

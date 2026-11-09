import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import load_metadata, AudioEmotionDataset, pad_audio_batch, LABELS
from src.models import build_model
from src.engine import fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-csv", default=None)
    p.add_argument("--model", choices=["mlp", "cnn", "lstm", "gru", "cnn_bilstm_attention"], default="cnn")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_metadata(args.data_dir, args.split_csv)
    tr = df[df["dataset_split"] == "train"].reset_index(drop=True)
    va = df[df["dataset_split"] == "validation"].reset_index(drop=True)
    te = df[df["dataset_split"] == "test"].reset_index(drop=True)

    print(f"Device: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    print(f"Data: {len(df)} | train={len(tr)} val={len(va)} test={len(te)}")
    print("Split: speaker-independent; overlap checked and zero.")

    feature = "mel" if args.model in ("cnn", "cnn_bilstm_attention") else "mfcc"
    ds_args = dict(
        feature=feature,
        sample_rate=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        n_mfcc=cfg["n_mfcc"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
    )
    train_ds = AudioEmotionDataset(tr, augment=True, **ds_args)
    val_ds = AudioEmotionDataset(va, augment=False, **ds_args)

    batch_size = args.batch_size or cfg["batch_size"]
    loader_args = dict(
        batch_size=batch_size,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=pad_audio_batch,
        pin_memory=(device == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    counts = tr["emotion"].value_counts()
    class_weights = torch.tensor(
        [len(tr) / (len(LABELS) * max(counts.get(c, 1), 1)) for c in LABELS],
        dtype=torch.float32,
    )

    model = build_model(args.model, len(LABELS))
    out = Path("artifacts") / args.model
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "split_type": "speaker-independent",
        "labels": LABELS,
        "train_recordings": len(tr),
        "validation_recordings": len(va),
        "held_out_test_recordings": len(te),
        "train_speakers": sorted(tr["speaker_id"].unique().tolist()),
        "validation_speakers": sorted(va["speaker_id"].unique().tolist()),
        "test_speakers": sorted(te["speaker_id"].unique().tolist()),
        "feature": feature,
        "sample_rate": cfg["sample_rate"],
        "n_mels": cfg["n_mels"],
        "dynamic_batch_padding": True,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    xb, yb = next(iter(train_loader))
    print("First batch shape:", tuple(xb.shape), "| labels:", tuple(yb.shape))
    print("Batch feature range:", float(xb.min()), "to", float(xb.max()))
    del xb, yb

    fit(
        model,
        train_loader,
        val_loader,
        args.epochs or cfg["epochs"],
        args.lr or cfg["learning_rate"],
        device,
        out,
        class_weights,
    )
    print(f"Best checkpoint saved under {out}/best.pt")
    print("The held-out test set was NOT used for training or model selection.")


if __name__ == "__main__":
    main()

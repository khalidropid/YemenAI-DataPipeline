import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import load_metadata, AudioEmotionDataset, pad_audio_batch, LABELS
from src.models import build_model
from src.engine import metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True, choices=["mlp", "cnn", "lstm", "gru", "cnn_bilstm_attention"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split-csv", default=None)
    args = p.parse_args()

    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_metadata(args.data_dir, args.split_csv)
    test = df[df["dataset_split"] == "test"].reset_index(drop=True)
    feature = "mel" if args.model in ("cnn", "cnn_bilstm_attention") else "mfcc"
    ds = AudioEmotionDataset(
        test,
        feature=feature,
        sample_rate=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        n_mfcc=cfg["n_mfcc"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        augment=False,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        collate_fn=pad_audio_batch,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, len(LABELS))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().tolist()
            ys.extend(y.tolist())
            ps.extend(pred)

    result = metrics(ys, ps)
    result["split_type"] = "speaker-independent"
    result["test_recordings"] = len(test)
    result["test_speakers"] = int(test["speaker_id"].nunique())
    out = Path(args.checkpoint).parent
    (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 6))
    plt.imshow(result["confusion_matrix"], cmap="Blues")
    plt.xticks(range(5), LABELS, rotation=35)
    plt.yticks(range(5), LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Speaker-independent test confusion matrix")
    for i in range(5):
        for j in range(5):
            plt.text(j, i, result["confusion_matrix"][i][j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    print(json.dumps({
        k: v for k, v in result.items()
        if k not in ("confusion_matrix", "classification_report")
    }, indent=2))


if __name__ == "__main__":
    main()

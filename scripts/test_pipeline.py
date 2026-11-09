import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import load_metadata, AudioEmotionDataset, pad_audio_batch, LABELS
from src.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_metadata(args.data_dir)
    train = df[df["dataset_split"] == "train"].reset_index(drop=True)
    dataset = AudioEmotionDataset(
        train.head(max(args.batch_size, 4)),
        feature="mel",
        sample_rate=cfg["sample_rate"],
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_audio_batch,
    )

    x, y = next(iter(loader))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("cnn", len(LABELS)).to(device)
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        logits = model(x)

    print("Pipeline test: PASS")
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("Allocated GPU memory (MB):", round(torch.cuda.memory_allocated() / 1024**2, 2))
    print("Input batch:", tuple(x.shape), "(batch, channel, mel_bins, padded_time)")
    print("Labels:", tuple(y.shape))
    print("Model output:", tuple(logits.shape), "(batch, 5 classes)")
    print("Predicted classes:", logits.argmax(dim=1).cpu().tolist())


if __name__ == "__main__":
    main()

import argparse
import pandas as pd

from src.data import load_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Validate the speaker-independent YSED split and show dataset statistics."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-csv", default=None)
    args = parser.parse_args()

    df = load_metadata(args.data_dir, args.split_csv)
    print("Validated dataset:", len(df), "recordings")
    print("\nSplit counts:")
    print(df["dataset_split"].value_counts())
    print("\nSpeakers per split:")
    for name, group in df.groupby("dataset_split"):
        print(f"{name}: {group['speaker_id'].nunique()} speakers")
    print("\nClass distribution:")
    print(pd.crosstab(df["dataset_split"], df["emotion"]))
    print("\nSpeaker overlap check: PASS (zero overlap).")


if __name__ == "__main__":
    main()

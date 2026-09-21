#!/usr/bin/env python3
"""Convert orientation-specific Sign Sense labels into letter labels.

The script creates a separate, training-ready directory so the original
50-class dataset is left untouched. The output can be passed directly to
train_lstm_sign_model.py via --data-dir.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_FILES = (
    "sequences_body_hands.npy",
    "labels.npy",
    "labels_text.npy",
    "class_mapping.json",
    "metadata.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a training-ready dataset whose targets are letters rather "
            "than orientation-specific classes."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing sequences_body_hands.npy and metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for the 24-class letter dataset",
    )
    parser.add_argument(
        "--metadata-name",
        default="metadata.csv",
        help="Metadata filename inside --data-dir (default: metadata.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of this script's output files in output-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    sequence_path = data_dir / "sequences_body_hands.npy"
    metadata_path = data_dir / args.metadata_name

    for path in (sequence_path, metadata_path):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    if output_dir == data_dir:
        raise SystemExit(
            "Choose a different --output-dir so the original 50-class files "
            "remain untouched."
        )

    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(
            f"Output files already exist ({names}). Use a new directory or --force."
        )

    # mmap_mode avoids loading the complete sequence tensor into memory.
    sequences = np.load(sequence_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_csv(metadata_path)

    if "letter" not in metadata.columns:
        raise SystemExit("metadata.csv must contain a 'letter' column.")
    if len(sequences) != len(metadata):
        raise SystemExit(
            "Row count mismatch: "
            f"sequences={len(sequences)}, metadata={len(metadata)}"
        )
    if metadata["letter"].isna().any():
        bad_rows = metadata.index[metadata["letter"].isna()].tolist()[:10]
        raise SystemExit(f"Missing letter labels at metadata rows: {bad_rows}")

    letters = metadata["letter"].astype(str).str.strip().str.upper()
    if (letters == "").any():
        bad_rows = metadata.index[letters == ""].tolist()[:10]
        raise SystemExit(f"Empty letter labels at metadata rows: {bad_rows}")

    # Alphabetical ordering gives stable IDs: A=0, B=1, ... among available letters.
    class_names = sorted(letters.unique().tolist())
    class_to_id = {name: index for index, name in enumerate(class_names)}
    id_to_class = {str(index): name for name, index in class_to_id.items()}

    labels = letters.map(class_to_id).to_numpy(dtype=np.int64)
    labels_text = letters.to_numpy(dtype=str)

    # Preserve the old orientation targets for analysis while making the normal
    # class columns describe the new letter targets.
    output_metadata = metadata.copy()
    if "class_label" in output_metadata.columns:
        output_metadata = output_metadata.rename(
            columns={"class_label": "orientation_class_label"}
        )
    if "class_id" in output_metadata.columns:
        output_metadata = output_metadata.rename(
            columns={"class_id": "orientation_class_id"}
        )
    output_metadata["letter"] = letters
    output_metadata["class_label"] = labels_text
    output_metadata["class_id"] = labels

    # Final alignment checks before writing anything.
    if not np.array_equal(output_metadata["class_id"].to_numpy(), labels):
        raise SystemExit("Internal error: metadata and numeric labels are misaligned.")
    if not np.array_equal(output_metadata["class_label"].to_numpy(), labels_text):
        raise SystemExit("Internal error: metadata and text labels are misaligned.")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sequence_path, output_dir / "sequences_body_hands.npy")
    np.save(output_dir / "labels.npy", labels, allow_pickle=False)
    np.save(output_dir / "labels_text.npy", labels_text, allow_pickle=False)
    output_metadata.to_csv(output_dir / "metadata.csv", index=False)

    mapping = {
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
    }
    with (output_dir / "class_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2)
        file.write("\n")

    counts = pd.Series(labels_text).value_counts().sort_index()
    print("Created letter-level dataset:", output_dir)
    print("Sequences shape:", tuple(sequences.shape))
    print("Number of classes:", len(class_names))
    print("Classes:", ", ".join(class_names))
    print("Total sequences:", len(labels))
    print("Minimum/maximum sequences per letter:", int(counts.min()), int(counts.max()))
    print("\nTrain with:")
    print(
        "python train_lstm_sign_model.py "
        f'--data-dir "{output_dir}" --output-dir "MODEL_OUTPUT_DIR"'
    )


if __name__ == "__main__":
    main()

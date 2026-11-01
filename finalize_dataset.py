"""Build the final single-speaker Piper training folder from dataset/verified.csv.

Writes <out>/wavs/<clip_id>.wav   (peak-normalized to -3 dBFS)
       <out>/metadata.csv         (`<clip_id>.wav|text`, training set)
       <out>/val.csv              (held-out clips, same format)

Usage: python finalize_dataset.py --speaker speaker1 [--min-sim 85] [--use-whisper-text] [--out dataset/train]
"""
import argparse
import random
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "dataset"


def peak_normalize(src, dst, target_db=-3.0):
    with wave.open(str(src), "rb") as r:
        params, raw = r.getparams(), r.readframes(r.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    peak = np.abs(x).max()
    if peak > 0:
        x *= (10 ** (target_db / 20) * 32767) / peak
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--min-sim", type=float, default=85.0, help="drop clips whose subtitle/Whisper similarity is lower")
    ap.add_argument("--use-whisper-text", action="store_true", help="use Whisper text (it has punctuation) instead of the subtitle")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--out", type=Path, default=ROOT / "train")
    args = ap.parse_args()

    kept = []
    for line in (ROOT / "verified.csv").read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) != 5 or parts[1] != args.speaker or float(parts[4]) < args.min_sim:
            continue
        text = parts[3] if args.use_whisper_text else parts[2]
        if text.strip():
            kept.append((parts[0], text.strip()))

    (args.out / "wavs").mkdir(parents=True, exist_ok=True)
    for cid, _ in kept:
        peak_normalize(ROOT / "wavs" / f"{cid}.wav", args.out / "wavs" / f"{cid}.wav")

    random.Random(0).shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_fraction))
    for name, rows in (("val.csv", kept[:n_val]), ("metadata.csv", kept[n_val:])):
        (args.out / name).write_text("".join(f"{c}.wav|{t}\n" for c, t in rows), encoding="utf-8")
    print(f"{len(kept) - n_val} train + {n_val} val clips -> {args.out}")


if __name__ == "__main__":
    main()

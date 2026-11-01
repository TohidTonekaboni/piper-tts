"""Check every clip's transcript against a Whisper transcription of its audio.

Reads  dataset/metadata.csv (`clip_id|speaker|text`) and dataset/wavs/<clip_id>.wav
Writes dataset/verified.csv (`clip_id|speaker|text|whisper_text|similarity`), resumable.

Usage: python verify_dataset.py [--speaker speaker1] [--limit N] [--model large-v3] [--device cpu|cuda]
"""
import argparse
import csv
import re
from pathlib import Path

from faster_whisper import WhisperModel
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent / "dataset"


def canon(text):
    # Compare words only: ignore punctuation, ZWNJ and spacing differences.
    text = text.replace("‌", " ")
    return re.sub(r"[^\w]+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", help="only verify this speaker's clips")
    ap.add_argument("--limit", type=int, help="only verify the first N clips (for a trial run)")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out = ROOT / "verified.csv"
    done = set()
    if out.exists():
        with open(out, encoding="utf-8", newline="") as f:
            done = {r[0] for r in csv.reader(f, delimiter="|")}

    rows = [l.rstrip("\n").split("|", 2) for l in (ROOT / "metadata.csv").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if len(r) == 3 and (not args.speaker or r[1] == args.speaker)]
    if args.limit:
        rows = rows[:args.limit]

    model = WhisperModel(args.model, device=args.device, compute_type="default")
    with open(out, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for i, (cid, speaker, text) in enumerate(rows):
            if cid in done:
                continue
            segs, _ = model.transcribe(str(ROOT / "wavs" / f"{cid}.wav"), language="fa", beam_size=5)
            heard = " ".join(s.text.strip() for s in segs).replace("|", " ")
            sim = fuzz.ratio(canon(text), canon(heard))
            w.writerow([cid, speaker, text, heard, f"{sim:.1f}"])
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()

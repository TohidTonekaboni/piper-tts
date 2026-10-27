"""Turn downloaded voices + timestamped texts into a Piper (LJSpeech-style) dataset.

Reads  dataset/voices/<video_id>.wav and dataset/texts/<video_id>.txt (`start|end|speaker|text`).
Writes dataset/wavs/<video_id>_<n>.wav (one clip per merged subtitle group)
       dataset/metadata.csv            (`clip_id|speaker|text`, no extension in the id)

Usage: python make_dataset.py [dataset_dir]
"""
import re
import shutil
import sys
import wave
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "dataset"
VOICES = ROOT / "voices"
TEXTS = ROOT / "texts"
WAVS = ROOT / "wavs"
METADATA = ROOT / "metadata.csv"

MIN_SEC = 2.0      # keep merging subtitles until a clip is at least this long
MAX_SEC = 15.0     # never build a clip longer than this
DROP_BELOW = 1.0   # discard clips shorter than this
MAX_GAP = 0.5      # only merge subtitles at most this far apart
PAD = 0.1          # margin kept around each clip (subtitle timing is approximate)

AR_TO_FA = str.maketrans({"ي": "ی", "ك": "ک", "ة": "ه", "ؤ": "و", "أ": "ا", "إ": "ا"})
UNUSABLE = re.compile(r"[0-9۰-۹٠-٩A-Za-z\[\]()♪]")  # digits, Latin, markers like [موسیقی]


def normalize(text):
    text = text.translate(AR_TO_FA).replace("ـ", "")
    text = re.sub(r"[ً-ٰٟ]", "", text)  # diacritics
    text = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", text)  # invisible bidi marks
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def read_subtitles(path):
    """Return ([(start, end, text)], speaker) for one text file."""
    subs, speakers = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            subs.append((float(parts[0]), float(parts[1]), parts[3]))
            speakers.add(parts[2])
    if len(speakers) != 1:
        raise ValueError(f"{path.name}: expected one speaker, found {sorted(speakers)}")
    return subs, speakers.pop()


def group_subtitles(subs):
    """Merge neighbouring subtitles into clips of roughly MIN_SEC..MAX_SEC."""
    groups, cur = [], None
    for start, end, text in subs:
        if cur:
            fits = end - cur[0] <= MAX_SEC and start - cur[1] <= MAX_GAP
            if cur[1] - cur[0] < MIN_SEC and fits:
                cur[1] = max(cur[1], end)
                cur[2].append(text)
                continue
            groups.append(cur)
        cur = [start, end, [text]]
    if cur:
        groups.append(cur)
    return groups


def main():
    if not VOICES.is_dir() or not TEXTS.is_dir():
        sys.exit(f"missing {VOICES} or {TEXTS}; run data_gathering.py first")
    shutil.rmtree(WAVS, ignore_errors=True)  # fully generated, rebuilt each run
    WAVS.mkdir(parents=True)

    speakers = set()
    rows, total_sec, skipped = [], 0.0, {"short": 0, "long": 0, "unusable text": 0}
    for txt in sorted(TEXTS.glob("*.txt")):
        voice = VOICES / f"{txt.stem}.wav"
        if not voice.exists():
            print(f"skip {txt.name}: no matching voice file", file=sys.stderr)
            continue
        with wave.open(str(voice), "rb") as src:
            params, rate = src.getparams(), src.getframerate()
            frames = src.readframes(src.getnframes())
        frame_bytes = params.nchannels * params.sampwidth
        n_frames = len(frames) // frame_bytes

        subs, speaker = read_subtitles(txt)
        n = 0
        for start, end, parts in group_subtitles(subs):
            text = normalize(" ".join(parts))
            if end - start < DROP_BELOW:
                skipped["short"] += 1
            elif end - start > MAX_SEC:
                skipped["long"] += 1
            elif not text or UNUSABLE.search(text):
                skipped["unusable text"] += 1
            else:
                a = max(0, int((start - PAD) * rate))
                b = min(n_frames, int((end + PAD) * rate))
                n += 1
                clip_id = f"{txt.stem}_{n:04d}"
                with wave.open(str(WAVS / f"{clip_id}.wav"), "wb") as dst:
                    dst.setparams(params)
                    dst.writeframes(frames[a * frame_bytes:b * frame_bytes])
                rows.append(f"{clip_id}|{speaker}|{text}\n")
                total_sec += end - start
        speakers.add(speaker)
        print(f"{txt.stem}: {n} clips ({speaker})")

    METADATA.write_text("".join(rows), encoding="utf-8")
    print(f"{len(rows)} clips, {total_sec / 3600:.2f} h, {len(speakers)} speakers -> {METADATA}")
    print("skipped:", ", ".join(f"{k}={v}" for k, v in skipped.items()))


if __name__ == "__main__":
    main()

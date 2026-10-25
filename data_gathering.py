"""Download audio + Farsi subtitles for every YouTube URL in dataset/source.json.

Audio  -> dataset/voices/<video_id>.wav
Text   -> dataset/texts/<video_id>.txt
After each URL succeeds, its entry in source.json gets `voice_name` and `text_name`
filled in. Entries that already have both (and the files exist) are skipped, so the
script can be interrupted and re-run to continue where it stopped.
"""
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yt_dlp

ROOT = Path(__file__).resolve().parent / "dataset"
SOURCE = ROOT / "source.json"
VOICES = ROOT / "voices"
TEXTS = ROOT / "texts"
LANG = "fa"


def save_source(entries):
    # Atomic write so an interruption can't corrupt source.json.
    tmp = SOURCE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)
    os.replace(tmp, SOURCE)


def is_done(entry):
    v, t = entry.get("voice_name"), entry.get("text_name")
    return bool(v and t and (VOICES / v).exists() and (TEXTS / t).exists())


def vtt_to_text(path):
    lines, prev = [], None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if (not line or line == "WEBVTT" or "-->" in line
                or line.startswith(("Kind:", "Language:", "NOTE"))
                or line.isdigit()):
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()  # inline timing/style tags
        if line and line != prev:
            lines.append(line)
            prev = line
    return "\n".join(lines) + "\n"


def process(entry, work):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,  # fallback when no manual subtitles exist
        "subtitleslangs": [LANG],
        "subtitlesformat": "vtt",
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": {"extractaudio": ["-ac", "1", "-ar", "22050", "-sample_fmt", "s16"]},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(entry["url"], download=True)
    vid = info["id"]

    wav = work / f"{vid}.wav"
    subs = sorted(work.glob(f"{vid}.{LANG}*.vtt"))
    if not wav.exists():
        raise RuntimeError("audio was not produced")
    if not subs:
        raise RuntimeError(f"no '{LANG}' subtitles available")

    text = vtt_to_text(subs[0])
    if not text.strip():
        raise RuntimeError("subtitle file is empty")

    VOICES.mkdir(parents=True, exist_ok=True)
    TEXTS.mkdir(parents=True, exist_ok=True)
    (TEXTS / f"{vid}.txt").write_text(text, encoding="utf-8")
    shutil.move(str(wav), VOICES / f"{vid}.wav")  # audio last: marks completion
    entry["voice_name"] = f"{vid}.wav"
    entry["text_name"] = f"{vid}.txt"


def main():
    entries = json.loads(SOURCE.read_text(encoding="utf-8"))
    failed = 0
    for i, entry in enumerate(entries):
        if not entry.get("url"):
            continue
        if is_done(entry):
            print(f"[{i}] skip (done): {entry['url']}")
            continue
        print(f"[{i}] downloading: {entry['url']}")
        with tempfile.TemporaryDirectory() as work:
            try:
                process(entry, Path(work))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failed += 1
                print(f"[{i}] FAILED: {e}", file=sys.stderr)
                continue
        save_source(entries)  # persist progress after every URL
        print(f"[{i}] saved {entry['voice_name']}, {entry['text_name']}")
    print(f"finished, {failed} failed")


if __name__ == "__main__":
    main()

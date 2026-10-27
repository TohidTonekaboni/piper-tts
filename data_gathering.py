"""Download audio + Farsi subtitles for every YouTube URL in dataset/source.json.

Audio  -> dataset/voices/<video_id>.wav
Text   -> dataset/texts/<video_id>.txt, one subtitle per line: `start|end|text`
          (start/end in seconds from the beginning of the audio)
After each URL succeeds, its entry in source.json gets `voice_name` and `text_name`
filled in. Entries that already have both (and the files exist) are skipped, so the
script can be interrupted and re-run to continue where it stopped.

If YouTube answers "Sign in to confirm you're not a bot", pass your logged-in cookies:
    python data_gathering.py --cookies-from-browser chrome
    python data_gathering.py --cookies cookies.txt
"""
import argparse
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


CUE_TIME = re.compile(r"((?:\d+:)?\d+:\d+[.,]\d+)\s*-->\s*((?:\d+:)?\d+:\d+[.,]\d+)")


def to_seconds(stamp):
    secs = 0.0
    for part in stamp.replace(",", ".").split(":"):
        secs = secs * 60 + float(part)
    return secs


def parse_cues(path):
    """Yield (start, end, [text lines]) for every cue in a VTT file."""
    cue = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = CUE_TIME.search(line)
        if m:
            if cue:
                yield cue
            cue = (to_seconds(m.group(1)), to_seconds(m.group(2)), [])
            continue
        if cue is None:
            continue  # header / NOTE blocks before the first cue
        if not line:
            yield cue
            cue = None
            continue
        line = re.sub(r"<[^>]+>", "", line).strip()  # inline timing/style tags
        if line:
            cue[2].append(line)
    if cue:
        yield cue


def vtt_to_text(path):
    """Return one `start|end|text` line per subtitle (times in seconds).

    YouTube auto-captions repeat the previous line inside the next cue (rolling
    captions); a repeated line is merged into the original one instead of being
    emitted twice.
    """
    entries, prev = [], {}
    for start, end, lines in parse_cues(path):
        cur = {}
        # Lines carried over into a cue that also adds new text are only still on
        # screen; the speech ended earlier, so only extend when nothing is new.
        has_new = any(l not in prev for l in lines)
        for line in lines:
            e = prev.get(line)
            if e and start <= e[1] + 0.05:
                if not has_new:
                    e[1] = max(e[1], end)
            else:
                e = [start, end, line]
                entries.append(e)
            cur[line] = e
        prev = cur
    return "".join(f"{s:.3f}|{e:.3f}|{t}\n" for s, e, t in entries)


def process(entry, work, auth):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(work / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,  # fallback when no manual subtitles exist
        "subtitleslangs": [LANG],
        "subtitlesformat": "vtt",
        "noplaylist": True,
        **auth,
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cookies", help="Netscape-format cookies.txt exported from your browser")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER",
                    help="read cookies from an installed browser (chrome, firefox, safari, ...)")
    args = ap.parse_args()
    auth = {}
    if args.cookies:
        auth["cookiefile"] = args.cookies
    if args.cookies_from_browser:
        auth["cookiesfrombrowser"] = (args.cookies_from_browser,)

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
                process(entry, Path(work), auth)
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

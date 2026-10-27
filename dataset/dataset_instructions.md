# Preparing a Farsi Dataset for Fine-Tuning Piper TTS

This guide describes how to build a single-speaker, LJSpeech-style dataset for fine-tuning Piper (VITS-based) on Farsi.

> Commands and flags can change between Piper versions. Always cross-check with the current training docs in the Piper repository (`OHF-Voice/piper1-gpl`).

---

## 1. Goal and Data Size

| Amount of clean audio | Expected outcome |
|---|---|
| 30 min - 2 h | Works when fine-tuning from a strong checkpoint; pronunciation errors are likely |
| 3 - 10 h | Good, stable quality (recommended target) |
| 10+ h | Best quality; training from scratch becomes realistic |

Use **one speaker** and **one speaking style**. Mixing speakers or styles in a single-speaker model degrades quality.

---

## 2. Folder Layout

```
dataset/
  wavs/
    0001.wav
    0002.wav
    ...
  metadata.csv
```

`metadata.csv` is pipe-separated: `file_id|text` (no file extension in the ID).

```
0001|سلام، امروز هوا خیلی خوب است.
0002|کتاب را روی میز گذاشتم.
```

If you need a multi-speaker dataset, check the current README for the supported column format (for example `id|speaker|text`).

---

## 3. Audio Requirements

- **Format:** mono, 16-bit PCM WAV
- **Sample rate:** must match the base checkpoint (22,050 Hz for "medium" voices, 16,000 Hz for "low")
- **Clip length:** about 2-15 seconds, one sentence or phrase per clip (avoid clips under 1 s or over 20 s)
- **Recording quality:** quiet room, same microphone and distance, no music, no reverb, no clipping
- **Silence:** trim long leading/trailing silence but keep a natural margin of roughly 0.1-0.3 s
- **Loudness:** consistent level across clips (for example about -23 LUFS, or peak-normalize to -3 dB)

Convert a single file:

```bash
ffmpeg -i input.wav -ac 1 -ar 22050 -sample_fmt s16 wavs/0001.wav
```

Convert a whole folder:

```bash
mkdir -p wavs
i=1
for f in raw/*.wav; do
  ffmpeg -y -i "$f" -ac 1 -ar 22050 -sample_fmt s16 "wavs/$(printf '%04d' $i).wav"
  i=$((i+1))
done
```

---

## 4. Transcript Requirements

The text must match **exactly what was spoken**. This is the most important factor for quality.

### 4.1 Character normalization
- Convert Arabic `ي` / `ك` to Persian `ی` / `ک`
- Unify hamza and alef variants consistently
- Remove tatweel (`ـ`) and stray diacritics, unless you deliberately keep diacritics (see 4.5)
- Normalize Persian/Arabic/Latin digits (see 4.3)

### 4.2 ZWNJ (half-space, `‌`)
Use it consistently: `می‌روم`, `کتاب‌ها`, `خانه‌ام`.

### 4.3 Numbers, dates, currency, abbreviations
Write them exactly as spoken. Piper does not handle digits reliably.

| Written | Spoken form to put in the transcript |
|---|---|
| ۱۲۰۰ | هزار و دویست |
| ۱۴۰۴/۰۷/۳۰ | سی ام مهر هزار و چهارصد و چهار |
| ۵۰٪ | پنجاه درصد |

### 4.4 Punctuation and foreign text
- Keep commas and periods (they influence pauses)
- Remove emojis, URLs, hashtags and markup
- Transliterate Latin words into Persian as they are pronounced, or remove the clip

### 4.5 Vowels and pronunciation
Persian text omits short vowels, so espeak-ng may mispronounce ambiguous words. If you hear errors:
- Add a vowel-restoring or G2P step before phonemization, or
- Add diacritics to the problem words in the transcript, or
- Add more training examples of those words

---

## 5. Normalization Script (starter)

```python
import re, csv, sys

AR_TO_FA = str.maketrans({"ي": "ی", "ك": "ک", "ة": "ه", "ؤ": "و", "أ": "ا", "إ": "ا"})
FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

def normalize(t: str) -> str:
    t = t.translate(AR_TO_FA)
    t = t.replace("ـ", "")                              # tatweel
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)         # diacritics (remove if not intentionally kept)
    t = re.sub(r"https?://\S+", "", t)                  # URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t

# NOTE: number-to-words must be done separately (e.g. with a Persian num2words library).
# After normalize(), flag any remaining digits for manual or automated conversion.

rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        fid, text = line.rstrip("\n").split("|", 1)
        text = normalize(text)
        if re.search(r"[0-9A-Za-z]", text.translate(FA_DIGITS)):
            print(f"WARNING: digits/Latin remain in {fid}: {text}")
        rows.append((fid, text))

with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
    for fid, text in rows:
        f.write(f"{fid}|{text}\n")
```

Usage:

```bash
python normalize.py raw_metadata.csv dataset/metadata.csv
```

---

## 6. Audio Validation Script (starter)

```python
import wave, glob, os

EXPECTED_SR = 22050
bad = []
total = 0.0

for path in sorted(glob.glob("dataset/wavs/*.wav")):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        dur = w.getnframes() / sr
    total += dur
    if sr != EXPECTED_SR or ch != 1 or sw != 2 or dur < 1.0 or dur > 20.0:
        bad.append((os.path.basename(path), sr, ch, sw, round(dur, 2)))

print(f"Total audio: {total/3600:.2f} h")
print(f"Problem files: {len(bad)}")
for b in bad:
    print(b)
```

Also verify that every line in `metadata.csv` has a matching WAV file and vice versa.

---

## 7. Quality Checklist

- [ ] Single speaker, consistent mic and room
- [ ] All clips mono, 16-bit, correct sample rate
- [ ] Clip lengths mostly 2-15 s
- [ ] No clipping, noise, music, or long silences
- [ ] Transcripts match the audio word for word
- [ ] Characters normalized (ی / ک, ZWNJ) and numbers spelled out
- [ ] No emojis, URLs, or leftover Latin text
- [ ] Random 5-10% of clips listened to by a human
- [ ] Suspicious transcripts checked (optional: compare with a Persian ASR model such as Whisper and review large mismatches)
- [ ] 1-2% of clips (or a few test sentences) held out for evaluation, including numbers, dates and loanwords

---

## 8. Next Steps

1. Choose a base checkpoint with a matching sample rate and quality tier (a Persian one if available; otherwise a strong multilingual/English one).
2. Run Piper's preprocessing step to phonemize the text (espeak-ng, `fa` voice) and build the training files.
3. Start fine-tuning from the base checkpoint with a small learning rate, monitor sample outputs, and stop when quality plateaus.
4. Export the best checkpoint to ONNX and test it with `piper`.

Follow the current Piper training docs for exact commands.

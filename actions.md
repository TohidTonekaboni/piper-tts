# Actions: from downloaded YouTube data to a fine-tuned Piper voice

## Where you are now

`data_gathering.py` finished. State of `dataset/`:

| Item | Value |
|---|---|
| Videos downloaded | 104 files in `dataset/voices` (22,050 Hz, mono, 16-bit: correct) |
| Raw audio | speaker1 ≈ **39.7 h** (85 videos), speaker2 ≈ **10.4 h** (22 videos) |
| Duplicates in `source.json` | 3 URLs listed twice (`g7P-FbLMQ8o`, `lePCAjHXbGM`, `zgvxbiydhHc`) |
| Broken entry | 1 entry with an empty `url` and no files |
| `dataset/wavs`, `metadata.csv` | not built yet |

**Is it ready for fine-tuning? No, not yet.** `make_dataset.py` is only the first step and you should run it. But
the data has problems it does not fix (below). Piper quality depends mostly on transcript accuracy and on there
being exactly one voice per clip.

**Decision: train on `speaker1` only** (the speaker with more audio). `speaker2` is ignored everywhere below. You
can leave its files in place; `make_dataset.py` still cuts its clips, but every later step filters on
`--speaker speaker1`, so they are never used. Do not spend Whisper time on `speaker2`.

## Problems to deal with

1. **Several voices per video.** The first text file is a podcast ("به پادکست صبح جمعه خوش اومدین"). A
   `speaker` label is applied per video, so a guest's voice will be labelled as `speaker1`. Mixed voices in one
   speaker degrade a TTS model badly. No script checks this: you must spot-check it (step 3) or use diarization.
2. **YouTube auto-captions are weak transcripts.** They have no punctuation (Piper uses punctuation for pauses and
   intonation), cut sentences mid-phrase (e.g. a line ending in a dangling "خیلی"), and have approximate timing.
   Fix: compare each clip to a Whisper transcription and drop mismatches (step 4).
3. **Music, laughter, overlapping speech, intro jingles** are not filtered. Spot-check and Whisper-filter.
4. **Digits are dropped, not converted.** Clips containing numbers are discarded (fine, just lossy).
5. **Two speakers vs. the base model.** The existing `fa_IR-amir-medium` is single-speaker. Fine-tuning from it is
   simplest with one speaker. You have 39.7 h for speaker1 alone, far more than needed, so **train on speaker1
   only** and ignore speaker2 (or train a second model later).
6. **Loudness** differs between videos. `finalize_dataset.py` peak-normalizes every clip.
7. **Hardware.** Your Mac has no NVIDIA GPU and no `espeak-ng`. Do steps 1-5 on the Mac (the Whisper step is slow
   on CPU, so use `--limit` or a GPU box), and train (step 6+) on a CUDA GPU machine (Colab, RunPod, Lambda, etc.).

---

## Step 1: Clean `source.json`

Remove the entry with the empty `"url": ""` and the 3 duplicate URLs (keep one of each). Then confirm:

```bash
cd /Users/tohid/Desktop/piper-tts
source .venv/bin/activate
python - <<'EOF'
import json, collections
e = json.load(open("dataset/source.json"))
c = collections.Counter(x.get("voice_name") for x in e)
print("duplicates:", [k for k, v in c.items() if v > 1])
print("empty:", [i for i, x in enumerate(e) if not x.get("url")])
EOF
```

Optional: re-run `python data_gathering.py` if you add new URLs. It skips finished ones.

## Step 2: Build the clips (run `make_dataset.py`)

```bash
python make_dataset.py
```

Look at the last two lines: total clips, hours, and how many were skipped. Expect the audio to drop a lot
versus the 50 h raw. If the skipped counts are very high, tell me and we'll tune `MIN_SEC` / `MAX_GAP`.

## Step 3: Listen to a random sample (mandatory)

```bash
grep '|speaker1|' dataset/metadata.csv | sort -R | head -40 > /tmp/sample.csv
while IFS='|' read -r id spk text; do
  echo "[$spk] $id: $text"; afplay "dataset/wavs/$id.wav"
done < /tmp/sample.csv
```

Check, for each clip: same single voice? text matches what you hear? no music or other people? If more than about
10% are bad, the video list itself needs pruning: note which video ids (the part of `clip_id` before `_NNNN`)
contain guests or music and delete those lines from `dataset/source.json` and their files in
`dataset/voices/` and `dataset/texts/`, then repeat Step 2.

## Step 4: Verify transcripts with Whisper

Trial first (200 clips, a few minutes on CPU) to see how many survive:

```bash
python verify_dataset.py --speaker speaker1 --limit 200 --model medium
python - <<'EOF'
import statistics
s = [float(l.split("|")[-1]) for l in open("dataset/verified.csv", encoding="utf-8")]
print(len(s), "clips; median sim", statistics.median(s), "; share >= 85:", sum(x >= 85 for x in s) / len(s))
EOF
```

If the share >= 85 is reasonable (say above 50%), delete `dataset/verified.csv` and run the full pass. This is
**slow on a Mac CPU (many hours)**, so run it on the GPU machine instead (copy `dataset/wavs` and
`dataset/metadata.csv`, then `--device cuda --model large-v3`). The script is resumable: re-run it and it continues.

```bash
python verify_dataset.py --speaker speaker1 --model large-v3 --device cuda
```

## Step 5: Build the final training folder

```bash
python finalize_dataset.py --speaker speaker1 --min-sim 85 --out dataset/train
```

- Add `--use-whisper-text` to train on Whisper's text instead of the subtitles. It includes punctuation, which
  helps prosody, but is occasionally wrong. Spot-check a few lines of `dataset/train/metadata.csv` and pick one.
- Result: `dataset/train/wavs/`, `metadata.csv` (`id.wav|text`, no speaker column), `val.csv` (1% held out).
- You only need 3-10 h of good audio. If more than 10 h survive, you can keep a random 10 h to speed up training.

Check the size:

```bash
wc -l dataset/train/metadata.csv dataset/train/val.csv
```

## Step 6: Set up training (on the GPU machine)

Copy `dataset/train/` to the GPU machine, then:

```bash
sudo apt-get install -y espeak-ng build-essential cmake ninja-build python3-dev
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[train]'
./build_monotonic_align.sh        # see TRAINING.md in the repo if this step has changed
python3 setup.py build_ext --inplace
```

Training docs for your installed version are authoritative: `piper1-gpl/docs/TRAINING.md`. Flags below follow that doc as I
remember it; if one is rejected, run `python3 -m piper.train fit --help`.

## Step 7: Get the base checkpoint

You need a `.ckpt` (training checkpoint), not the `.onnx` in `model/`. Download Farsi medium from the Piper
checkpoints repo (find the exact path with `hf repo-files`/the web page first; I could not verify it from here):

```bash
hf download rhasspy/piper-checkpoints --include "fa/fa_IR/amir/medium/*" --local-dir base
find base -name "*.ckpt"
```

If there is no Farsi checkpoint, use any **medium, single-speaker** one (e.g. English) as the base. Pronunciation
comes from espeak `fa` phonemes, so cross-language fine-tuning still works, but needs more steps.

## Step 8: Train

```bash
python3 -m piper.train fit \
  --data.voice_name fa-speaker1 \
  --data.csv_path /path/to/train/metadata.csv \
  --data.audio_dir /path/to/train/wavs/ \
  --data.espeak_voice fa \
  --data.cache_dir /path/to/cache \
  --data.config_path /path/to/out/fa-speaker1.onnx.json \
  --data.batch_size 32 \
  --model.sample_rate 22050 \
  --ckpt_path /path/to/base/<the-file>.ckpt \
  --trainer.max_epochs 3000
```

- The first run phonemizes everything and builds the cache. Phoneme errors or skipped lines are printed; check them.
- Watch the loss curves with TensorBoard (`tensorboard --logdir lightning_logs`) and listen to the generated
  samples. Stop when quality stops improving. With 3-10 h and a good base this is usually a few hundred to a
  couple of thousand epochs. Keep several checkpoints and compare by ear.
- If you run out of GPU memory, lower `--data.batch_size`.

## Step 9: Export to ONNX and test

```bash
python3 -m piper.train.export_onnx \
  --checkpoint /path/to/lightning_logs/version_X/checkpoints/<best>.ckpt \
  --output-file fa-speaker1.onnx
cp /path/to/out/fa-speaker1.onnx.json fa-speaker1.onnx.json
echo "سلام، این یک آزمایش است." | piper -m fa-speaker1.onnx -f test.wav
```

Test with sentences from `val.csv`, plus numbers written as words, loanwords and long sentences. If specific
words are mispronounced, that is espeak's missing short vowels (see `dataset/dataset_instructions.md` section 4.5):
add diacritics to those words in the training text, or add more clips containing them.

---

## Checklist

- [ ] Step 1: `source.json` has no empty URL or duplicates
- [ ] Step 2: `make_dataset.py` run, stats look sane
- [ ] Step 3: 40 random clips listened to, single voice, text matches
- [ ] Step 4: `verified.csv` complete (on the GPU box)
- [ ] Step 5: `dataset/train/` has at least 3 h
- [ ] Steps 6-8: trained from a `.ckpt` on GPU
- [ ] Step 9: exported and tested

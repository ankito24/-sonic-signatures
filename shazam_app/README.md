# Sonic Signatures — Audio Fingerprinting App (Q3B)

A from-scratch Shazam-style audio identifier built with `numpy`/`scipy` (no
external audio-fingerprinting library — the spectrogram, peak-picking,
hashing, and matching are all implemented by hand in `fingerprint.py`).

## What's in this folder

```
app.py              Streamlit app (single-clip mode + batch mode)
fingerprint.py       Core fingerprinting engine (spectrogram, peaks, hashing, matching)
songs/                The provided song library (mp3s) — ships with the app
fingerprints.pkl     Pre-built fingerprint database for songs/ (so the app starts instantly)
requirements.txt     Python dependencies
packages.txt          System packages needed (ffmpeg, for MP3 decoding)
```

## Running locally

```bash
pip install -r requirements.txt
# also need ffmpeg on your system PATH:
#   macOS:   brew install ffmpeg
#   Ubuntu:  sudo apt install ffmpeg
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Push this whole folder to a **public GitHub repo** (all files, including
   `songs/` and `fingerprints.pkl` — do not gitignore them, the app needs them
   to work immediately after deploy).
2. Go to https://share.streamlit.io, sign in, click **"New app"**.
3. Point it at your repo, branch, and set the main file path to `app.py`.
4. Streamlit Cloud automatically reads `requirements.txt` (Python packages)
   and `packages.txt` (system packages — this installs `ffmpeg` via apt,
   which is required for MP3 decoding).
5. Deploy. First boot will load the prebuilt `fingerprints.pkl` instantly
   (it only rebuilds from `songs/` if `fingerprints.pkl` is missing).

## Notes on repo size

The `songs/` folder is ~90 MB and `fingerprints.pkl` is ~12 MB. This is
within GitHub's 100MB-per-file limit and well within Streamlit Cloud's
free-tier app size limits. If your actual assignment song library is much
larger, consider Git LFS, or rebuild the database from your own copy of the
songs at first boot (the app supports this automatically — just omit
`fingerprints.pkl` and make sure `songs/` contains your library; first boot
will then take roughly 5-8 seconds per song to index).

## If you swap in a different/larger song library

Just replace the contents of `songs/` with your own files (any names, mp3 or
wav) and delete `fingerprints.pkl` — the app will rebuild the database
automatically on first run and cache it as `fingerprints.pkl` so subsequent
restarts are instant. Remember: the filename (without extension) is what the
identifier reports as the song name, so keep filenames matching the
assignment's required labels and don't rename them.

"""
app.py — Streamlit audio fingerprinting app (Q3B: 'Zapp tain America').

Two modes:
  1. Single-clip mode: upload one query clip, see the recognised song plus
     the intermediate steps (spectrogram, constellation, offset histogram).
  2. Batch mode: upload many query clips, get a results.csv with
     filename, prediction columns.
"""
import os
import io
import csv
import glob
import tempfile
import subprocess
import zipfile

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fingerprint import (
    load_wav, fingerprint_audio, FingerprintDB, match_query, compute_spectrogram
)

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_BIN = "ffmpeg"  # fall back to system PATH / packages.txt-installed ffmpeg

st.set_page_config(page_title="Audio Fingerprint", layout="wide")

SONG_DIR = "songs"          # ships with the app: original mp3/wav library
DB_PATH = "fingerprints.pkl"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def decode_to_wav_array(file_bytes, suffix):
    """Decode any audio file (mp3/wav/etc) to mono 22050Hz float32 numpy array via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_in:
        f_in.write(file_bytes)
        in_path = f_in.name
    out_path = in_path + "_decoded.wav"
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", in_path, "-ac", "1", "-ar", "22050", out_path],
            check=True, capture_output=True,
        )
        sr, samples = load_wav(out_path)
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)
    return sr, samples


@st.cache_resource(show_spinner=False)
def build_or_load_database():
    """
    Build the fingerprint database from the song library (songs/ folder) once,
    and cache it. If a pre-built fingerprints.pkl ships with the app, load
    that instead (much faster startup).
    """
    if os.path.exists(DB_PATH):
        return FingerprintDB.load(DB_PATH)

    db = FingerprintDB()
    audio_files = sorted(
        glob.glob(os.path.join(SONG_DIR, "*.mp3"))
        + glob.glob(os.path.join(SONG_DIR, "*.wav"))
    )
    progress = st.progress(0.0, text="Indexing song database...")
    for i, path in enumerate(audio_files):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as fh:
            sr, samples = decode_to_wav_array(fh.read(), os.path.splitext(path)[1])
        result = fingerprint_audio(samples, sr)
        db.add_song(name, result["hashes"], meta={"sr": sr, "duration": len(samples) / sr})
        progress.progress((i + 1) / max(len(audio_files), 1),
                           text=f"Indexed {name} ({i+1}/{len(audio_files)})")
    progress.empty()
    db.save(DB_PATH)
    return db


def plot_spectrogram(samples, sr, peaks=None):
    f, t, Sxx_db = compute_spectrogram(samples, sr)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.pcolormesh(t, f, Sxx_db, shading="auto", cmap="magma", vmin=-80, vmax=-10)
    ax.set_ylim(0, 5000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.tight_layout()
    return fig, (f, t, Sxx_db)


def plot_constellation(f, t, Sxx_db, peaks):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.pcolormesh(t, f, Sxx_db, shading="auto", cmap="gray_r", vmin=-80, vmax=-10)
    if peaks:
        peak_t = [t[p[1]] for p in peaks]
        peak_f = [f[p[0]] for p in peaks]
        ax.scatter(peak_t, peak_f, s=14, facecolors="none", edgecolors="#e53e3e", linewidths=1.0)
    ax.set_ylim(0, 5000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Constellation map ({len(peaks)} peaks)")
    fig.tight_layout()
    return fig


def plot_offset_histogram(hist, best_song):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if best_song and best_song in hist:
        offsets = np.array(list(hist[best_song].keys()))
        counts = np.array(list(hist[best_song].values()))
        peak_offset = offsets[np.argmax(counts)]
        window = 250
        mask = np.abs(offsets - peak_offset) <= window
        ax.bar(offsets[mask], counts[mask], width=2.0, color="#2b6cb0")
        ax.set_title(f"Offset histogram for best match: {best_song}")
    else:
        ax.set_title("No match found")
    ax.set_xlabel("Offset (song time bin - query time bin)")
    ax.set_ylabel("# matching hashes")
    fig.tight_layout()
    return fig


def identify_clip(file_bytes, suffix):
    sr, samples = decode_to_wav_array(file_bytes, suffix)
    result = fingerprint_audio(samples, sr)
    matches, hist = match_query(result["hashes"], st.session_state.db, top_k=5)
    return sr, samples, result, matches, hist


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

st.title("🎵 Audio Fingerprint")
st.caption(
    "A from-scratch Shazam-style identifier: spectrogram → constellation map → "
    "paired-peak hashes → offset-histogram matching."
)

if "db" not in st.session_state:
    with st.spinner("Loading / building song database..."):
        st.session_state.db = build_or_load_database()

db = st.session_state.db
st.success(f"Database ready: **{len(db.songs)}** songs indexed, **{len(db.db):,}** unique hashes.")
with st.expander("Show indexed songs"):
    st.write(sorted(db.songs.keys()))

tab1, tab2 = st.tabs(["🎯 Single-clip mode", "📦 Batch mode"])

# ------------------------------- Tab 1: single clip ----------------------
with tab1:
    st.subheader("Identify one query clip")
    query_file = st.file_uploader(
        "Upload a short audio clip (mp3 or wav)", type=["mp3", "wav", "m4a", "ogg"], key="single"
    )

    if query_file is not None:
        suffix = os.path.splitext(query_file.name)[1] or ".mp3"
        with st.spinner("Fingerprinting and matching..."):
            sr, samples, result, matches, hist = identify_clip(query_file.read(), suffix)

        st.audio(query_file)

        if matches and matches[0][1] >= 5:  # minimal confidence floor
            best_song, best_score, best_offset = matches[0]
            st.markdown(f"## ✅ Match found: **{best_song.replace('_', ' ')}**")
            second_score = matches[1][1] if len(matches) > 1 else 0
            st.caption(f"Score {best_score} vs. next-best {second_score} "
                       f"({best_score / max(second_score,1):.1f}x margin)")
        else:
            best_song = matches[0][0] if matches else None
            st.markdown("## ❌ No confident match found")

        st.markdown("#### Top candidates")
        cols = st.columns(min(len(matches), 5) or 1)
        for col, (name, score, offset) in zip(cols, matches):
            col.metric(name.replace("_", " "), score)

        st.markdown("#### Intermediate steps")
        c1, c2 = st.columns(2)
        f, t, Sxx_db = result["f"], result["t"], result["Sxx_db"]
        with c1:
            fig1, _ = plot_spectrogram(samples, sr)
            st.pyplot(fig1)
        with c2:
            fig2 = plot_constellation(f, t, Sxx_db, result["peaks"])
            st.pyplot(fig2)

        fig3 = plot_offset_histogram(hist, best_song)
        st.pyplot(fig3)

# ------------------------------- Tab 2: batch mode ------------------------
with tab2:
    st.subheader("Batch-identify many query clips")
    st.write(
        "Upload multiple clips (or a .zip of clips). Produces a `results.csv` "
        "with columns `filename, prediction`."
    )
    batch_files = st.file_uploader(
        "Upload query clips", type=["mp3", "wav", "m4a", "ogg", "zip"],
        accept_multiple_files=True, key="batch"
    )

    if batch_files and st.button("Run batch identification"):
        # collect (filename, bytes) pairs, expanding any zip files
        items = []
        for uf in batch_files:
            if uf.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(uf.read())) as zf:
                    for name in zf.namelist():
                        if name.lower().endswith((".mp3", ".wav", ".m4a", ".ogg")):
                            items.append((os.path.basename(name), zf.read(name)))
            else:
                items.append((uf.name, uf.read()))

        rows = []
        progress = st.progress(0.0)
        status = st.empty()
        for i, (fname, fbytes) in enumerate(items):
            status.text(f"Identifying {fname} ({i+1}/{len(items)})...")
            suffix = os.path.splitext(fname)[1] or ".mp3"
            try:
                sr, samples, result, matches, hist = identify_clip(fbytes, suffix)
                prediction = matches[0][0] if matches else ""
            except Exception as e:
                prediction = ""
            rows.append({"filename": fname, "prediction": prediction})
            progress.progress((i + 1) / len(items))
        status.empty()
        progress.empty()

        st.session_state.batch_rows = rows

    if "batch_rows" in st.session_state:
        rows = st.session_state.batch_rows
        st.markdown("#### Results")
        st.dataframe(rows, use_container_width=True)

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "⬇️ Download results.csv", data=buf.getvalue(),
            file_name="results.csv", mime="text/csv"
        )

st.divider()
st.caption(
    "Audio Fingerprint — spectrogram-based audio fingerprinting "
    "implemented from scratch with numpy/scipy (no external audio-fingerprint libraries)."
)
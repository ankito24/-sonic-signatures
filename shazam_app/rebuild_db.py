"""
rebuild_db.py
Standalone script to (re)build fingerprints.pkl from everything in songs/.
Run this any time you change the contents of the songs/ folder.

Usage:
    python3 rebuild_db.py
"""
import os
import sys
import glob
import time
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fingerprint import load_wav, fingerprint_audio, FingerprintDB

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_BIN = "ffmpeg"  # fall back to system PATH if imageio-ffmpeg isn't installed

SONG_DIR = "songs"
DB_PATH = "fingerprints.pkl"


def decode_to_wav(path):
    """Decode any audio file to mono 22050Hz wav via ffmpeg, return (sr, samples)."""
    suffix = os.path.splitext(path)[1]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        out_path = f_out.name
    try:
        subprocess.run(
            [FFMPEG_BIN, "-y", "-i", path, "-ac", "1", "-ar", "22050", out_path],
            check=True, capture_output=True,
        )
        sr, samples = load_wav(out_path)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    return sr, samples


def main():
    audio_files = sorted(
        glob.glob(os.path.join(SONG_DIR, "*.mp3"))
        + glob.glob(os.path.join(SONG_DIR, "*.wav"))
        + glob.glob(os.path.join(SONG_DIR, "*.m4a"))
    )
    if not audio_files:
        print(f"No audio files found in '{SONG_DIR}/'. Put your songs there first.")
        return

    print(f"Found {len(audio_files)} songs. Indexing...")
    db = FingerprintDB()
    t0 = time.time()

    for i, path in enumerate(audio_files):
        name = os.path.splitext(os.path.basename(path))[0]
        t_song = time.time()
        sr, samples = decode_to_wav(path)
        result = fingerprint_audio(samples, sr)
        db.add_song(name, result["hashes"], meta={"sr": sr, "duration": len(samples) / sr})
        print(f"  [{i+1}/{len(audio_files)}] {name}: "
              f"{len(samples)/sr:.1f}s, {len(result['peaks'])} peaks, "
              f"{len(result['hashes'])} hashes  ({time.time()-t_song:.1f}s)")

    db.save(DB_PATH)
    print(f"\nDone. {len(db.songs)} songs, {len(db.db):,} unique hashes.")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"Saved to {DB_PATH}")


if __name__ == "__main__":
    main()

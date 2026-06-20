"""
fingerprint.py
A from-scratch Shazam-style audio fingerprinting engine.

Pipeline:
  audio -> spectrogram (STFT, manual sliding-window DFT via scipy.signal.stft)
        -> constellation map (local-maxima peak picking)
        -> combinatorial hashing (pair nearby peaks -> (f1, f2, dt) hash)
        -> database of hash -> [(song_id, anchor_time), ...]
        -> matching by offset-histogram voting
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure
import hashlib
import pickle
import os

# ----------------------------- Audio I/O -----------------------------------

def load_wav(path):
    """Load a mono WAV file. Returns (sample_rate, float32 samples in [-1,1])."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    # normalize integer PCM to [-1, 1]
    max_val = np.iinfo(np.int16).max if data.dtype != np.float32 else 1.0
    if np.abs(data).max() > 1.5:  # looks like raw int16 range
        data = data / 32768.0
    return sr, data


# --------------------------- Spectrogram ------------------------------------

def compute_spectrogram(samples, sr, window_sec=0.025, hop_sec=0.010):
    """
    Compute the magnitude spectrogram using short-time DFTs (STFT).
    window_sec: length of each analysis window, in seconds
    hop_sec:    hop between successive windows, in seconds

    Returns: f (freq bins, Hz), t (time bins, sec), Sxx (magnitude in dB)
    """
    nperseg = int(round(window_sec * sr))
    noverlap = nperseg - int(round(hop_sec * sr))
    noverlap = max(noverlap, 0)

    f, t, Zxx = stft(samples, fs=sr, window='hann',
                      nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Zxx)
    # log-magnitude (dB), with floor to avoid log(0)
    Sxx_db = 20 * np.log10(mag + 1e-6)
    return f, t, Sxx_db


# --------------------------- Constellation map ------------------------------

def find_peaks_2d(Sxx_db, amp_min_db=None, neighborhood_size=20,
                   amp_percentile=75):
    """
    Find local maxima ('peaks') in a 2D spectrogram that stand out from
    their local neighborhood -- the constellation points.

    neighborhood_size: size (in bins) of the local max-filter footprint
                        (a larger value = peaks must be more locally dominant
                        and fewer peaks are kept)
    amp_min_db: absolute loudness floor (dB) below which peaks are discarded.
                If None (default), the floor is set adaptively as the
                `amp_percentile`-th percentile of the spectrogram's own
                dB values, since the absolute dB scale depends on how the
                STFT magnitude happens to be normalised.
    amp_percentile: used only when amp_min_db is None.

    Returns: list of (freq_bin_idx, time_bin_idx) peak coordinates
    """
    if amp_min_db is None:
        amp_min_db = np.percentile(Sxx_db, amp_percentile)

    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, neighborhood_size)

    local_max = maximum_filter(Sxx_db, footprint=neighborhood) == Sxx_db
    # discard quiet peaks (mostly background / silence)
    detected = local_max & (Sxx_db > amp_min_db)

    freq_idx, time_idx = np.where(detected)
    peaks = list(zip(freq_idx, time_idx))
    return peaks


# --------------------------- Hashing ----------------------------------------

def generate_hashes(peaks, fan_out=10, min_dt_bins=1, max_dt_bins=100):
    """
    Pair each peak (the 'anchor') with several nearby peaks that follow it
    in time ('targets'), within a fan-out window. Each pair becomes a hash:

        hash = H(f_anchor, f_target, delta_t)

    stored together with the anchor's own time (t_anchor), so that at
    matching time we can recover what time offset would line up song-time
    with query-time.

    Returns: list of (hash_str, t_anchor)
    """
    # sort by time so "nearby in time, forward only" pairing is easy
    peaks_sorted = sorted(peaks, key=lambda p: p[1])
    n = len(peaks_sorted)
    hashes = []

    for i in range(n):
        f1, t1 = peaks_sorted[i]
        # look forward at up to `fan_out` subsequent peaks
        for j in range(1, fan_out + 1):
            if i + j >= n:
                break
            f2, t2 = peaks_sorted[i + j]
            dt = t2 - t1
            if min_dt_bins <= dt <= max_dt_bins:
                h = hash_peak_pair(f1, f2, dt)
                hashes.append((h, t1))
    return hashes


def hash_peak_pair(f1, f2, dt):
    """Combine (f1, f2, dt) into a compact hash string."""
    raw = f"{int(f1)}|{int(f2)}|{int(dt)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def generate_single_peak_hashes(peaks):
    """
    Baseline / ablation: hash each peak ALONE (just its frequency bin),
    used to demonstrate why single-peak hashing performs far worse than
    paired hashing (too many collisions -> poor discriminative power).
    """
    hashes = []
    for f, t in peaks:
        h = hashlib.sha1(f"{int(f)}".encode()).hexdigest()[:20]
        hashes.append((h, t))
    return hashes


# --------------------------- Database ---------------------------------------

class FingerprintDB:
    """
    In-memory (picklable) hash database.
    Maps hash -> list of (song_name, anchor_time_bin)
    """
    def __init__(self):
        self.db = {}          # hash -> [(song_name, t_anchor), ...]
        self.songs = {}       # song_name -> metadata (duration, n_hashes, sr, hop_sec)

    def add_song(self, song_name, hashes, meta=None):
        for h, t in hashes:
            self.db.setdefault(h, []).append((song_name, t))
        self.songs[song_name] = meta or {}

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"db": self.db, "songs": self.songs}, fh)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj.db = data["db"]
        obj.songs = data["songs"]
        return obj


# --------------------------- Matching ---------------------------------------

def match_query(query_hashes, fp_db, top_k=5):
    """
    For each candidate song, build a histogram of (t_anchor_in_song -
    t_anchor_in_query) offsets across all matching hashes. A genuinely
    matching song produces one sharp peak in this histogram (everything
    lines up at one consistent offset); a wrong song gives only scattered,
    near-random offsets.

    Returns: sorted list of (song_name, score, offset) for the top_k
             candidates, plus the full per-song offset histograms (for
             visualisation).
    """
    # song_name -> {offset: count}
    offset_counts = {}

    for h, t_query in query_hashes:
        if h not in fp_db.db:
            continue
        for song_name, t_song in fp_db.db[h]:
            offset = t_song - t_query
            d = offset_counts.setdefault(song_name, {})
            d[offset] = d.get(offset, 0) + 1

    results = []
    for song_name, hist in offset_counts.items():
        best_offset, best_score = max(hist.items(), key=lambda kv: kv[1])
        results.append((song_name, best_score, best_offset))

    results.sort(key=lambda r: r[1], reverse=True)
    return results[:top_k], offset_counts


# --------------------------- High-level pipeline -----------------------------

def fingerprint_audio(samples, sr, window_sec=0.025, hop_sec=0.010,
                       amp_min_db=None, neighborhood_size=20,
                       fan_out=10, max_dt_bins=100, single_peak_mode=False):
    """
    Full pipeline from raw samples -> hashes, also returning the
    intermediate spectrogram + peaks for visualisation.
    """
    f, t, Sxx_db = compute_spectrogram(samples, sr, window_sec, hop_sec)
    peaks = find_peaks_2d(Sxx_db, amp_min_db=amp_min_db,
                           neighborhood_size=neighborhood_size)
    if single_peak_mode:
        hashes = generate_single_peak_hashes(peaks)
    else:
        hashes = generate_hashes(peaks, fan_out=fan_out, max_dt_bins=max_dt_bins)

    return {
        "f": f, "t": t, "Sxx_db": Sxx_db,
        "peaks": peaks, "hashes": hashes,
    }

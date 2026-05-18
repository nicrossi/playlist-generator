from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
import wave
import numpy as np
import essentia.standard as es
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_serializer


try:
    from essentia.standard import (
        TensorflowPredict2D,
        TensorflowPredictMusiCNN,
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


EXTRACTOR_VERSION = "0.1.0"
EXPECTED_SAMPLE_RATE = 44100
EXPECTED_SAMPWIDTH = 2  # 16-bit PCM = 2 bytes

MODELS_DIR = Path("./models")

MODEL_URLS = {
    "emomusic": "https://essentia.upf.edu/models/classification-heads/emomusic/emomusic-msd-musicnn-2.pb",
    "voice_instrumental": "https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-msd-musicnn-1.pb",
    "mood_acoustic": "https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-msd-musicnn-1.pb",
    "musicnn_embedding": "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb",
}

# MusiCNN models expect this embedding-extraction node
MUSICNN_EMBEDDING_OUTPUT = "model/dense/BiasAdd"
MUSICNN_PENULTIMATE_OUTPUT = "model/dense_1/BiasAdd"
MUSICNN_INPUT = "model/Placeholder"

SPOTIFY_KEY_INDEX_TO_STR = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SPOTIFY_KEY_STR_TO_INDEX = {k: i for i, k in enumerate(SPOTIFY_KEY_INDEX_TO_STR)}
PITCH_CLASSES = frozenset(SPOTIFY_KEY_INDEX_TO_STR)
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


class ExtractionError(Exception):
    """Raised when audio cannot be processed."""


def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


# Output schema
class AudioFeatures(BaseModel):
    """Extracted audio features.

    `liveness` and `time_signature` are present in Spotify's schema but not
    inferred from audio here. They take fixed defaults so CSV output stays
    schema-compatible:

      - liveness = 0.0: no Essentia equivalent for audience-presence detection.
        Most music in any popular-music dataset is studio-recorded, so 0 is the
        correct call for the majority case. Downstream query rewriters should
        treat liveness filters as a no-op for fallback-path tracks.
      - time_signature = 4: meter estimation via BeatTrackerMultiFeature is
        possible but rarely useful — odd meters appear mainly in classical and
        prog/jazz, a small fraction of popular music, and users almost never
        filter on this field.
    """

    source_path: str
    source_sha256: str
    extractor_version: str
    extracted_at: datetime

    duration_ms: int
    tempo: float
    tempo_confidence: float
    key: int = Field(ge=0, le=11)
    mode: Literal[0, 1]
    key_confidence: float
    loudness: float

    energy: float = Field(ge=0.0, le=1.0)
    danceability: float = Field(ge=0.0, le=1.0)
    liveness: float = Field(default=0.0, ge=0.0, le=1.0)
    time_signature: int = Field(default=4, ge=3, le=7)

    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    arousal: float | None = Field(default=None, ge=0.0, le=1.0)
    acousticness: float | None = Field(default=None, ge=0.0, le=1.0)
    instrumentalness: float | None = Field(default=None, ge=0.0, le=1.0)
    speechiness: float | None = Field(default=None, ge=0.0, le=1.0)

    embedding: list[float] | None = None

    @field_serializer(
        "tempo", "tempo_confidence", "key_confidence",
        "loudness", "energy", "danceability", "liveness", "valence", "arousal",
        "acousticness", "instrumentalness", "speechiness",
        when_used="json",
    )
    def _round_float(self, v: float | None) -> float | None:
        return round(v, 4) if v is not None else None

    @field_serializer("embedding", when_used="json")
    def _round_embedding(self, v: list[float] | None) -> list[float] | None:
        return [round(x, 4) for x in v] if v is not None else None


# Audio loading + validation
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate(path: Path) -> np.ndarray:
    """Validate WAV header against contract, return mono float32 samples."""
    if not path.exists():
        raise ExtractionError(f"file not found: {path}")

    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            sw = wf.getsampwidth()
    except wave.Error as e:
        raise ExtractionError(f"invalid WAV header for {path}: {e}")

    if sr != EXPECTED_SAMPLE_RATE:
        print(
            f"warning: {path}: sample rate {sr} != {EXPECTED_SAMPLE_RATE}; "
            f"MonoLoader will resample",
            file=sys.stderr,
        )
    if sw != EXPECTED_SAMPWIDTH:
        raise ExtractionError(
            f"{path}: sample width {sw*8}-bit != expected {EXPECTED_SAMPWIDTH*8}-bit PCM"
        )

    return es.MonoLoader(filename=str(path), sampleRate=EXPECTED_SAMPLE_RATE)()


# --- Tier 1: Digital Signal Processing (DSP) ---
def extract_dsp_features(audio: np.ndarray) -> dict:
    """Objective DSP features: duration, tempo, key, loudness."""
    out: dict = {"duration_ms": int(round(1000 * len(audio) / EXPECTED_SAMPLE_RATE))}

    try:
        bpm, _ticks, tempo_conf, _est, _intervals = es.RhythmExtractor2013(method="multifeature")(audio)
        out["tempo"] = float(bpm)
        # RhythmExtractor2013 confidence is roughly [0, 5.32]; clamp to [0, 1]
        out["tempo_confidence"] = _clamp01(tempo_conf / 5.32)
    except Exception as e:
        print(f"warning: tempo extraction failed: {e}", file=sys.stderr)
        out["tempo"], out["tempo_confidence"] = 0.0, 0.0

    try:
        # "edma" profile trained on EDM/pop, generally outperforms default "temperley" on contemporary music.
        key, scale, strength = es.KeyExtractor(profileType="edma")(audio)
        if key not in PITCH_CLASSES:
            # Essentia returns e.g. "Db"; normalize to sharp notation
            key = FLAT_TO_SHARP.get(key, key)
        out["key"] = SPOTIFY_KEY_STR_TO_INDEX[key]
        out["mode"] = 1 if scale == "major" else 0
        out["key_confidence"] = float(strength)
    except Exception as e:
        print(f"warning: key extraction failed: {e}", file=sys.stderr)
        out["key"], out["mode"], out["key_confidence"] = 0, 1, 0.0

    try:
        # LoudnessEBUR128 requires stereo — duplicate mono channel.
        stereo = np.stack([audio, audio], axis=1).astype(np.float32)
        _momentary, _short, integrated, _range = es.LoudnessEBUR128(sampleRate=EXPECTED_SAMPLE_RATE)(stereo)
        out["loudness"] = float(integrated)
    except Exception as e:
        print(f"warning: loudness extraction failed: {e}", file=sys.stderr)
        out["loudness"] = -60.0

    return out


# --- Tier 2: Composite ---
def extract_composite_features(audio: np.ndarray, loudness: float) -> dict:
    """Composite features: energy (heuristic blend), danceability."""
    out: dict = {}

    try:
        dance_raw, _dfa = es.Danceability()(audio)
        out["danceability"] = _clamp01(dance_raw / 3.0)
    except Exception as e:
        print(f"warning: danceability extraction failed: {e}", file=sys.stderr)
        out["danceability"] = 0.0

    # Energy: heuristic blend (loudness + spectral brightness + flux).
    try:
        # Tighter denominator avoids saturation; modern pop sits in [-15, -5] LUFS.
        loudness_norm = _clamp01((loudness + 30.0) / 30.0)

        spectrum = es.Spectrum()
        window = es.Windowing(type="hann")
        centroid = es.Centroid(range=EXPECTED_SAMPLE_RATE / 2)
        flux = es.Flux()
        centroid_vals: list[float] = []
        flux_vals: list[float] = []
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            spec = spectrum(window(frame))
            centroid_vals.append(centroid(spec))
            flux_vals.append(flux(frame))
        centroid_mean = float(np.mean(centroid_vals)) if centroid_vals else 0.0
        centroid_norm = _clamp01(centroid_mean / 4000.0)
        flux_mean = float(np.mean(flux_vals)) if flux_vals else 0.0
        flux_norm = float(np.tanh(flux_mean * 5.0))

        out["energy"] = _clamp01(0.5 * loudness_norm + 0.3 * centroid_norm + 0.2 * flux_norm)
    except Exception as e:
        print(f"warning: energy extraction failed: {e}", file=sys.stderr)
        out["energy"] = 0.0

    return out


# --- Tier 3: ML inference ---
def _model_path(name: str) -> Path:
    return MODELS_DIR / Path(MODEL_URLS[name]).name


def ensure_models_downloaded() -> None:
    """Download Essentia model files to ./models/ if missing."""
    if not TF_AVAILABLE:
        return
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = _model_path(name)
        if dest.exists():
            continue
        print(f"downloading {name} model from {url} ...", file=sys.stderr)
        last_err: Exception | None = None
        for attempt in (1, 2):
            try:
                urllib.request.urlretrieve(url, dest)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt == 1:
                    time.sleep(2.0)
        if last_err is not None:
            if dest.exists():
                dest.unlink()
            raise ExtractionError(
                f"failed to download model '{name}' from {url}: {last_err}. "
                f"Download manually to {dest}."
            )


def _musicnn_head_predict(audio: np.ndarray, head_model: str, head_output: str) -> np.ndarray:
    """Run MusiCNN embedding → TensorflowPredict2D head, return mean-pooled prediction."""
    emb = TensorflowPredictMusiCNN(
        graphFilename=str(_model_path("musicnn_embedding")),
        output=MUSICNN_EMBEDDING_OUTPUT,
    )(audio)
    pred = TensorflowPredict2D(
        graphFilename=str(_model_path(head_model)),
        output=head_output,
    )(emb)
    return np.mean(pred, axis=0)


def extract_ml_features(audio: np.ndarray, key_confidence: float) -> dict:
    """ML inference: valence, arousal, acousticness, instrumentalness, speechiness, embedding."""
    out: dict = {
        "valence": None,
        "arousal": None,
        "acousticness": None,
        "instrumentalness": None,
        "speechiness": None,
        "embedding": None,
    }
    if not TF_AVAILABLE:
        return out

    # MusiCNN penultimate-layer embedding.
    try:
        embeddings = TensorflowPredictMusiCNN(
            graphFilename=str(_model_path("musicnn_embedding")),
            output=MUSICNN_PENULTIMATE_OUTPUT,
        )(audio)
        # embeddings is (n_frames, 200). Mean-pool over time.
        emb_mean = np.mean(embeddings, axis=0)
        out["embedding"] = [float(x) for x in emb_mean.tolist()]
    except Exception as e:
        print(f"warning: embedding extraction failed: {e}", file=sys.stderr)

    # Emomusic — output is [valence, arousal] in [1, 9], normalize to [0, 1] via (x - 1) / 8.
    # Note: index order verified empirically — strong negative correlation with Spotify
    # valence under the previously-assumed [valence, arousal] order indicated a swap.
    try:
        emo_mean = _musicnn_head_predict(audio, "emomusic", "model/Identity")
        arousal_raw, valence_raw = float(emo_mean[0]), float(emo_mean[1])
        out["valence"] = _clamp01((valence_raw - 1.0) / 8.0)
        out["arousal"] = _clamp01((arousal_raw - 1.0) / 8.0)
    except Exception as e:
        print(f"warning: emomusic inference failed: {e}", file=sys.stderr)

    # Voice/instrumental — output is [instrumental_prob, voice_prob] (classes alphabetical).
    voice_prob: float | None = None
    try:
        vi_mean = _musicnn_head_predict(audio, "voice_instrumental", "model/Softmax")
        out["instrumentalness"] = _clamp01(vi_mean[0])
        voice_prob = float(vi_mean[1])
    except Exception as e:
        print(f"warning: voice_instrumental inference failed: {e}", file=sys.stderr)

    # Mood acoustic — output [acoustic_prob, non_acoustic_prob]
    try:
        ac_mean = _musicnn_head_predict(audio, "mood_acoustic", "model/Softmax")
        out["acousticness"] = _clamp01(ac_mean[0])
    except Exception as e:
        print(f"warning: mood_acoustic inference failed: {e}", file=sys.stderr)

    # Speechiness: approximation (Essentia has no direct equivalent).
    # speech / rap / screams produce flatter spectra than sung melody — combine voice probability
    # with mean spectral flatness as a noise-likeness proxy.
    if voice_prob is not None:
        try:
            spectrum = es.Spectrum()
            window = es.Windowing(type="hann")
            flatness = es.Flatness()
            flat_vals = [flatness(spectrum(window(frame)))
                         for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024)]
            flat_mean = float(np.mean(flat_vals)) if flat_vals else 0.0
            # Scale flatness (~0-0.3 typical) into [0,1] before blending with voice probability.
            flat_norm = _clamp01(flat_mean * 3.0)
            out["speechiness"] = _clamp01(voice_prob * flat_norm)
        except Exception as e:
            print(f"warning: speechiness flatness failed: {e}", file=sys.stderr)
            out["speechiness"] = _clamp01(voice_prob * (1.0 - _clamp01(key_confidence)))

    return out


# --- Pipeline orchestrator ---
def extract_all_features(path: Path) -> AudioFeatures:
    audio = load_and_validate(path)

    dsp = extract_dsp_features(audio)
    composite = extract_composite_features(audio, dsp["loudness"])
    ml = extract_ml_features(audio, dsp["key_confidence"])

    return AudioFeatures(
        source_path=str(path),
        source_sha256=_sha256_file(path),
        extractor_version=EXTRACTOR_VERSION,
        extracted_at=datetime.now(timezone.utc),
        **dsp,
        **composite,
        **ml,
    )


def _extract_with_progress(path: Path, i: int, total: int) -> AudioFeatures | None:
    """Run extraction with stderr progress + timing. Returns None on failure."""
    print(f"[{i}/{total}] Extracting {path}...", file=sys.stderr, end=" ")
    t0 = time.time()
    try:
        feats = extract_all_features(path)
        print(f"done ({time.time() - t0:.1f}s)", file=sys.stderr)
        return feats
    except ExtractionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
    except Exception as e:
        print(f"FAILED: unexpected error: {e}", file=sys.stderr)
    return None


# --- Validation harness ---
def _tempo_pass(extracted: float, expected: float, tol: float) -> tuple[bool, bool]:
    """Return (pass, soft_pass). soft_pass = octave error tolerated."""
    if abs(extracted - expected) <= tol:
        return True, False
    if abs(extracted - 2 * expected) <= tol or abs(extracted - expected / 2) <= tol:
        return True, True
    return False, False


def _key_pass(extracted: int, expected: int) -> bool:
    if extracted == expected:
        return True
    # Perfect fifth confusion — ±7 semitones mod 12.
    return (extracted - expected) % 12 in (5, 7)


# Per-feature tolerance for Spotify ground-truth validation. Keys match AudioFeatures
# field names (post-conversion from Spotify columns). Loose tolerances on heuristic
# features — correlation across the set is the real signal.
SPOTIFY_TOLERANCES: dict[str, float] = {
    "loudness": 3.0,          # dB FS vs LUFS — different metrics, ±3 is loose
    "duration": 2000.0,       # ms — encoder/trim drift on long tracks
    "energy": 0.25,
    "danceability": 0.25,
    "valence": 0.25,
    "acousticness": 0.25,
    "instrumentalness": 0.30,
    "speechiness": 0.15,
}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r over paired non-NaN values. Returns None if <2 valid pairs or zero variance."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    a, b = (np.array(v, dtype=float) for v in zip(*pairs))
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _cell(sym: str, val, width: int, fmt: str = "") -> str:
    return f"{sym} {val:<{width}{fmt}}"


def run_validation(csv_path: Path) -> int:
    """Validate extracted features against a Spotify audio_features CSV.

    Expected columns (subset): file_path, energy, tempo, danceability, loudness,
    valence, speechiness, instrumentalness, acousticness, mode, key, duration_ms.
    Missing columns are skipped per row. `liveness` and `time_signature` are not
    in AudioFeatures and are reported as schema gaps.
    """
    if not csv_path.exists():
        print(f"validation CSV not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("file_path")]

    if not rows:
        print("validation CSV empty or missing file_path column", file=sys.stderr)
        return 1

    if TF_AVAILABLE:
        try:
            ensure_models_downloaded()
        except ExtractionError as e:
            print(f"model setup failed: {e}", file=sys.stderr)
            return 1

    total_checks = 0
    passed_checks = 0
    soft_passes = 0
    report_lines: list[str] = []
    notes: list[str] = []

    # Collect per-feature paired values for correlation analysis.
    paired: dict[str, tuple[list, list]] = {
        k: ([], []) for k in (
            "tempo", "loudness", "duration", "energy", "danceability",
            "valence", "acousticness", "instrumentalness", "speechiness",
        )
    }

    columns = ["File", "Tempo", "Key", "Mode", "Loud", "Dur(ms)", "Energy",
               "Dance", "Valence", "Acoust", "Instr", "Speech"]
    widths = [28, 11, 9, 8, 9, 10, 9, 9, 9, 9, 9, 9]
    header = " ".join(f"{c:<{w}}" for c, w in zip(columns, widths))
    separator = "─" * len(header)
    report_lines += [header, separator]

    for i, row in enumerate(rows, 1):
        file_path = Path(row["file_path"])
        feats = _extract_with_progress(file_path, i, len(rows))
        if feats is None:
            report_lines.append(f"{file_path.name:<28} ✗ extraction failed")
            continue

        cells: list[str] = []

        # Tempo (BPM, octave-tolerant)
        if row.get("tempo"):
            total_checks += 1
            exp = float(row["tempo"])
            ok, soft = _tempo_pass(feats.tempo, exp, 5.0)
            if ok:
                passed_checks += 1
                if soft:
                    soft_passes += 1
                    notes.append(
                        f"  {file_path.name}: expected tempo {exp:.1f}, "
                        f"got {feats.tempo:.1f} — octave error, soft pass"
                    )
                sym = "⚠" if soft else "✓"
            else:
                sym = "✗"
            cells.append(_cell(sym, feats.tempo, 9, ".1f"))
            paired["tempo"][0].append(feats.tempo)
            paired["tempo"][1].append(exp)
        else:
            cells.append(_cell(" ", feats.tempo, 9, ".1f"))

        # Key (Spotify int 0-11)
        if row.get("key") not in (None, ""):
            total_checks += 1
            try:
                exp_key = int(row["key"])
                ok = _key_pass(feats.key, exp_key)
            except ValueError:
                ok = False
            if ok:
                passed_checks += 1
            sym = "✓" if ok else "✗"
            cells.append(_cell(sym, feats.key, 7))
        else:
            cells.append(_cell(" ", feats.key, 7))

        # Mode (Spotify int 0/1)
        if row.get("mode") not in (None, ""):
            total_checks += 1
            ok = feats.mode == int(row["mode"])
            if ok:
                passed_checks += 1
            sym = "✓" if ok else "✗"
            cells.append(_cell(sym, feats.mode, 6))
        else:
            cells.append(_cell(" ", feats.mode, 6))

        # Loudness (Spotify dB vs our LUFS — unit mismatch, ±3 dB)
        if row.get("loudness"):
            total_checks += 1
            exp = float(row["loudness"])
            ok = abs(feats.loudness - exp) <= SPOTIFY_TOLERANCES["loudness"]
            if ok:
                passed_checks += 1
            sym = "✓" if ok else "✗"
            cells.append(_cell(sym, feats.loudness, 7, ".2f"))
            paired["loudness"][0].append(feats.loudness)
            paired["loudness"][1].append(exp)
        else:
            cells.append(_cell(" ", feats.loudness, 7, ".2f"))

        # Duration (Spotify ms)
        if row.get("duration_ms"):
            total_checks += 1
            exp = float(row["duration_ms"])
            ok = abs(feats.duration_ms - exp) <= SPOTIFY_TOLERANCES["duration"]
            if ok:
                passed_checks += 1
            sym = "✓" if ok else "✗"
            cells.append(_cell(sym, feats.duration_ms, 8))
            paired["duration"][0].append(feats.duration_ms)
            paired["duration"][1].append(exp)
        else:
            cells.append(_cell(" ", feats.duration_ms, 8))

        # [0,1] features with per-feature tolerance.
        for col_name in ("energy", "danceability", "valence",
                         "acousticness", "instrumentalness", "speechiness"):
            our_val = getattr(feats, col_name)
            if row.get(col_name) and our_val is not None:
                total_checks += 1
                exp = float(row[col_name])
                ok = abs(our_val - exp) <= SPOTIFY_TOLERANCES[col_name]
                if ok:
                    passed_checks += 1
                sym = "✓" if ok else "✗"
                cells.append(_cell(sym, our_val, 7, ".2f"))
                paired[col_name][0].append(our_val)
                paired[col_name][1].append(exp)
            else:
                disp = f"{our_val:.2f}" if our_val is not None else "n/a"
                cells.append(_cell(" ", disp, 7))

        report_lines.append(f"{file_path.name[:28]:<28} " + " ".join(cells))

    report_lines.append(separator)
    report_lines.append(
        f"PASSED: {passed_checks}/{total_checks} checks across {len(rows)} files"
        + (f" ({soft_passes} soft)" if soft_passes else "")
    )

    print("\n".join(report_lines))

    if notes:
        print("\nNotes:")
        for n in notes:
            print(n)

    print("\nPearson correlation (our extraction vs Spotify reference)")
    print("─" * 60)
    print(f"{'Feature':<20} {'r':<10} {'n':<6}")
    print("─" * 60)
    for feat, (xs, ys) in paired.items():
        r = _pearson(xs, ys)
        r_str = f"{r:+.3f}" if r is not None else "n/a"
        print(f"{feat:<20} {r_str:<10} {len(xs):<6}")
    print("─" * 60)

    print("\nUnsupported features (fixed defaults — see AudioFeatures docstring):")
    print("  - liveness: defaulted to 0.0 (studio-recorded assumption)")
    print("  - time_signature: defaulted to 4 (4/4 assumption)")

    return 0 if total_checks == passed_checks else 1


# --- CLI ---
# Exact column order of the Kaggle Spotify audio_features CSV. Output rows can be
# concatenated directly with that dataset.
SPOTIFY_CSV_COLUMNS = [
    "file_path", "energy", "tempo", "danceability", "playlist_genre",
    "loudness", "liveness", "valence", "track_artist", "time_signature",
    "speechiness", "track_popularity", "track_href", "uri",
    "track_album_name", "playlist_name", "analysis_url", "track_id",
    "track_name", "track_album_release_date", "instrumentalness",
    "track_album_id", "mode", "key", "duration_ms", "acousticness",
    "id", "playlist_subgenre", "type", "playlist_id",
]


def _features_to_jsonable(feats: AudioFeatures) -> dict:
    return json.loads(feats.model_dump_json())


def _features_to_csv_row(feats: AudioFeatures) -> dict[str, str]:
    """Map AudioFeatures to a row matching the Spotify audio_features CSV schema.

    Spotify metadata columns we can't infer from audio (track_artist, playlist_*,
    Spotify ids/URLs) are emitted as empty strings. `liveness` and `time_signature`
    are unsupported by this extractor (see AudioFeatures docstring).
    """
    j = json.loads(feats.model_dump_json())  # picks up rounded floats
    def _maybe(key: str) -> str:
        v = j.get(key)
        return "" if v is None else str(v)
    row = {col: "" for col in SPOTIFY_CSV_COLUMNS}
    row.update({
        "file_path": feats.source_path,
        "energy": str(j["energy"]),
        "tempo": str(j["tempo"]),
        "danceability": str(j["danceability"]),
        "loudness": str(j["loudness"]),
        "liveness": str(j["liveness"]),
        "valence": _maybe("valence"),
        "time_signature": str(feats.time_signature),
        "speechiness": _maybe("speechiness"),
        "instrumentalness": _maybe("instrumentalness"),
        "acousticness": _maybe("acousticness"),
        "mode": str(feats.mode),
        "key": str(feats.key),
        "duration_ms": str(feats.duration_ms),
        "track_name": Path(feats.source_path).stem,
        "type": "audio_features",
    })
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="extract_features.py",
        description="Extract audio features from WAV files using Essentia.",
    )
    parser.add_argument("files", nargs="*", help="Audio file paths (WAV, 16-bit PCM; non-44.1kHz resampled, stereo auto-downmixed)")
    parser.add_argument("--output", help="Write results to file instead of stdout")
    parser.add_argument("--csv", action="store_true", help="Emit Spotify-compatible CSV instead of JSON")
    parser.add_argument("--no-header", action="store_true", help="Skip CSV header (use when appending to an existing CSV)")
    parser.add_argument("--validate", metavar="CSV", help="Validate features against expected values CSV")
    args = parser.parse_args(argv)

    if not TF_AVAILABLE:
        print(
            "warning: essentia-tensorflow not installed — Tier 3 (ML) features will be None",
            file=sys.stderr,
        )

    if args.validate:
        return run_validation(Path(args.validate))

    if not args.files:
        parser.print_help(sys.stderr)
        return 1

    if TF_AVAILABLE:
        try:
            ensure_models_downloaded()
        except ExtractionError as e:
            print(f"model setup failed: {e}", file=sys.stderr)
            return 1

    features: list[AudioFeatures] = []
    had_failure = False

    for i, raw_path in enumerate(args.files, 1):
        feats = _extract_with_progress(Path(raw_path), i, len(args.files))
        if feats is None:
            had_failure = True
            continue
        features.append(feats)

    if args.csv:
        target = open(args.output, "w", newline="") if args.output else sys.stdout
        try:
            writer = csv.DictWriter(target, fieldnames=SPOTIFY_CSV_COLUMNS)
            if not args.no_header:
                writer.writeheader()
            for f in features:
                writer.writerow(_features_to_csv_row(f))
        finally:
            if args.output:
                target.close()
    else:
        results = [_features_to_jsonable(f) for f in features]
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
        else:
            # Single file: emit object; multiple: emit array.
            print(json.dumps(results[0] if len(results) == 1 else results, indent=2))

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())

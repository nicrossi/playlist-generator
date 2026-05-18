# Audio Feature Extraction PoC

CLI tool that takes WAV files in, runs them through three layers of audio 
analysis, and produces a structured JSON feature vector out. It also has 
a second mode that validates those extracted features against a reference CSV.

## Essentia Adapter

Spotify deprecated the `audio_features` Web API endpoint in November 2024, so
for any track outside our pre-collected Kaggle dataset there is no longer an
upstream source for the per-track audio attributes (energy, danceability,
valence, tempo, key, etc.) that the recommender relies on. This tool fills that
gap by extracting equivalent features locally via [Essentia](https://essentia.upf.edu/):
DSP primitives cover the objective fields (tempo, key, loudness, duration),
Essentia's `Danceability` plus a heuristic blend cover the composite fields
(danceability, energy), and pre-trained MusiCNN-based classifier heads cover
the perceptual ones (valence, arousal, acousticness, instrumentalness,
speechiness). The output schema is Spotify-compatible — the `--csv` mode
produces rows that drop straight into the Kaggle CSV — so downstream code
treats Kaggle-sourced and Essentia-sourced tracks uniformly. Two Spotify
fields aren't inferred from audio and take fixed defaults: `liveness = 0.0`
(studio-recorded majority) and `time_signature = 4` (4/4 majority); query
rewriters should treat filters on those fields as no-ops for Essentia-path
tracks.

```
                                                                                                      ┌─→ JSON (default)
WAV file → [Validate format] → [Load audio] → [Tier 1: DSP] → [Tier 2: Composite] → [Tier 3: ML] → AudioFeatures
                                                                                                      └─→ Spotify-schema CSV (--csv)
```
## Install

```bash
pip install -r requirements.txt
```
## Usage

```bash
# Single file → pretty-printed JSON on stdout
python extract_features.py track.wav

# Multiple files → JSON array written to file
python extract_features.py *.wav --output results.json

# Spotify-compatible CSV (single row per track, exact column order)
python extract_features.py *.wav --csv --output new_tracks.csv

# Append to an existing Spotify CSV (skip header)
python extract_features.py track.wav --csv --no-header >> spotify_audio_features.csv

# Validate against expected values CSV
python extract_features.py --validate expected_values.csv
```

First run downloads ~4 TensorFlow models into `./models/` (~200 MB). Cached after.

## Stage 1 - Digital Signal Processing (DSP) Features

**Input:** Raw audio array `np.ndarray` (1-D, float32).

**Output:** Scalar values describing objectively measurable properties. Deterministic stuff.
They depend only on the math applied to the signal.

| Output | Type  | Range | Description |
| ------ |-------| ----- | ----------- |
| `duration_ms` | `int` | [0, ∞) | `round(1000 * len(audio) / 44100)`. Matches Spotify's `duration_ms`. |
| `tempo` | `float` | typically 60 - 200 | Estimated beats per minute. |
| `tempo_confidence` | `float` | [0, 1] | How sure the algorithm is. Low = ambiguous rhythm (e.g. free jazz, ambient). |
| `key` | `int` | 0–11 | Spotify pitch-class index (0 = C, 1 = C#, ..., 11 = B). |
| `mode` | `int` | 0 or 1 | Scale modality, Spotify convention (1 = major, 0 = minor). |
| `key_confidence` | `float` | [0, 1] | How sure the algorithm is about the key. |
| `loudness` | `float` | typically -30 to -5 | Integrated loudness (LUFS internally; named `loudness` to match Spotify, which uses peak/RMS dB — values are systematically offset by a few dB). |

## Stage 2 - Composite Features

**Input:** The audio array + `loudness` from Stage 1.

**Output:** Two scalar values that don't exist as a single algorithm, they're computed by combining or transforming other measurements.

**Danceability** uses Essentia's built-in `Danceability()` algorithm directly.
Returns a value in [0, 3], normalized to [0, 1].

**Energy** is 50% normalized loudness, 30% RMS energy, 20% spectral flux:
```python
energy = 0.5 * loudness_norm + 0.3 * rms_norm + 0.2 * flux_norm
```
The weights can be tuned against Spotify's energy.

| Output | Type | Range | Description                                                                 |
| ------ | ---- | ----- |-----------------------------------------------------------------------------|
| `danceability` | `float` | [0, 1] | Rhythm stability + beat strength + tempo (blended by Essentia's algorithm). |
| `energy` | `float` | [0, 1] | 50% loudness + 30% RMS + 20% spectral flux.                                 |


## Stage 3 - ML Features

We use Essentia's Tensorflow models.
1. Run audio through MusiCNN (a pre-trained music CNN) to get embeddings, a high-dimensional representation of the audio's "musical content."
2. Feed those embeddings into a classifier head trained on a specific task (emomusic, voice/instrumental, mood/acoustic).
3. The classifier outputs per-frame predictions; the agent mean-pools across time to get a single value for the whole track.

Specifically:

- Valence + arousal come from the `emomusic` model — a regression head that outputs both values in `[1, 9]`, normalized to `[0, 1]`.
- Instrumentalness is the `instrumental_prob` from a binary classifier.
- Acousticness is the `acoustic_prob` from another binary classifier.
- Speechiness is the agent's approximation: `voice_prob * (1 - key_confidence)`. Spoken word = high voice presence + weak tonal structure. Cute heuristic — we'll see if it holds up.
- Embedding is the 200-dimensional penultimate layer of MusiCNN itself, mean-pooled — this is your similarity-search vector for later.

**Input:** The audio array + `key_confidence` from Stage 1.
**Output:** Six values from neural-network inference.

| Output | Type | Range | Description                                               |
| ------ | ---- | ----- |-----------------------------------------------------------|
| `valence` | `float` | [0, 1] | Musical positivity (0 = Sad/Angry, 1 = happy).            |
| `arousal` | `float` | [0, 1] | Energetic activation (0 = calm, 1 = intense).             |
| `acousticness` | `float` | [0, 1] | Confidence the track is acoustic vs. electronic/produced. |
| `instrumentalness` | `float` | [0, 1] | Confidence the track lacks vocals.                        |
| `speechiness` | `float` | [0, 1] | Confidende the track contains spoken words.               |
| `embedding` | `list[float]` | 200-D vector) | Dense audio representation for similarity search.         |
## Validation CSV format

`expected_values.csv` uses Spotify's `audio_features` schema, with one extra leading column `file_path` pointing at the local WAV. The harness:

- Compares each numeric feature to its Spotify reference with a per-feature tolerance (tempo ±5 BPM octave-tolerant, loudness ±3 dB, duration ±2000 ms, [0,1] features ±0.15–0.30).
- Compares `key` (0–11) and `mode` (0/1) as ints directly; key allows fifth-confusion (±7 semitones mod 12).
- Reports Pearson `r` per feature across all rows — the real signal for heuristic features like `energy`, `danceability`, `speechiness`.
- Reports `liveness` and `time_signature` as fixed-default fields (0.0 and 4 respectively — see "Essentia Adapter" above for rationale).
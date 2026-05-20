from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from pathlib import Path

DEFAULT_DATASET = "anxods/spotify-top-50-playlist-songs-anxods"
DEFAULT_OUTPUT_DIR = Path("tracks/top50")
TRACK_COLS = ("track_name", "song", "name", "title")
ARTIST_COLS = ("artists", "artist", "track_artist")
DURATION_COLS = ("duration_ms", "duration", "track_duration_ms")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")


def setup_kaggle_env() -> None:
    username = os.environ.get("KAGGLE_USERNAME")
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not username or not token:
        sys.exit("error: set KAGGLE_USERNAME and KAGGLE_API_TOKEN env vars")
    os.environ["KAGGLE_KEY"] = token


def download_dataset(dataset: str, dest: Path) -> Path:
    setup_kaggle_env()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(dest), unzip=True)
    csvs = sorted(dest.rglob("*.csv"))
    if not csvs:
        sys.exit(f"error: no CSV found in dataset {dataset}")
    return csvs[0]


def pick_column(header: list[str], candidates: tuple[str, ...]) -> str:
    lower = {h.lower(): h for h in header}
    for c in candidates:
        if c in lower:
            return lower[c]
    sys.exit(f"error: none of {candidates} found in CSV header {header}")


def read_tracks(csv_path: Path, limit: int) -> list[tuple[str, str, int | None]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        track_col = pick_column(header, TRACK_COLS)
        artist_col = pick_column(header, ARTIST_COLS)
        # Duration column is optional — soft lookup so datasets without it still work.
        lower = {h.lower(): h for h in header}
        duration_col = next((lower[c] for c in DURATION_COLS if c in lower), None)
        rows = []
        for row in reader:
            title = (row.get(track_col) or "").strip()
            artist = (row.get(artist_col) or "").strip()
            duration_ms: int | None = None
            if duration_col:
                raw = (row.get(duration_col) or "").strip()
                try:
                    duration_ms = int(float(raw)) if raw else None
                except ValueError:
                    duration_ms = None
            if title and artist:
                rows.append((artist, title, duration_ms))
            if len(rows) >= limit:
                break
        return rows


def _score_candidate(entry: dict, expected_ms: int | None) -> float:
    dur = entry.get("duration")
    if dur is None:
        return float("inf")
    if expected_ms is None:
        return 0.0
    return abs(int(dur * 1000) - expected_ms)


def download_track(
    artist: str,
    title: str,
    expected_ms: int | None,
    out_dir: Path,
    browser: str | None,
) -> tuple[str, str, int | None]:
    """Two-phase: ytsearch5 metadata, pick by duration delta, download winner.

    Returns (slug, status, delta_ms) where delta_ms is the winner's duration
    delta vs expected_ms (None if either is unknown).
    """
    from yt_dlp import YoutubeDL

    slug = slugify(f"{artist}_{title}")
    target = out_dir / f"{slug}.wav"
    if target.exists():
        return slug, "skipped", None

    query = f"ytsearch5:{artist} {title} official audio"

    # Phase A: pull metadata for 5 candidates, no download.
    search_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extract_flat": False,
        "remote_components": ["ejs:github"],
    }
    if browser:
        search_opts["cookiesfrombrowser"] = (browser,)
    with YoutubeDL(search_opts) as ydl:
        info = ydl.extract_info(query, download=False)
    candidates = [e for e in (info.get("entries") or []) if e]
    if not candidates:
        return slug, "no-candidates", None

    # Phase B: closest duration wins; unknown-duration entries score inf so they only
    # win when nothing else exists. With expected_ms=None all scores tie at 0 → first hit.
    winner = min(candidates, key=lambda e: _score_candidate(e, expected_ms))
    winner_dur = winner.get("duration")
    delta_ms: int | None = None
    if expected_ms is not None and winner_dur is not None:
        delta_ms = int(winner_dur * 1000) - expected_ms

    # Phase C: download the chosen video by URL.
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / f"{slug}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "remote_components": ["ejs:github"],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([winner["webpage_url"]])

    if not target.exists():
        return slug, "missing-wav", delta_ms
    return slug, "downloaded", delta_ms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="print queries only")
    p.add_argument("--cookies-from-browser", default=None,
                   help="browser to source cookies from (e.g. chrome, firefox, safari) for YouTube bot-check")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = download_dataset(args.dataset, Path(tmp))
        tracks = read_tracks(csv_path, args.limit)

    if not tracks:
        sys.exit("error: no tracks parsed from CSV")
    print(f"parsed {len(tracks)} tracks from {csv_path.name}")

    if args.dry_run:
        for artist, title, expected_ms in tracks:
            suffix = f"  (expect ~{expected_ms / 1000:.0f}s)" if expected_ms else ""
            print(f"  ytsearch5:{artist} {title} official audio  ->  {slugify(f'{artist}_{title}')}.wav{suffix}")
        return 0

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for i, (artist, title, expected_ms) in enumerate(tracks, 1):
        prefix = f"[{i}/{len(tracks)}]"
        try:
            slug, status, delta_ms = download_track(
                artist, title, expected_ms, args.output_dir, args.cookies_from_browser
            )
            counts[status if status in ("downloaded", "skipped") else "failed"] += 1
            delta_str = ""
            if delta_ms is not None and expected_ms is not None:
                delta_str = f" (Δ={delta_ms / 1000:+.0f}s vs spotify {expected_ms / 1000:.0f}s)"
            print(f"{prefix} {status}: {slug}{delta_str}")
        except Exception as e:
            counts["failed"] += 1
            print(f"{prefix} failed: {artist} - {title}: {e}")

    print(f"\n{counts['downloaded']} downloaded, {counts['skipped']} skipped, {counts['failed']} failed")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

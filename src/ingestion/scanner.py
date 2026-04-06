"""Recursively discover CSV files across all data-source directories."""

from pathlib import Path

from src.config import get_settings


def scan_corpus() -> list[Path]:
    """Return sorted list of all *_output.csv files under configured data sources."""
    settings = get_settings()
    root = settings.data_root
    # If settings.data_root points to workspace root, but a nested data/ folder
    # exists (common local layout), scan that folder automatically.
    if root.name != "data" and (root / "data").is_dir():
        root = root / "data"
    csv_files: list[Path] = []

    for source in settings.data_sources:
        source_dir = root / source
        if not source_dir.is_dir():
            continue
        csv_files.extend(sorted(source_dir.rglob("*_output.csv")))

    return csv_files

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_ratings(path: Path | str | None = None) -> pd.DataFrame:
    """Load MovieLens user-item ratings."""
    source = Path(path) if path is not None else _DATA_DIR / "ratings.csv"
    return pd.read_csv(source)


def load_items(path: Path | str | None = None) -> pd.DataFrame:
    """Load MovieLens movie metadata (items)."""
    source = Path(path) if path is not None else _DATA_DIR / "movies.csv"
    return pd.read_csv(source)

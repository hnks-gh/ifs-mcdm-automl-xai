"""
src/utils/io_utils.py
---------------------
I/O utilities for saving and loading pipeline outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.core.schema import RankingResult


def save_ranking_results(
    result: RankingResult,
    filepath: Union[str, Path],
    format: str = "csv",
) -> None:
    """
    Save ranking results to file.

    Parameters
    ----------
    result : RankingResult
        Ranking result from IF-WASPAS, IF-TOPSIS, or IF-PROMETHEE II.
    filepath : str or Path
        Output file path.
    format : str
        Output format: "csv" or "parquet". Default "csv".
    """
    filepath = Path(filepath)

    # Create output dataframe
    df = pd.DataFrame({
        "Province": result.provinces,
        "Rank": result.ranks,
        "Score": result.scores,
    })

    # Add metadata columns
    df.insert(0, "Year", result.year)
    df.insert(1, "Method", result.method.value)

    # Save to file
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "csv":
        df.to_csv(filepath, index=False)
    elif format.lower() == "parquet":
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")

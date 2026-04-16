"""Shared pytest fixtures using a real slice of ATP match data.

No synthetic data, no mocks. Tests that depend on the ATP database will be
skipped automatically if the file is not present. Set the ATP_DATA_PATH
environment variable to override the default location.
"""

import os

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_ATP_PATH = os.path.expanduser("~/Data/tennis/tennis_data/atp_database.csv")
_ATP_DATA_ENV = "ATP_DATA_PATH"
_FIXTURE_ROWS = 200  # small enough to be fast; large enough to be representative


def _locate_atp_data() -> str | None:
    """Return the ATP database path if it exists, otherwise None."""
    path = os.environ.get(_ATP_DATA_ENV, _DEFAULT_ATP_PATH)
    return path if os.path.isfile(path) else None


# ---------------------------------------------------------------------------
# Session-scoped fixtures (loaded once per pytest run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def atp_data_path() -> str:
    """Absolute path to the ATP database CSV.

    Skips all dependent tests if the file is not present.
    """
    path = _locate_atp_data()
    if path is None:
        pytest.skip(
            f"ATP database not found. Place it at {_DEFAULT_ATP_PATH} "
            f"or set the {_ATP_DATA_ENV} environment variable."
        )
    return path


@pytest.fixture(scope="session")
def atp_sample(atp_data_path: str) -> pd.DataFrame:
    """200-row slice of real ATP match data, chronologically ordered.

    Columns include: tourney_date, player1_id, player2_id, game_winner,
    player1_elo, player2_elo, player1_winning_streak, player1_losing_streak,
    player1_weeks_inactive, player1_last_two_weeks, player1_v_player2_wins,
    sine_day, cosine_day, year, surface, round, etc.
    """
    df = pd.read_csv(atp_data_path, nrows=_FIXTURE_ROWS, low_memory=False)
    df = df.sort_values("tourney_date").reset_index(drop=True)
    return df

from __future__ import annotations

from bisect import bisect_left
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")


def fetch_benchmark_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers=ticker,
        start=start_date.isoformat(),
        end=(end_date + pd.Timedelta(days=2)).isoformat(),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise ValueError(f"No benchmark data returned for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            close = data.xs("Adj Close", axis=1, level=0)
        else:
            close = data.xs("Close", axis=1, level=0)
    else:
        close_column = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = data[close_column]

    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("Expected a single benchmark close series.")
        close = close.iloc[:, 0]

    sessions = close.rename("close").reset_index()
    date_column = sessions.columns[0]
    sessions["session_date"] = pd.to_datetime(sessions[date_column]).dt.date
    sessions = sessions[["session_date", "close"]].dropna()
    sessions = sessions.drop_duplicates(subset=["session_date"], keep="last").sort_values("session_date")
    sessions.reset_index(drop=True, inplace=True)
    return sessions


def resolve_base_session(snapshot_at: datetime, session_dates: list[date]) -> date | None:
    local_time = snapshot_at.astimezone(MARKET_TZ)
    candidate = local_time.date()
    market_close = datetime.combine(candidate, time(hour=16), tzinfo=MARKET_TZ)
    if candidate in session_dates and local_time >= market_close:
        return candidate
    index = bisect_left(session_dates, candidate) - 1
    if index < 0:
        return None
    return session_dates[index]


def resolve_future_session(base_session_date: date, target_date: date, session_dates: list[date]) -> date | None:
    if base_session_date is None:
        return None
    index = bisect_left(session_dates, target_date)
    if index >= len(session_dates):
        return None
    return session_dates[index]

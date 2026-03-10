# sp500_tech_analyser

Investtech-based S&P 500 signal ingestion, out-of-sample evaluation, and Streamlit dashboarding.

The project captures raw Investtech snapshots, rebuilds normalized datasets, evaluates each score-to-horizon mapping with walk-forward validation, and presents the results in a read-first dashboard.

## What This Does

- Captures and stores raw Investtech snapshots
- Normalizes snapshots into a stable internal schema
- Enriches signals with benchmark market returns for `^GSPC`
- Evaluates signals out of sample with expanding-window walk-forward logic
- Builds calibration, strategy, and summary artifacts for the dashboard
- Serves a Streamlit dashboard focused on latest signal usability and historical validation

## Project Layout

```text
.
├── app.py
├── main.py
├── requirements.txt
├── data/
│   ├── raw/investtech/
│   └── processed/
├── sp500_tech_analyser/
│   ├── cli.py
│   ├── cloud.py
│   ├── config.py
│   ├── constants.py
│   ├── dashboard.py
│   ├── evaluation.py
│   ├── market.py
│   ├── pipeline.py
│   ├── plotting.py
│   ├── providers/
│   └── storage.py
└── tests/
```

## Installation

Create and activate a virtualenv, then install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Main Commands

Refresh raw snapshots into `data/raw/investtech/`:

```bash
python -m sp500_tech_analyser refresh-raw
```

Rebuild processed artifacts into `data/processed/`:

```bash
python -m sp500_tech_analyser build
```

Run the dashboard:

```bash
streamlit run app.py
```

Run tests:

```bash
.venv/bin/python -m pytest -q
```

## Cloud Capture

`main.py` is the cloud capture entrypoint. It fetches the latest Investtech snapshot, writes it locally, and uploads it to GCS.

For local execution:

```bash
python main.py
```

## Evaluation Methodology

The system does not optimize thresholds on the full sample and report that as performance. Instead:

1. Raw Investtech scores are mapped only to these supported horizons:
   - `short_score -> 1w, 2w, 4w`
   - `medium_score -> 13w`
   - `long_score -> 26w`
2. Each snapshot is aligned to the last completed US market session.
3. Forward benchmark returns are computed for each supported horizon.
4. Walk-forward evaluation starts only after at least 20 labeled observations.
5. At each step, thresholds are fit on past data only.
6. The next observation is scored strictly out of sample.
7. Strategy returns are then simulated with non-overlapping trades per horizon, so capital is not double-counted across overlapping forward windows.
8. Metrics are aggregated across the full out-of-sample sequence.

The strategy interpretation is:

- `score > threshold`: long
- `score < -threshold`: short
- otherwise: flat

## Verdict Logic

Each signal-to-horizon mapping is assigned one verdict:

- `Reliable`: at least 25 out-of-sample observations, directional accuracy at least 55%, and strategy return beats buy-and-hold
- `Experimental`: at least 15 observations and exactly one of those two performance checks passes
- `Unreliable`: everything else

## Processed Artifacts

The dashboard reads only from `data/processed/`.

- `snapshots.csv`: normalized snapshot rows
- `signals.csv`: normalized rows plus benchmark alignment and forward returns
- `evaluation.csv`: out-of-sample metrics and verdicts
- `calibration.csv`: fixed score-bucket calibration statistics
- `strategies.csv`: executed non-overlapping trades, with trade returns and cumulative equity
- `dashboard_summary.json`: executive-summary data for the UI

## Environment Variables

These are optional. Defaults are defined in `sp500_tech_analyser/config.py`.

- `SP500_TECH_LOG_LEVEL`
- `SP500_TECH_PROJECT`
- `SP500_TECH_BUCKET`
- `SP500_TECH_DATA_ROOT`
- `SP500_TECH_BENCHMARK`
- `SP500_TECH_RAW_GCS_PREFIX`
- `SP500_TECH_INVESTTECH_URL`

Default local behavior uses:

- data root: `data`
- benchmark: `^GSPC`
- log level: `INFO`

## Logging

CLI commands emit standard Python logging to stderr and return the final machine-readable JSON payload to stdout.

Example:

```bash
python -m sp500_tech_analyser --log-level DEBUG build
```

Or via environment:

```bash
SP500_TECH_LOG_LEVEL=DEBUG python -m sp500_tech_analyser build
```

## Using The Dashboard For Decisions

Read the dashboard in this order:

1. `Executive Summary`
   - Check the latest score, verdict, and usage guidance.
2. `Historical Validation`
   - Focus on sample count, directional accuracy, excess return vs buy-and-hold, and max drawdown.
3. `Calibration`
   - Check what historically happened after similar score buckets.
4. `Diagnostics`
   - Review warnings, missing labels, and threshold history.

Practical rule:

- `Reliable` signals can be used as a directional input for S&P 500 exposure.
- `Experimental` signals should only confirm other evidence.
- `Unreliable` signals should not drive capital allocation on their own.

This project is a research and decision-support tool, not personalized investment advice.

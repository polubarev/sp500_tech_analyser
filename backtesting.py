import os
import glob
import json
from datetime import timedelta
import warnings

import pandas as pd
import yfinance as yf
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    classification_report
)

# 1) Load JSON predictions into a DataFrame
def load_predictions(folder='predictions/'):
    records = []
    for filepath in glob.glob(os.path.join(folder, '*.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
        dt = (
            pd.to_datetime(data['datetime'], utc=True)
              .tz_convert('Asia/Jerusalem')
              .tz_convert('UTC')
              .round('h')
              .tz_localize(None)
        )
        records.append({
            'datetime': dt,
            'short_score': data['short_term']['score'],
            'medium_score': data['medium_term']['score'],
            'long_score': data['long_term']['score'],
        })
    df = pd.DataFrame(records).set_index('datetime').sort_index()
    return df

def prepare_predictions_for_horizons(preds_df, horizons):
    """
    Prepare predictions DataFrame with scores mapped to different horizons
    
    Args:
        preds_df: DataFrame with original predictions
        horizons: dict of horizon names to (timedelta, score_type) tuples
    """
    preds = preds_df.copy()
    # Map scores to horizons
    for horizon_name, (_, score_type) in horizons.items():
        if score_type not in ['short', 'medium', 'long']:
            raise ValueError(f"Invalid score type '{score_type}'. Must be one of: short, medium, long")
        preds[f'{horizon_name}_score'] = preds[f'{score_type}_score']
    return preds

# 2) Fetch daily S&P 500 closes covering your prediction range
def fetch_daily_close(start, end):
    data = yf.download(
        tickers='^GSPC',
        start=start,
        end=end,
        interval='1d',
        progress=False,
        auto_adjust=False
    )
    if 'Close' in data.columns:
        close = data['Close']
    else:
        try:
            close = data.xs('Close', axis=1, level=0)
        except KeyError:
            raise ValueError("Close column not found in downloaded data")
    close = close.copy()
    close.name = 'close'
    if close.index.tz is None:
        close.index = close.index.tz_localize('America/New_York', ambiguous='infer')
    close.index = close.index.tz_convert('UTC').tz_localize(None)
    return close

# 3) Compute realized returns at fixed horizons
def compute_returns(preds, price, horizons):
    preds = preds.copy()
    preds['date'] = preds.index.normalize()

    # Prepare price_by_date lookup
    if isinstance(price, pd.Series):
        price_df = price.to_frame('close')
    else:
        price_df = price.copy()
        if 'close' not in price_df.columns:
            if price_df.shape[1] == 1:
                price_df.columns = ['close']
            else:
                raise ValueError("Price DataFrame must have a 'close' column or be single-column.")
    price_df['date'] = price_df.index.normalize()
    price_by_date = price_df.groupby('date')['close'].last()

    # Compute returns per term, only where future price exists
    for term, delta in horizons.items():
        base = preds['date'].map(price_by_date)
        future_dates = preds['date'] + delta
        future = future_dates.map(price_by_date)
        mask = base.notna() & future.notna()
        if mask.sum() == 0:
            warnings.warn(f"No overlapping price data for horizon '{term}' (delta={delta}).")
        # Compute returns only for valid
        returns = pd.Series(index=preds.index, dtype=float)
        returns.loc[mask] = (future[mask] - base[mask]) / base[mask]
        preds[f'R_{term}'] = returns
    return preds

# 4) Evaluate both regression (scores) and classification (recommendations)
def evaluate(preds, score_terms, thresholds):
    results = []
    for term in score_terms:
        score = preds[f'{term}_score']
        ret = preds[f'R_{term}']
        # only valid pairs
        mask = score.notna() & ret.notna()
        if mask.sum() < 2:
            warnings.warn(f"Insufficient data for term '{term}' (valid pairs: {mask.sum()}), skipping evaluation.")
            continue
        score_v = score[mask]
        ret_v = ret[mask]
        # Correlation metric
        pearson_r = score_v.corr(ret_v)
        # Classification: map returns to Up/Flat/Down
        def true_label(x):
            if x > thresholds['ret_up']:
                return 'Up'
            if x < -thresholds['ret_up']:
                return 'Down'
            return 'Flat'
        true_cls = ret_v.apply(true_label)
        bins = [-100, -thresholds['score_flat'], thresholds['score_flat'], 100]
        labels = ['Down', 'Flat', 'Up']
        pred_cls = pd.cut(score_v, bins=bins, labels=labels)
        report = classification_report(true_cls, pred_cls, output_dict=True, zero_division=0)
        results.append({
            'term': term,
            'pearson_r': pearson_r,
            'accuracy': report['accuracy'],
            'precision_up': report['Up']['precision'],
            'recall_up': report['Up']['recall'],
            'f1_up': report['Up']['f1-score'],
        })
    return pd.DataFrame(results).set_index('term')

# —— Main ——
if __name__ == '__main__':
    preds_df = load_predictions('predictions/')
    horizons = {
        'short_1w': (timedelta(weeks=1), 'short'),
        'short_2w': (timedelta(weeks=2), 'short'),
        'short_3w': (timedelta(weeks=3), 'short'),
        'short_4w': (timedelta(weeks=4), 'short'),
        'medium_13w': (timedelta(weeks=13), 'medium'),
        'long_26w': (timedelta(weeks=26), 'long')
    }
    preds_df = prepare_predictions_for_horizons(preds_df, horizons)
    start = preds_df.index.min().strftime('%Y-%m-%d')
    end = (preds_df.index.max() + timedelta(weeks=27)).strftime('%Y-%m-%d')
    daily_close = fetch_daily_close(start, end)
    # Extract just the timedeltas for compute_returns
    horizons_deltas = {name: delta for name, (delta, _) in horizons.items()}
    preds_with_ret = compute_returns(preds_df, daily_close, horizons_deltas)
    thresholds = {'ret_up': 0.00, 'score_flat': 1}
    results = evaluate(
        preds_with_ret,
        score_terms=list(horizons.keys()),
        thresholds=thresholds
    )
    print('\n=== Backtest Quality Metrics ===')
    print(results.to_string(float_format='%.4f'))

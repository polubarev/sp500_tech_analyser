import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# 1) Load JSON predictions and align to UTC hours
folder = 'predictions/'  # adjust to your JSON folder
records = []
for filepath in glob.glob(os.path.join(folder, '*.json')):
    with open(filepath, 'r') as f:
        data = json.load(f)
    dt = pd.to_datetime(data['datetime'])
    # Localize or convert to Israel time
    if dt.tzinfo is None:
        dt = dt.tz_localize('Asia/Jerusalem')
    else:
        dt = dt.tz_convert('Asia/Jerusalem')
    # Convert to UTC and round
    dt = dt.tz_convert('UTC')
    dt_hour = dt.round('h').tz_localize(None)
    records.append({
        'datetime': dt_hour,
        'short_score': data['short_term']['score'],
        'medium_score': data['medium_term']['score'],
        'long_score': data['long_term']['score'],
    })

preds_df = (
    pd.DataFrame(records)
      .set_index('datetime')
      .sort_index()
)

# 2) Fetch intraday S&P 500
start = preds_df.index.min().strftime('%Y-%m-%d')
end   = preds_df.index.max().strftime('%Y-%m-%d')

yf_data = yf.download(
    tickers='^GSPC',
    start=start,
    end=end,
    interval='30m',
    progress=False,
    auto_adjust=False,
)

# Extract the Close series, ensuring it's a single Series
if 'Close' in yf_data.columns:
    close_ser = yf_data['Close']
else:
    close_ser = yf_data.xs('Close', axis=1, level=0)
if isinstance(close_ser, pd.DataFrame):
    if close_ser.shape[1] == 1:
        close_ser = close_ser.iloc[:, 0]
    else:
        raise ValueError('Expected single Close column, got multiple')

# 3) Build continuous hourly price DataFrame
price_df = close_ser.to_frame(name='Close')
if price_df.index.tz is None:
    price_df.index = price_df.index.tz_localize('America/New_York', ambiguous='infer')
price_df.index = price_df.index.tz_convert('UTC').tz_localize(None)
hourly_price = (
    price_df
      .resample('h')
      .last()
      .ffill()
)
hourly_price.index.name = 'datetime'

# 4) Merge and plot all term scores against price
merged = hourly_price.join(preds_df, how='inner')
fig, ax_price = plt.subplots(figsize=(12, 6))
# Price on primary axis
ax_price.plot(merged.index, merged['Close'], label='S&P 500 Hourly Close (UTC)', color='tab:blue')
ax_price.set_xlabel('Datetime (UTC)')
ax_price.set_ylabel('Close Price (USD)')
ax_price.grid(True)

# Prediction scores on secondary axis
ax_scores = ax_price.twinx()
ax_scores.plot(merged.index, merged['short_score'], linestyle='--', label='Short-Term Score')
ax_scores.plot(merged.index, merged['medium_score'], linestyle='-.', label='Medium-Term Score')
ax_scores.plot(merged.index, merged['long_score'], linestyle=':', label='Long-Term Score')
ax_scores.set_ylabel('Prediction Score')

# Combine legends from both axes
price_handles, price_labels = ax_price.get_legend_handles_labels()
score_handles, score_labels = ax_scores.get_legend_handles_labels()
ax_price.legend(price_handles + score_handles, price_labels + score_labels, loc='upper left')

plt.title('Hourly S&P 500 Close vs. Prediction Scores (UTC)')
plt.tight_layout()
plt.show()

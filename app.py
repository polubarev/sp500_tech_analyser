import streamlit as st
import backtesting
import plot_results
import update_predictions
from datetime import timedelta

st.title("S&P 500 Technical Analysis")

if st.button("Update Predictions"):
    with st.spinner("Updating predictions from GCS..."):
        update_predictions.update_predictions_from_gcs()
    st.success("Predictions updated.")

if st.button("Run Analysis"):
    with st.spinner("Running backtesting analysis..."):
        horizons = {
            'short_1w': (timedelta(weeks=1), 'short'),
            'short_2w': (timedelta(weeks=2), 'short'),
            'short_3w': (timedelta(weeks=3), 'short'),
            'short_4w': (timedelta(weeks=4), 'short'),
            'medium_13w': (timedelta(weeks=13), 'medium'),
            'long_26w': (timedelta(weeks=26), 'long')
        }
        thresholds = {'ret_up': 0.00, 'score_flat': 1}
        
        preds_df = backtesting.load_predictions('predictions/')
        preds_df = backtesting.prepare_predictions_for_horizons(preds_df, horizons)
        
        start = preds_df.index.min().strftime('%Y-%m-%d')
        end = (preds_df.index.max() + timedelta(weeks=27)).strftime('%Y-%m-%d')
        daily_close = backtesting.fetch_daily_close(start, end)

        horizons_deltas = {name: delta for name, (delta, _) in horizons.items()}
        preds_with_ret = backtesting.compute_returns(preds_df, daily_close, horizons_deltas)
        
        results = backtesting.evaluate(
            preds_with_ret,
            score_terms=list(horizons.keys()),
            thresholds=thresholds
        )
    st.success("Backtesting complete.")
    st.write("### Backtest Quality Metrics")
    st.dataframe(results)

    with st.spinner("Generating plots..."):
        fig = plot_results.generate_plot()
    st.success("Plot generated.")
    st.pyplot(fig)
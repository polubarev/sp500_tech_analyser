import streamlit as st
import backtesting
import plot_results
import update_predictions
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def run_backtest(df, score_col, return_col, threshold):
    """
    Runs a simple backtest for a given score and return column.

    Strategy:
    - Go long (buy) if score > threshold.
    - Go short (sell) if score < -threshold.
    - Hold the position for the period defined by the return_col.

    Args:
        df (pd.DataFrame): The data to backtest on.
        score_col (str): The name of the score column to use.
        return_col (str): The name of the future return column.
        threshold (int): The score threshold to trigger a trade.

    Returns:
        float: The cumulative return of the strategy as a multiplier (e.g., 1.1 for 10% return).
    """
    # Create signals: 1 for long, -1 for short, 0 for hold
    df['signal'] = 0
    df.loc[df[score_col] > threshold, 'signal'] = 1
    df.loc[df[score_col] < -threshold, 'signal'] = -1

    # Calculate the returns from this strategy
    # The return is the signal (1, -1, or 0) multiplied by the actual future market return
    strategy_returns = df['signal'] * df[return_col]

    # Calculate cumulative return (compounded)
    cumulative_return = (1 + strategy_returns).prod()

    return cumulative_return


def optimize_thresholds(df, term):
    """
    Optimizes the score threshold for a given term (short, medium, or long).

    Args:
        df (pd.DataFrame): The data to use for optimization.
        term (str): The term to optimize ('short_term', 'medium_term', 'long_term').

    Returns:
        tuple: A tuple containing (best_threshold, best_return, results_df).
    """
    score_col = f'{term}_score'
    return_col = f'{term}_return'

    # Define the range of thresholds to test
    thresholds_to_test = range(0, 101, 5)
    results = []

    # Run backtest for each threshold
    for threshold in thresholds_to_test:
        performance = run_backtest(df.copy(), score_col, return_col, threshold)
        results.append({'threshold': threshold, 'performance': performance})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Find the best performing threshold
    best_result = results_df.loc[results_df['performance'].idxmax()]
    best_threshold = int(best_result['threshold'])
    best_performance = best_result['performance']

    return best_threshold, best_performance, results_df


def plot_optimization_results(results_df, term, best_threshold, best_performance):
    """Plots the performance vs. threshold."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(results_df['threshold'], results_df['performance'], marker='o', linestyle='-', label='Strategy Performance')

    # Highlight the best point
    ax.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {best_threshold}')
    ax.axhline(1.0, color='black', linestyle='-', linewidth=0.7, label='Buy and Hold Baseline')

    ax.set_title(f'Optimization for {term.replace("_", " ").title()}', fontsize=16)
    ax.set_xlabel('Score Threshold', fontsize=12)
    ax.set_ylabel('Cumulative Return Multiplier', fontsize=12)
    ax.legend()
    ax.grid(True)

    return fig


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

if st.button("Show Analysis with Market Data"):
    st.write("### Analysis with Market Data")
    input_csv = 'add_analytics/analysis_with_market_data.csv'
    if not os.path.exists(input_csv):
        st.error(f"Error: File not found at '{input_csv}'.")
    else:
        df = pd.read_csv(input_csv, parse_dates=['date'])
        st.dataframe(df)

if st.button("Run Optimization"):
    st.write("### Backtesting and Optimizing Score Thresholds")
    input_csv = 'add_analytics/analysis_with_market_data.csv'

    if not os.path.exists(input_csv):
        st.error(f"Error: File not found at '{input_csv}'. Please run the consolidation and merge scripts first.")
    else:
        main_df = pd.read_csv(input_csv, parse_dates=['date'])

        for term in ['short_term', 'medium_term', 'long_term']:
            with st.spinner(f"Optimizing for {term.replace('_', ' ').title()}..."):
                best_threshold, best_perf, results = optimize_thresholds(main_df, term)

                st.write(f"#### Optimization for {term.replace('_', ' ').title()}")
                st.write(f"Optimal Score Threshold: **{best_threshold}**")
                st.write(f"Best Performance (Return Multiplier): **{best_perf:.4f}**")
                st.write(f"This represents a **{(best_perf - 1) * 100:.2f}%** cumulative return over the period.")

                st.write("##### Optimization Performance")
                st.dataframe(results)

                fig = plot_optimization_results(results, term, best_threshold, best_perf)
                st.pyplot(fig)
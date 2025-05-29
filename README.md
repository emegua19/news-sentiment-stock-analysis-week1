# AIM Week 1 Project: Predicting Price Moves with News Sentiment

This project analyzes news sentiment to predict stock price movements using the FNSPID dataset for the 10 Academy AIM Week 1 Challenge.

## Project Overview
- **Goal**: Correlate financial news sentiment with Apple (AAPL) stock price changes to inform trading strategies.
- **Datasets**:
  - `raw_analyst_ratings.csv`: Financial news headlines (>1,000,000 rows).
  - `AAPL_historical_data.csv`: AAPL stock prices (1980–2024).
  - `sp500_historical_data_2020.csv`: S&P 500 index data for 2020 (benchmark).

## Setup Instructions (Windows)
1. Clone the repository: `git clone https://github.com/your-username/aim-week1-project.git`
2. Activate virtual environment: `venv\Scripts\activate`
3. Install TA-Lib: `pip install TA_Lib-0.4.0-cp310-cp310-win_amd64.whl` (download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
4. Install dependencies: `pip install -r requirements.txt`

## Git and GitHub Usage
- **Repository**: Hosted on GitHub with CI/CD pipeline (`.github/workflows/ci.yml`).
- **Branches**:
  - `task-1`: Project setup, data cleaning, and exploratory data analysis (EDA).
  - `task-2`: Quantitative analysis with technical indicators.
- **Commits**: 3+ daily, e.g., "Added data cleaning for 2020", "Completed topic modeling", "Implemented technical indicators".
- **Current Branch**: `task-1` for interim submission, with plans to merge into `main`.

## Task 1: Git, GitHub, and EDA (Completed)
- **Setup**:
  - Configured Python 3.10.11 environment with dependencies (`requirements.txt`).
  - Established project structure: `notebooks/`, `src/`, `data/`, `plots/`.
- **Data Cleaning** (`data_cleaning.ipynb`):
  - Processed `raw_analyst_ratings.csv` and `AAPL_historical_data.csv` for 2020 (~300–500 news rows, ~252 stock rows).
  - Standardized dates, cleaned headlines, and saved to `fnspid_news_cleaned_2020.csv`, `stock_prices_cleaned_2020.csv`.
- **Exploratory Data Analysis**:
  - **Descriptive Statistics** (`descriptive_statistics.ipynb`): Analyzed headline lengths (mean ~50 chars), publisher counts, and publication trends.
  - **Topic Modeling** (`topic_modeling.ipynb`): Extracted keywords (e.g., "price target") and 5 topics (e.g., earnings, market) using NLP.
  - **Time Series Analysis** (`time_series_analysis.ipynb`): Identified daily publication frequency and peak hours (e.g., 9 AM UTC).
  - **Publisher Analysis** (`publisher_analysis.ipynb`): Ranked top publishers (e.g., Reuters), extracted email domains (e.g., yahoo.com), and categorized news types (earnings, analyst, etc.).
- **Modularity**: Implemented functions in `src/` (`data_utils.py`, `nlp_utils.py`, `time_series_utils.py`, `publisher_utils.py`).

## Task 2: Quantitative Analysis (Partial Progress)
- **Stock Data Preparation** (`quantitative_analysis.ipynb`):
  - Loaded `stock_prices_cleaned_2020.csv` (~252 rows) with columns `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
  - Integrated S&P 500 data (`sp500_historical_data_2020.csv`) for market comparison.
- **Technical Indicators** (using `pynance` and `TA-Lib`):
  - Computed Simple Moving Average (SMA-20), Relative Strength Index (RSI-14), Moving Average Convergence Divergence (MACD), and Bollinger Bands.
  - Visualized indicators and AAPL vs. S&P 500 normalized prices (`plots/aapl_sma_2020.png`, `aapl_vs_sp500_2020.png`, etc.).
  - Saved results to `aapl_with_indicators_2020.csv`.
- **Functions**: Added `load_stock_data`, `compute_technical_indicators` in `src/finance_utils.py`.

## Next Steps
- **Task 2 Completion**: Add more TA-Lib indicators (e.g., ADX, Stochastic Oscillator) and validate results.
- **Task 3**: Correlate news sentiment (from topic modeling) with stock returns and technical indicators.
- **Final Submission**: Merge `task-1` and `task-2` into `main`, finalize report, and submit by June 3, 2025.

## Interim Submission
- **Deadline**: May 30, 2025, 8:00 PM UTC (11:00 PM EAT).
- **Deliverables**: GitHub main branch link, 3-page report (PDF/Word) with plots (e.g., `publisher_counts_2020.png`, `aapl_sma_2020.png`).
- **Submission**: Via Slack `#all-week1`.

## Key Outputs
- **Data**: `fnspid_news_cleaned_2020.csv`, `stock_prices_cleaned_2020.csv`, `aapl_with_indicators_2020.csv`.
- **Plots**: `headline_length_distribution_2020.png`, `publication_frequency_2020.png`, `publisher_counts_2020.png`, `aapl_sma_2020.png`, etc.
- **Notebooks**: `data_cleaning.ipynb`, `descriptive_statistics.ipynb`, `topic_modeling.ipynb`, `time_series_analysis.ipynb`, `publisher_analysis.ipynb`, `quantitative_analysis.ipynb`.
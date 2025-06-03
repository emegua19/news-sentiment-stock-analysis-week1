
# AIM Week 1 Project: Predicting Price Moves with News Sentiment

This project analyzes news sentiment to predict stock price movements using the FNSPID dataset for the **10 Academy AIM Week 1 Challenge**.

---

## Project Overview

### Goal

Correlate financial news sentiment with **Apple (AAPL)** stock price changes to inform trading strategies.

---

## Datasets

- `raw_analyst_ratings.csv`: Financial news headlines (>1,000,000 rows, filtered to ~300–500 rows for 2020 to align with stock data timeframe and reduce computational load)  
- `AAPL_historical_data.csv`: AAPL stock prices (1980–2024, focused on 2020 with ~252 rows)  
- `sp500_historical_data_2020.csv`: S&P 500 index data for 2020 (benchmark)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/emegua19/news-sentiment-stock-analysis-week1.git
cd news-sentiment-stock-analysis-week1
```

### 2. Create and Activate a Virtual Environment

#### On **Windows**:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### On **Ubuntu/Linux**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install TA-Lib (Required for Local Quantitative Analysis)

> **Note**: TA-Lib is needed to compute technical indicators locally for Task 2. Skip if using precomputed data.

#### On **Windows**:

1. Download the appropriate `.whl` file from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
2. Install it:

```cmd
pip install path\to\TA_Lib-<version>-cp<version>-cp<version>-win_amd64.whl
```

#### On **Ubuntu/Linux**:

```bash
sudo apt-get install -y libta-lib0 libta-lib-dev
pip install ta-lib
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

CI pipeline is configured via `.github/workflows/ci.yml` to trigger on:

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
```

It runs unit tests (e.g., for data processing and analysis functions) on multiple OS and Python versions:

- **OS**: Ubuntu, Windows
- **Python Versions**: 3.10, 3.11, 3.12

Includes caching and dependency installation steps for efficiency.

---

## Project Structure

```
.github/                  # CI/CD workflows
.gitignore                # Git ignore file
.venv/                    # Virtual environment
.vscode/                  # VS Code settings
LICENSE                   # Project license
README.md                 # Project documentation
data/                     # Dataset storage
notebooks/                # Jupyter notebooks for analysis
requirements.txt          # Python dependencies
scripts/                  # Additional scripts
src/                      # Source code modules
tests/                    # Test files
```

---

## Git and GitHub Usage

### Repository

Hosted on GitHub with CI/CD using `.github/workflows/ci.yml`.

### Branches

- `task-1`: Project setup, data cleaning, and exploratory data analysis (EDA)
- `task-2`: Quantitative analysis with technical indicators
- `task-3`: Correlation analysis between news sentiment and stock returns
- `dashboard`: Streamlit dashboard development and integration

All tasks merged into `main`.

### Commit Examples

- `Added data cleaning for 2020`
- `Completed topic modeling`
- `Implemented technical indicators`
- `Added sentiment correlation analysis`
- `Deployed Streamlit dashboard`

---

## Tasks

### Task 1: Git, GitHub, and EDA

#### Setup

- Python 3.10, 3.11, 3.12
- Installed requirements from `requirements.txt`

#### Data Cleaning

- Cleaned: `raw_analyst_ratings.csv`, `AAPL_historical_data.csv`
- Filtered for 2020 and saved:
  - `data/fnspid_news_cleaned_2020.csv`
  - `data/stock_prices_cleaned_2020.csv`

#### Exploratory Data Analysis

- **Descriptive Statistics**: Headline length, publisher frequency
- **Topic Modeling**: Common themes like price targets
- **Time Series Analysis**: Peak publish time ~9 AM UTC
- **Publisher Analysis**: Top publishers and domains

Scripts used:

- `src/data_utils.py`
- `src/nlp_utils.py`
- `src/time_series_utils.py`
- `src/publisher_utils.py`

---

### Task 2: Quantitative Analysis

#### Stock Data Preparation

- Merged `stock_prices_cleaned_2020.csv` and `sp500_historical_data_2020.csv`

#### Technical Indicators

Computed:

- SMA-20
- RSI-14
- MACD
- Bollinger Bands
- ADX
- Stochastic Oscillator

Saved results to:

- `data/aapl_with_indicators_2020.csv`

Visuals:

- `plots/aapl_sma_2020.png`
- `plots/aapl_vs_sp500_2020.png`

Functions in `src/finance_utils.py`:

- `load_stock_data`
- `compute_technical_indicators`

---

### Task 3: Correlation Analysis

#### Sentiment Analysis

- Used TextBlob
- Aggregated sentiment by day into `data/sentiment_returns_aapl_2020.csv`

#### Stock Returns & Correlation

- Computed daily returns
- Pearson correlation (lags: 0 to 3 days)
- Scatter plot: `task-3-plots/sentiment_vs_returns_aapl.png`

Functions in `src/correlation_analysis.py`:

- `perform_sentiment_analysis`
- `calculate_stock_returns`
- `align_data`
- `calculate_correlation`
- `plot_correlation`

---

## Dashboard

> **Note**: Deployment postponed for future work. Run locally with `streamlit run src/dashboard.py`.

### Development

- Streamlit app in `src/dashboard.py`
- Features:
  - Metrics: closing price, volatility, return
  - Charts: SMA, RSI, MACD
  - Correlation: sentiment vs. returns

---

## Deliverables

- GitHub Repo: [https://github.com/emegua19/news-sentiment-stock-analysis-week1](https://github.com/emegua19/news-sentiment-stock-analysis-week1)

---

## Key Outputs

### Data

- `data/fnspid_news_cleaned_2020.csv`
- `data/stock_prices_cleaned_2020.csv`
- `data/aapl_with_indicators_2020.csv`
- `data/sentiment_returns_aapl_2020.csv`

### Plots

- **Task 1**:
  - `task-1-plots/headline_length_distribution_2020.png`
  - `task-1-plots/publication_frequency_2020.png`
  - `task-1-plots/publisher_counts_2020.png`
- **Task 2**:
  - `task-2-plots/aapl_sma_2020.png`
- **Task 3**:
  - `task-3-plots/sentiment_vs_returns_aapl.png`

### Notebooks

- `notebooks/data_cleaning.ipynb`
- `notebooks/descriptive_statistics.ipynb`
- `notebooks/topic_modeling.ipynb`
- `notebooks/time_series_analysis.ipynb`
- `notebooks/publisher_analysis.ipynb`
- `notebooks/quantitative_analysis.ipynb`
- `notebooks/correlation_analysis.ipynb`

### Scripts

- `src/data_utils.py`
- `src/nlp_utils.py`
- `src/time_series_utils.py`
- `src/publisher_utils.py`
- `src/finance_utils.py`
- `src/correlation_analysis.py`
- `src/dashboard.py`

---

## Acknowledgments

Thanks to the 10 Academy team for their guidance and support. Built using Python, Streamlit, Plotly, and TA-Lib.



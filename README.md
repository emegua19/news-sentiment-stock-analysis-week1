# AIM Week 1 Project: Predicting Price Moves with News Sentiment

This project analyzes news sentiment to predict stock price movements using the FNSPID dataset for the **10 Academy AIM Week 1 Challenge**.

---

## 📊 Project Overview

### 🎯 Goal

Correlate financial news sentiment with **Apple (AAPL)** stock price changes to inform trading strategies.

---

## 📁 Datasets

- `raw_analyst_ratings.csv`: Financial news headlines (>1,000,000 rows, filtered to ~300–500 rows for 2020)
- `AAPL_historical_data.csv`: AAPL stock prices (1980–2024, focused on 2020)
- `sp500_historical_data_2020.csv`: S&P 500 index data for 2020 (benchmark)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/emegua19/news-sentiment-stock-analysis-week1.git
cd news-sentiment-stock-analysis-week1
````

### 2. Create and Activate a Virtual Environment

#### On Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### On Ubuntu/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install TA-Lib (Optional for Task 2)

#### On Windows:

Download and install the `.whl` file from [Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

```cmd
pip install TA_Lib‑<version>.whl
```

#### On Ubuntu/Linux:

```bash
sudo apt-get install -y libta-lib0 libta-lib-dev
pip install ta-lib
```

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Python Packages Used

* `streamlit==1.35.0`
* `pandas==2.2.2`
* `plotly==5.21.0`
* `statsmodels==0.14.1`
* `textblob==0.17.1`
* `matplotlib==3.8.4`
* `seaborn==0.13.2`
* `scikit-learn==1.4.2`
* `yfinance==0.2.40`
* `gensim==4.3.2`
* `numpy==1.26.4`
* `spacy==3.7.4`
* `python-dateutil==2.9.0.post0`
* `keybert==0.8.5`
* `ta-lib` (optional, for technical indicators)

---

## 🔄 CI/CD Pipeline

Continuous Integration is implemented via **GitHub Actions** to ensure reliability across environments.

### ✅ CI Workflow Configuration

Triggered on:

* Push to `main`
* Pull request to `main`

Runs on:

* Ubuntu and Windows
* Python 3.10, 3.11, 3.12

Performs:

* Dependency caching
* Installation of packages
* Python version check
* Placeholder for tests

### 📄 `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ['3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip cache purge
          pip install -r requirements.txt --index-url https://pypi.org/simple

      - name: Check Python version
        run: python --version

      - name: Run tests
        run: |
          if [ -d tests ]; then
            pytest tests/
          else
            echo "No tests defined yet."
```

---

## 🗂️ Project Structure

```
.github/                  # CI/CD workflows
data/                     # Dataset storage
notebooks/                # Jupyter notebooks for analysis
plots/                    # Output visualizations
scripts/                  # Helper scripts
src/                      # Core source code
tests/                    # Unit tests (optional)
requirements.txt          # Dependencies
README.md                 # Documentation
```

---

## 🧠 Tasks Overview

### ✅ Task 1: Git & EDA

* Cleaned and filtered datasets for 2020
* Performed topic modeling, publication frequency analysis, and headline stats
* Scripts: `src/data_utils.py`, `src/nlp_utils.py`, `src/time_series_utils.py`, `src/publisher_utils.py`

### ✅ Task 2: Quantitative Analysis

* Computed SMA, RSI, MACD, Bollinger Bands, ADX, Stochastic Oscillator
* Script: `src/finance_utils.py`
* Saved indicators to `data/aapl_with_indicators_2020.csv`

### ✅ Task 3: Sentiment & Correlation

* Performed sentiment analysis using TextBlob
* Correlated sentiment scores with AAPL returns
* Script: `src/correlation_analysis.py`

### ✅ Dashboard (Streamlit)

* App: `src/dashboard.py`
* Features: metrics, charts (SMA, RSI), sentiment-return correlation
* Run with:

```bash
streamlit run src/dashboard.py
```

---

## 🧪 Deliverables

### 📁 Data Files

* `data/fnspid_news_cleaned_2020.csv`
* `data/stock_prices_cleaned_2020.csv`
* `data/aapl_with_indicators_2020.csv`
* `data/sentiment_returns_aapl_2020.csv`

### 📊 Plots

* `plots/aapl_sma_2020.png`
* `task-3-plots/sentiment_vs_returns_aapl.png`

### 📓 Notebooks

* `notebooks/data_cleaning.ipynb`
* `notebooks/descriptive_statistics.ipynb`
* `notebooks/topic_modeling.ipynb`
* `notebooks/correlation_analysis.ipynb`

---

## 🙏 Acknowledgments

Special thanks to **10 Academy** for this challenge opportunity. Built using Python, Streamlit, Plotly, TA-Lib, and NLP libraries.

---

## 🔗 Repository

[https://github.com/emegua19/news-sentiment-stock-analysis-week1](https://github.com/emegua19/news-sentiment-stock-analysis-week1)
```

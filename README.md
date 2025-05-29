# AIM Week 1 Project: Predicting Price Moves with News Sentiment

This project analyzes news sentiment to predict stock price movements using the FNSPID dataset.

## Project Overview
- **Goal**: Correlate financial news sentiment with stock price changes.
- **Datasets**:
  - `raw_analyst_ratings.csv`: News headlines with over 1,000,000 rows.
  - `AAPL_historical_data.csv`: Apple stock prices.

## Setup Instructions (Windows)
1. Clone: `git clone https://github.com/your-username/aim-week1-project.git`
2. Activate venv: `venv\Scripts\activate`
3. Install TA-Lib: `pip install ta_lib-0.4.0-cp310-cp310-win_amd64.whl` (download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
4. Install deps: `pip install -r requirements.txt`

## Git and GitHub Usage
- **Repository**: Hosted on GitHub with CI/CD pipeline (`.github/workflows/ci.yml`).
- **Branches**: `task-1` (setup and EDA), `task-2` (quantitative analysis).
- **Commits**: 3+ daily, e.g., "Added project structure", "Completed EDA", "Started stock analysis".

## Task 1: Git, GitHub, and EDA
- **Setup**: Python 3.10.11 environment with all dependencies installed.
- **EDA**:
  - Cleaned news data (`data_cleaning.ipynb`).
  - Analyzed headline lengths, publisher activity, and publication trends (`descriptive_statistics.ipynb`, `publisher_analysis.ipynb`).
  - Extracted topics from headlines (`text_analysis.ipynb`).

## Task 2 Partial Progress: Quantitative Analysis
- Cleaned stock data (`AAPL_historical_data.csv`) and normalized dates (`stock_analysis.ipynb`).
- Started calculating technical indicators (e.g., SMA) using TA-Lib.

## Next Steps
- Complete Task 2: Finish technical indicators (RSI, MACD).
- Start Task 3: Correlate news sentiment with stock returns.

**Interim Submission**: Due 8:00 PM UTC, May 30, 2025.
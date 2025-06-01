AIM Week 1 Project: Predicting Price Moves with News Sentiment
This project analyzes news sentiment to predict stock price movements using the FNSPID dataset for the 10 Academy AIM Week 1 Challenge.
Project Overview
Goal: Correlate financial news sentiment with Apple (AAPL) stock price changes to inform trading strategies.
Datasets:

raw_analyst_ratings.csv: Financial news headlines (>1,000,000 rows, filtered to ~300–500 rows for 2020).
AAPL_historical_data.csv: AAPL stock prices (1980–2024, focused on 2020 with ~252 rows).
sp500_historical_data_2020.csv: S&P 500 index data for 2020 (benchmark).

Setup Instructions (Ubuntu)

Clone the repository:git clone https://github.com/emegua19/news-sentiment-stock-analysis-week1.git


Activate the virtual environment:python3 -m venv .venv
source .venv/bin/activate


Install TA-Lib (optional for local quantitative analysis):sudo apt-get install libta-lib-dev
pip install TA-Lib


Alternatively, download a compatible wheel from TA-Lib Python if available for your Python version.


Install dependencies for deployment:pip install -r requirements.txt


Install additional dependencies for local development (sentiment analysis, visualization, etc.):pip install -r requirements-dev.txt



Project Structure

.github/ - CI/CD workflows
.gitignore - Git ignore file
.venv/ - Virtual environment
.vscode/ - VS Code settings
LICENSE - Project license
README.md - Project documentation
data/ - Dataset storage
notebooks/ - Jupyter notebooks for analysis
requirements.txt - Python dependencies for deployment
requirements-dev.txt - Additional dependencies for local development
scripts/ - Additional scripts
src/ - Source code modules
tests/ - Test files

Git and GitHub Usage
Repository: Hosted on GitHub with CI/CD pipeline (.github/workflows/ci.yml).
Branches:

task-1: Project setup, data cleaning, and exploratory data analysis (EDA).
task-2: Quantitative analysis with technical indicators.
task-3: Correlation analysis between news sentiment and stock returns.
dashboard: Streamlit dashboard development and integration.

Commits: 3+ daily, e.g., "Added data cleaning for 2020", "Completed topic modeling", "Implemented technical indicators", "Added sentiment correlation analysis", "Deployed Streamlit dashboard".
Current Branch: All tasks and dashboard merged into main.
Tasks
Task 1: Git, GitHub, and EDA (Completed)

Setup:
Configured Python 3.10.11 or 3.13.3 environment with dependencies (requirements.txt and requirements-dev.txt).
Established project structure as shown above.


Data Cleaning (notebooks/data_cleaning.ipynb):
Processed raw_analyst_ratings.csv and AAPL_historical_data.csv for 2020 (~300–500 news rows, ~252 stock rows).
Standardized dates, cleaned headlines, and saved to data/fnspid_news_cleaned_2020.csv, data/stock_prices_cleaned_2020.csv.


Exploratory Data Analysis:
Descriptive Statistics (notebooks/descriptive_statistics.ipynb): Analyzed headline lengths (mean ~50 chars), publisher counts, and publication trends.
Topic Modeling (notebooks/topic_modeling.ipynb): Extracted keywords (e.g., "price target") and 5 topics (e.g., earnings, market) using NLP.
Time Series Analysis (notebooks/time_series_analysis.ipynb): Identified daily publication frequency and peak hours (e.g., 9 AM UTC).
Publisher Analysis (notebooks/publisher_analysis.ipynb): Ranked top publishers (e.g., Reuters), extracted email domains (e.g., yahoo.com), and categorized news types (earnings, analyst, etc.).


Modularity: Implemented functions in src/ (data_utils.py, nlp_utils.py, time_series_utils.py, publisher_utils.py).

Task 2: Quantitative Analysis (Completed)

Stock Data Preparation (notebooks/quantitative_analysis.ipynb):
Loaded data/stock_prices_cleaned_2020.csv (~252 rows) with columns Date, Open, High, Low, Close, Volume.
Integrated S&P 500 data (data/sp500_historical_data_2020.csv) for market comparison.


Technical Indicators (using pynance and TA-Lib):
Computed Simple Moving Average (SMA-20), Relative Strength Index (RSI-14), Moving Average Convergence Divergence (MACD), Bollinger Bands, Average Directional Index (ADX), and Stochastic Oscillator.
Visualized indicators and AAPL vs. S&P 500 normalized prices (plots/aapl_sma_2020.png, plots/aapl_vs_sp500_2020.png, etc.).
Saved results to data/aapl_with_indicators_2020.csv.


Functions: Added load_stock_data, compute_technical_indicators in src/finance_utils.py.

Task 3: Correlation Analysis (Completed)

Sentiment Analysis (notebooks/correlation_analysis.ipynb):
Performed sentiment analysis on data/fnspid_news_cleaned_2020.csv using TextBlob, adding Sentiment and Tone columns.
Aggregated daily sentiment into a daily_sentiment DataFrame.


Stock Returns:
Computed daily returns from data/aapl_with_indicators_2020.csv and saved to data/sentiment_returns_aapl_2020.csv.


Correlation Analysis:
Calculated Pearson correlation between sentiment and returns (lag 0, 1, 2, 3 days) using calculate_correlation in src/correlation_analysis.py.
Visualized sentiment vs. returns with a scatter plot (plots/sentiment_vs_returns_aapl.png).


Functions: Implemented perform_sentiment_analysis, calculate_stock_returns, align_data, calculate_correlation, plot_correlation in src/correlation_analysis.py.

Dashboard (Completed & Deployed)

Dashboard Development (src/dashboard.py):
Created an interactive Streamlit app displaying:
Key metrics (year-end close, cumulative return, max volatility).
Technical indicators (Close vs. SMA-20, RSI, MACD).
Sentiment vs. returns correlation with OLS trendline.


Data fetched from Google Drive for deployment compatibility.


Deployment:
Deployed on Streamlit Community Cloud with frozen requirements.txt (streamlit, pandas, plotly, statsmodels).
Public URL: https://emegua19-2pwns-week1-project.streamlit.app


Features: Responsive layout, interactive Plotly charts, and a conclusion summarizing insights.

Final Submission

Deadline: June 3, 2025, 11:59 PM EAT.
Deliverables:
GitHub repository link: https://github.com/emegua19/news-sentiment-stock-analysis-week1
Deployed Streamlit dashboard URL: https://emegua19-2pwns-week1-project.streamlit.app
5-page report (PDF/Word) with plots (e.g., plots/aapl_sma_2020.png, plots/sentiment_vs_returns_aapl.png) and analysis summary.


Submission: Via Slack #all-week1.

Key Outputs

Data:
data/fnspid_news_cleaned_2020.csv
data/stock_prices_cleaned_2020.csv
data/aapl_with_indicators_2020.csv
data/sentiment_returns_aapl_2020.csv


Plots:
plots/headline_length_distribution_2020.png
plots/publication_frequency_2020.png
plots/publisher_counts_2020.png
plots/aapl_sma_2020.png
plots/sentiment_vs_returns_aapl.png


Notebooks:
notebooks/data_cleaning.ipynb
notebooks/descriptive_statistics.ipynb
notebooks/topic_modeling.ipynb
notebooks/time_series_analysis.ipynb
notebooks/publisher_analysis.ipynb
notebooks/quantitative_analysis.ipynb
notebooks/correlation_analysis.ipynb


Scripts:
src/data_utils.py
src/nlp_utils.py
src/time_series_utils.py
src/publisher_utils.py
src/finance_utils.py
src/correlation_analysis.py
src/dashboard.py



Acknowledgments
Thanks to the 10 Academy team for the challenge and support.Built with ❤️ using Python, Streamlit, Plotly, and TA-Lib.
Note

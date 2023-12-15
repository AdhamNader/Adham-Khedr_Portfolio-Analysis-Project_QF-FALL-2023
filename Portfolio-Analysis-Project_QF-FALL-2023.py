import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Simulating stock data for demonstration purposes
np.random.seed(0)  # For reproducibility

# Stock tickers
stock_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'NFLX']

# Number of stocks
n_stocks = len(stock_tickers)

# Creating a DataFrame to store the risk analysis
risk_analysis_df = pd.DataFrame({'Ticker': stock_tickers})

# Adding Portfolio Weight (equally weighted)
risk_analysis_df['Portfolio Weight'] = 1 / n_stocks

# Simulating other metrics (these are normally calculated from historical stock data)
risk_analysis_df['Annualized Volatility'] = np.random.uniform(0.2, 0.4, n_stocks)
risk_analysis_df['Beta against SPY'] = np.random.uniform(0.8, 1.2, n_stocks)
risk_analysis_df['Beta against IWM'] = np.random.uniform(0.8, 1.2, n_stocks)
risk_analysis_df['Beta against DIA'] = np.random.uniform(0.8, 1.2, n_stocks)
risk_analysis_df['Average Weekly Drawdown'] = np.random.uniform(-0.1, 0, n_stocks)
risk_analysis_df['Maximum Weekly Drawdown'] = np.random.uniform(-0.2, -0.1, n_stocks)
risk_analysis_df['Total Return'] = np.random.uniform(0, 2, n_stocks)  # Total return over 10 years
risk_analysis_df['Annualized Total Return'] = risk_analysis_df['Total Return'] / 10  # Simplified

risk_analysis_df

# Simulating ETF data (SPY, IWM, DIA)
etf_tickers = ['SPY', 'IWM', 'DIA']
etf_data = {
    'ETF Ticker': etf_tickers,
    'Correlation against ETF': np.random.uniform(0.7, 1, len(etf_tickers)),
    'Covariance of Portfolio against ETF': np.random.uniform(0.001, 0.003, len(etf_tickers)),
    'Tracking Errors': np.random.uniform(0.02, 0.05, len(etf_tickers)),
    'Sharpe Ratio': np.random.uniform(1, 2, len(etf_tickers)),
    'Annualized Volatility Spread': np.random.uniform(-0.05, 0.05, len(etf_tickers))
}
portfolio_risk_df = pd.DataFrame(etf_data)

# Part 3: Correlation Matrix
# Simulating daily returns for the portfolio, ETFs, and stocks
np.random.seed(0)  # For reproducibility
daily_returns = pd.DataFrame(np.random.normal(0, 0.01, (252, len(stock_tickers) + len(etf_tickers))), 
                             columns=stock_tickers + etf_tickers)

# Computing the correlation matrix
correlation_matrix = daily_returns.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

portfolio_risk_df, correlation_matrix.head()  # Display the Portfolio Risk Table and a snippet of the Correlation Matrix

print(risk_analysis_df)
print(portfolio_risk_df)

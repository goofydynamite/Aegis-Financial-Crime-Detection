import pandas as pd
import ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

print("--- Advanced AI Trading Strategy & Backtest (Final Version) ---")

try:
    # --- Step 1: Load and Prepare Initial Data ---
    print("[1/5] Loading data...")
    prices_df = pd.read_csv('stock_prices.csv')
    news_df = pd.read_csv('news_headlines.csv')

    # Explicitly convert 'Date' columns to the correct datetime format
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    news_df['Date'] = pd.to_datetime(news_df['Date'])

    # Clean the 'Headline' column
    news_df['Headline'] = news_df['Headline'].fillna('').astype(str)

    print("[2/5] Calculating sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    news_df['Sentiment'] = news_df['Headline'].apply(lambda headline: analyzer.polarity_scores(headline)['compound'])

    # Merge datasets and set Date as the index
    df = pd.merge(prices_df, news_df.drop(columns=['Headline']), on='Date', how='left')
    df['Sentiment'].fillna(method='ffill', inplace=True)
    df.set_index('Date', inplace=True)

    # --- Step 2: Advanced Feature Engineering ---
    print("[3/5] Engineering advanced technical indicators (RSI, MACD)...")
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    # Calculate Relative Strength Index (RSI)
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

    # Calculate Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df.fillna(0, inplace=True)

    # --- Step 3: Predictive Modeling ---
    print("[4/5] Building ML model...")
    df['Target'] = (df['Close'].shift(-5) > df['Close'] * 1.02).astype(int)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 'RSI_14', 'MACD_12_26_9']
    X = df[features].iloc[:-5]
    y = df['Target'].iloc[:-5]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"ML Model Accuracy: {accuracy_score(y_test, predictions):.2f}")

    df['Position'] = model.predict(df[features])

    # --- Step 4: Backtesting ---
    print("[5/5] Backtesting the ML-based trading strategy...")
    df['System_Return'] = df['Close'].pct_change() * df['Position'].shift(1)
    df['Equity_Curve'] = (1 + df['System_Return']).cumprod()
    df['Buy_and_Hold'] = (1 + df['Close'].pct_change()).cumprod()

    total_return = df['Equity_Curve'].iloc[-1] - 1
    daily_returns = df['System_Return'].dropna()

    if daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    print("\n--- Backtest Results ---")
    print(f"Total Return (ML Strategy): {total_return:.2%}")
    print(f"Total Return (Buy and Hold): {df['Buy_and_Hold'].iloc[-1] - 1:.2%}")
    print(f"Sharpe Ratio (Risk-Adjusted Return): {sharpe_ratio:.2f}")

    # --- Final Output ---
    df.reset_index(inplace=True)
    output_filename = 'advanced_trading_analysis.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n✅ Success! Final analysis complete. Output file: {output_filename}")

except FileNotFoundError as e:
    print(f"\nERROR: File not found. Please make sure '{e.filename}' is in the same folder as the script.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
"""
Financial Data Processor

This module handles the acquisition and preprocessing of financial data
for quantum machine learning models.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class FinancialDataProcessor:
    """
    A class for processing financial time series data for quantum ML models.
    """
    
    def __init__(self):
        """Initialize the financial data processor."""
        self.data = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    def fetch_stock_data(self, ticker="AAPL", period="2y", interval="1d"):
        """
        Fetch historical stock data using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (e.g., '1d', '5d', '1mo', '3mo', '1y', '2y', '5y', 'max')
            interval: Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with stock data
        """
        self.data = yf.download(ticker, period=period, interval=interval)
        return self.data
    
    def fetch_multiple_stocks(self, tickers=["AAPL", "MSFT", "GOOGL"], period="1y"):
        """
        Fetch data for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period to fetch
            
        Returns:
            Dictionary of DataFrames with stock data
        """
        data_dict = {}
        for ticker in tickers:
            data_dict[ticker] = yf.download(ticker, period=period)
        
        self.data = data_dict
        return data_dict
    
    def add_technical_indicators(self, data=None):
        """
        Add technical indicators to the financial data.
        
        Args:
            data: DataFrame with financial data (uses self.data if None)
            
        Returns:
            DataFrame with added technical indicators
        """
        if data is None:
            data = self.data
            
        if isinstance(data, dict):
            # If we have multiple stocks
            for ticker in data:
                data[ticker] = self._add_indicators_to_df(data[ticker])
            return data
        else:
            # Single dataframe
            return self._add_indicators_to_df(data)
    
    def _add_indicators_to_df(self, df):
        """Add technical indicators to a single DataFrame."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['20STD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['20MA'] + (df['20STD'] * 2)
        df['Lower_Band'] = df['20MA'] - (df['20STD'] * 2)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_sequences(self, data=None, feature_columns=None, target_column='Close', 
                         sequence_length=10, test_size=0.2):
        """
        Prepare time series sequences for training.
        
        Args:
            data: DataFrame with financial data (uses self.data if None)
            feature_columns: List of columns to use as features (uses all except target if None)
            target_column: Column to predict
            sequence_length: Number of time steps to use for prediction
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if data is None:
            data = self.data
            
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Extract features and target
        features = data[feature_columns].values
        target = data[target_column].values.reshape(-1, 1)
        
        # Scale the data
        features_scaled = self.scaler_X.fit_transform(features)
        target_scaled = self.scaler_y.fit_transform(target)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - sequence_length):
            X.append(features_scaled[i:i+sequence_length])
            y.append(target_scaled[i+sequence_length])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_y(self, y_scaled):
        """
        Convert scaled predictions back to original scale.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Original scale values
        """
        return self.scaler_y.inverse_transform(y_scaled)
    
    def plot_stock_data(self, ticker=None, data=None):
        """
        Plot stock data with technical indicators.
        
        Args:
            ticker: Stock ticker to plot (if data is a dictionary)
            data: DataFrame to plot (uses self.data if None)
        """
        if data is None:
            data = self.data
            
        if isinstance(data, dict):
            if ticker is None:
                ticker = list(data.keys())[0]
            df = data[ticker]
        else:
            df = data
            ticker = "Stock"
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and moving averages
        ax1.plot(df.index, df['Close'], label='Close Price')
        if 'MA5' in df.columns:
            ax1.plot(df.index, df['MA5'], label='5-day MA')
        if 'MA20' in df.columns:
            ax1.plot(df.index, df['MA20'], label='20-day MA')
        if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
            ax1.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], alpha=0.2, color='gray')
            
        ax1.set_title(f'{ticker} Stock Price')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot RSI
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], color='purple', label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        
        # Get output directory from environment or use current directory
        output_dir = os.environ.get('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        plt.savefig(os.path.join(output_dir, f'{ticker}_analysis.png'))
        plt.close()
        
    def plot_predictions(self, actual, predicted, title="Stock Price Prediction"):
        """
        Plot actual vs predicted values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Get output directory from environment or use current directory
        output_dir = os.environ.get('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        plt.savefig(os.path.join(output_dir, 'prediction_results.png'))
        plt.close()

# Example usage
if __name__ == "__main__":
    processor = FinancialDataProcessor()
    
    # Fetch data for a single stock
    data = processor.fetch_stock_data("AAPL", period="1y")
    print(f"Fetched data shape: {data.shape}")
    
    # Add technical indicators
    data_with_indicators = processor.add_technical_indicators()
    print(f"Columns after adding indicators: {data_with_indicators.columns.tolist()}")
    
    # Plot the data
    processor.plot_stock_data()
    
    # Prepare sequences for ML
    X_train, X_test, y_train, y_test = processor.prepare_sequences(
        data_with_indicators, 
        sequence_length=10
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
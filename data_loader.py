import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os

def get_data(tickers, start='2009-01-01', end='2025-01-01'):
    """
    TO_DO: Download stock data for given tickers
    Return a dictionary of ticker -> dataframe 
    """
    stock_data = {}
    for ticker in tickers: 
        df = yf.download(ticker, start=start, end=end) 
        stock_data[ticker] = df
    return stock_data

def save_data_to_csv(stock_data, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    for ticker, df in stock_data.items():  
        df.to_csv(f'{data_dir}/{ticker}.csv')


def load_data_from_csv(tickers, data_dir='data'):
    stock_data = {}
    for ticker in tickers:
        try:
            df = pd.read_csv(f'{data_dir}/{ticker}.csv', skiprows=2, index_col='Date', parse_dates=True)
            # Rename columns to match training data format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            stock_data[ticker] = df
        except Exception as e:
            print(f"Failed to load {ticker}: {str(e)[:50]}")
    return stock_data


def split_data(stock_data, training_range=('2009-01-01', '2019-12-31'), 
                validation_range=('2020-01-01', '2020-12-31'),
                test_range=('2021-01-01', '2025-01-01')):

    training_data = {}
    validation_data = {}
    test_data = {}

    """
    TO_DO: Split stock data into training, validation, and test sets
    Returns: training_data, validation_data, test_data (all dicts of ticker -> dataframe)
    """
    
    for ticker, df in stock_data.items():
        training_data[ticker] = df.loc[training_range[0]:training_range[1]]
        validation_data[ticker] = df.loc[validation_range[0]:validation_range[1]]
        test_data[ticker] = df.loc[test_range[0]:test_range[1]]

    return # TO_DO


def process_data_with_indicators(stock_data):
    """
    Returns: dict of ticker -> dataframe with technical indicators
    """
    processed_data = {}
    # TO_DO: Add technical indicators to all stocks in stock_data
    return processed_data



def load_processed_data(data_dir='data'):
    """
    Load processed datasets from pickle files
    
    Args:
        data_dir: directory containing pickle files (default: 'data')
    
    Returns:
        training_data, validation_data, test_data (all dicts of ticker -> dataframe)
    """
    with open(f'{data_dir}/training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)
    with open(f'{data_dir}/validation_data.pkl', 'rb') as f:
        validation_data = pickle.load(f)
    with open(f'{data_dir}/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    return training_data, validation_data, test_data

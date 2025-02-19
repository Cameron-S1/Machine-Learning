from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import plotly.graph_objects as go
import random
import json
import logging
import signal
import sys
import os
import threading
import time
import pandas as pd
import numpy as np
import krakenex
import openai
import requests
from dash import Dash, html, dcc
from dash.dependencies import Input as DashInput, Output as DashOutput, State as DashState
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import uuid  # For unique strategy IDs
import concurrent.futures
import re  # At the top of your file, ensure you import the 're' module for regular expressions.
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import sqlite3
import dash
from dash import html, dcc
from datetime import datetime
import time
import websockets
import asyncio
import json
import logging

StrategyResult = namedtuple('StrategyResult', ['data', 'metrics', 'trades', 'id', 'strategy'])
# Configuration
MIN_RECORDS = 10  # Reduced for testing purposes
DATA_WINDOW_SIZE = 100  # Maximum number of records to keep

# Initialize a deque to store incoming data
data_window = deque(maxlen=DATA_WINDOW_SIZE)
# ============================
# Configuration and Setup
# ============================

# Fetch your API keys 
OPENAI_API_KEY = ''
if not OPENAI_API_KEY:
    print("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    sys.exit(1)
KRKN_API_KEY = ''  # Replace with your Kraken API Key 
KRKN_SECRET = ''  # Replace with your Kraken Secret Key
if not KRKN_API_KEY or not KRKN_SECRET:
    print("Kraken API credentials not found. Please set 'KRKN_API_KEY' and 'KRKN_SECRET' environment variables.")
    sys.exit(1)

# Initialize Kraken API client
api = krakenex.API(key=KRKN_API_KEY, secret=KRKN_SECRET)

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(
    filename='bot.log',
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format='%(asctime)s %(levelname)s:%(message)s'
)


# Threading lock for safe data access
data_lock = threading.Lock()

# Event to signal threads to stop
stop_event = threading.Event()

# Latest RSI value
latest_rsi_value = None

# File paths for saving/loading parameters and logs
PARAMS_FILE = 'strategy_params.json'
TRADES_HISTORY_PAPER = 'trades_history_paper.csv'
PERFORMANCE_METRICS_PAPER = 'performance_metrics_paper.json'
TRADES_HISTORY_BACKTEST = 'trades_history_backtest.csv'
PERFORMANCE_METRICS_BACKTEST = 'performance_metrics_backtest.json'

# Define performance thresholds
PERFORMANCE_THRESHOLDS = {
    'Net PnL': 0,            # Strategies must have a positive Net PnL
    'Sharpe Ratio': 1,      # Minimum Sharpe Ratio
    'Profit Factor': 1      # Minimum Profit Factor
}

# Initialize strategy parameters (will be loaded from file if exists)
strategy_params = {
    'rsi_strategy': {
        'periods': 14,
        'overbought': 70,
        'oversold': 30
    },
    'ml_strategy': {
        'model_params': {
            'C': 1.0,
            'epsilon': 0.1
        }
    },
    'best_evolved_strategies': []  # Initialize as empty list
}

# Initialize conversation history
conversation_history = []

# ============================
# Global Model Instances
# ============================

# Initialize global models to avoid repeated building and retracing
autoencoder_model = None
pca_model = None
kmeans_model = None
anomaly_detector = None
scaler = MinMaxScaler()
population = []
def initialize_models(data, encoding_dim=2, n_clusters=3, contamination=0.05):
    """
    Initializes and fits global models: Autoencoder, PCA, KMeans, and IsolationForest.

    Parameters:
        data (pd.DataFrame): DataFrame containing features.
        encoding_dim (int): Dimensionality of the encoding for Autoencoder.
        n_clusters (int): Number of clusters for KMeans.
        contamination (float): Proportion of outliers for IsolationForest.
    """
    global autoencoder_model, pca_model, kmeans_model, anomaly_detector, scaler

    logging.info("Initializing and fitting global models.")

    # Feature Engineering
    features = data[['return', 'volatility', 'momentum']].dropna()
    scaled_features = scaler.fit_transform(features)

    # Initialize and fit PCA
    pca_model = PCA(n_components=2)
    principal_components = pca_model.fit_transform(scaled_features)
    data['pc1'] = principal_components[:, 0]
    data['pc2'] = principal_components[:, 1]
    logging.info("PCA model initialized and fitted.")

    # Initialize and fit Autoencoder
    input_dim = scaled_features.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder_model = Model(inputs=input_layer, outputs=decoded)
    autoencoder_model.compile(optimizer='adam', loss='mse')
    autoencoder_model.fit(scaled_features, scaled_features, epochs=50, batch_size=32, verbose=0)
    logging.info("Autoencoder model initialized and fitted.")

    # Initialize and fit KMeans
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    regimes = kmeans_model.fit_predict(scaled_features)
    data['market_regime'] = regimes
    logging.info("KMeans model initialized and fitted.")

    # Initialize and fit IsolationForest
    anomaly_detector = IsolationForest(contamination=contamination, random_state=42)
    anomalies = anomaly_detector.fit_predict(scaled_features)
    data['anomaly'] = anomalies
    logging.info("IsolationForest model initialized and fitted.")

    logging.info("Global models initialization completed.")

# ============================
# Data Structures
# ============================

# Global variables for backtesting
data_backtest = pd.DataFrame()
portfolio_backtest = pd.DataFrame()
metrics_backtest = {}
accuracy_metrics_backtest = {}
trades_log_backtest = pd.DataFrame()
aggregated_sentiment_backtest = 0  # Initialize sentiment score
selected_strategies = []  # Initialize as an empty list

# Initialize a DataFrame to store incoming data
data_paper = pd.DataFrame(columns=['timestamp', 'close'])

import websocket

def on_message(ws, message):
    global data_paper
    try:
        # Decode the JSON message
        data = json.loads(message)
        logging.debug(f"Received WebSocket message: {data}")
        
        # Handle subscription status messages
        if isinstance(data, dict):
            if data.get('event') == 'subscriptionStatus':
                if data.get('status') == 'subscribed':
                    logging.info(f"Successfully subscribed to {data.get('pair')}.")
                else:
                    logging.error(f"Subscription failed: {data}")
                return  # No further processing needed for subscriptionStatus messages

        # Process the live data
        new_data = process_live_data(data)

        # Update the data_paper DataFrame in a thread-safe manner
        with data_lock:
            if not new_data.empty:
                data_paper = pd.concat([data_paper, new_data]).drop_duplicates().reset_index(drop=True)
                logging.debug(f"data_paper updated with new data. Total records: {len(data_paper)}")
            else:
                logging.warning("Received empty data after processing.")

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in on_message: {e}")

def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket connection closed.")

def on_open(ws):
    logging.info("WebSocket connection opened.")
    # Subscribe to the ticker feed for SOL/USD
    subscribe_message = {
        "event": "subscribe",
        "pair": ["SOL/USD"],
        "subscription": {"name": "ticker"}
    }
    ws.send(json.dumps(subscribe_message))
    logging.info("Subscribed to SOL/USD ticker feed.")

def start_live_data_stream():
    """
    Initializes the live data stream by connecting to the WebSocket server.
    
    Returns:
        None
    """
    uri = "wss://ws.kraken.com/"  # Replace with your actual WebSocket URL
    try:
        listen_to_websocket(uri)
    except Exception as e:
        logging.error(f"Exception in start_live_data_stream: {e}")

def listen_to_websocket(uri):
    """
    Connects to the WebSocket server and listens for incoming messages.
    
    Parameters:
        uri (str): The WebSocket server URI.
    
    Returns:
        None
    """
    try:
        ws = websocket.WebSocketApp(uri,
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        ws.run_forever()
    except Exception as e:
        logging.error(f"Failed to listen to WebSocket: {e}")

# Start the WebSocket in a separate daemon thread
data_stream_thread = threading.Thread(target=start_live_data_stream, daemon=True)
data_stream_thread.start()
logging.info("Live data stream started.")
portfolio_paper = pd.DataFrame()

from queue import Queue

# Initialize order queue
order_queue = Queue()

def order_execution_worker():
    global portfolio_paper
    while not stop_event.is_set():
        if not order_queue.empty():
            order = order_queue.get()
            execute_order(order)
        time.sleep(0.1)  # Prevent tight loop

def execute_order(order):
    global portfolio_paper
    trade_action = order['trade_action']
    trade_price = order['trade_price']
    trade_size = order['trade_size']
    timestamp = order['trade_time']
    order_type = order.get('order_type', 'market')  # Default to 'market' order

    # Simulate latency
    simulated_latency = random.uniform(0.1, 0.5)  # 100ms to 500ms
    time.sleep(simulated_latency)

    # Simulate slippage based on order type
    if order_type == 'market':
        slippage_rate = random.uniform(-0.005, 0.005)  # +/-0.5%
    elif order_type == 'limit':
        slippage_rate = 0  # No slippage for limit orders
    else:
        slippage_rate = random.uniform(-0.005, 0.005)  # Default behavior

    slippage = trade_price * slippage_rate
    executed_price = trade_price + slippage

    # Simulate partial fill for market orders
    if order_type == 'market':
        fill_probability = random.uniform(0.9, 1.0)  # 90% to 100% fill
    else:
        fill_probability = 1.0  # Full fill for limit orders

    executed_size = trade_size * fill_probability

    # **Assign 'strategy_returns' if it's a closing trade**
    if trade_action in ['Sell', 'Buy']:
        # Find the corresponding opening trade to calculate returns
        opposite_action = 'Buy' if trade_action == 'Sell' else 'Sell'
        corresponding_trades = portfolio_paper[portfolio_paper['trade_action'] == opposite_action]
        if not corresponding_trades.empty:
            # Get the latest corresponding trade
            corresponding_trade = corresponding_trades.iloc[-1]
            if corresponding_trade['strategy_returns'] == 0:
                if trade_action == 'Sell':
                    # Long position return
                    trade_return = (executed_price - corresponding_trade['trade_price']) / corresponding_trade['trade_price']
                else:
                    # Short position return
                    trade_return = (corresponding_trade['trade_price'] - executed_price) / corresponding_trade['trade_price']
                order['strategy_returns'] = trade_return
            else:
                order['strategy_returns'] = 0  # Already calculated
        else:
            order['strategy_returns'] = 0  # No corresponding trade found

    # Create a new trade DataFrame
    new_trade = pd.DataFrame([{
        'timestamp': timestamp,
        'action': trade_action,
        'price': executed_price,
        'size': executed_size,
        'slippage': slippage,
        'fill_rate': fill_probability,
        'order_type': order_type,
        'strategy_returns': order['strategy_returns']
    }])

    # Concatenate the new trade to the portfolio_paper DataFrame
    portfolio_paper = pd.concat([portfolio_paper, new_trade], ignore_index=True)

    logging.info(
        f"Executed {trade_action} {order_type} order at ${executed_price:.2f} "
        f"with size {executed_size:.4f} on {timestamp} "
        f"(Slippage: {slippage:.2f}, Fill Rate: {fill_probability:.2%}) after {simulated_latency:.3f}s latency"
    )



# Start the order execution worker in a separate daemon thread
execution_thread = threading.Thread(target=order_execution_worker, daemon=True)
execution_thread.start()
logging.info("Order execution worker started.")
def save_generation_state(generation_number, strategies):
    """
    Saves the current generation's strategies to a JSON file.

    Parameters:
        generation_number (int): The current generation number.
        strategies (list): A list of strategy dictionaries.
    """
    filename = f'generation_{generation_number}_strategies.json'
    try:
        with open(filename, 'w') as file:
            json.dump(strategies, file, indent=4)
        logging.info(f"Generation {generation_number} strategies saved to {filename}.")
    except Exception as e:
        logging.error(f"Failed to save generation {generation_number} strategies: {e}")
metrics_paper = {}
accuracy_metrics_paper = {}
trades_log_paper = pd.DataFrame()
aggregated_sentiment_paper = 0  # Initialize sentiment score

# Global mode variable
app_mode = ''  # 'backtest' or 'paper'

# Global variable for selected strategy
selected_strategy = 'ml_strategy'  # Default strategy

# ============================
# Utility Functions
# ============================
def calculate_rsi(close_prices, period):
    """
    Calculates the Relative Strength Index (RSI) for the given close prices.

    Parameters:
        close_prices (pd.Series): Series of close prices.
        period (int): Number of periods for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.fillna(0)  # Replace NaN with 0 for initial periods
    logging.debug(f"RSI calculated. Length: {len(rsi)}")
    
    return rsi


def preprocess_data(data):
    """
    Preprocesses raw OHLC data by engineering features, applying PCA, clustering, anomaly detection.

    Parameters:
        data (pd.DataFrame): Raw OHLC data.

    Returns:
        pd.DataFrame: Preprocessed data ready for strategy execution.
    """
    data = data.copy()

    # Feature Engineering
    data['return'] = data['close'].pct_change()
    data['volatility'] = data['return'].rolling(window=10).std()
    data['momentum'] = data['close'] - data['close'].shift(10)
    data = data.dropna()

    # Apply initialized models
    if autoencoder_model is None or pca_model is None or kmeans_model is None or anomaly_detector is None:
        initialize_models(data)

    # Apply PCA (already fitted globally)
    features = data[['return', 'volatility', 'momentum']].dropna()
    if features.empty:
        logging.warning("Preprocessed features are empty. Skipping PCA and further processing.")
        return pd.DataFrame()  # Return empty DataFrame to handle downstream

    scaled_features = scaler.transform(features)
    principal_components = pca_model.transform(scaled_features)
    data.loc[features.index, 'pc1'] = principal_components[:, 0]
    data.loc[features.index, 'pc2'] = principal_components[:, 1]

    # Apply Autoencoder for feature extraction
    encoded_features = autoencoder_model.predict(scaled_features)
    for i in range(encoded_features.shape[1]):
        data.loc[features.index, f'ae_{i+1}'] = encoded_features[:, i]

    # Clustering Market Regimes (already fitted globally)
    regimes = kmeans_model.predict(scaled_features)
    data.loc[features.index, 'market_regime'] = regimes

    # Anomaly Detection (already fitted globally)
    anomalies = anomaly_detector.predict(scaled_features)
    data.loc[features.index, 'anomaly'] = anomalies

    # Exclude Anomalies
    data = data[data['anomaly'] == 1]  # Adjust based on IsolationForest's labeling

    return data


import random

def generate_random_strategy():
    """
    Generates a random trading strategy with random parameters.
    
    Returns:
        dict: A dictionary representing the strategy.
    """
    strategy = {
        'id': str(uuid.uuid4()),  # Unique identifier for the strategy
        'indicators': {
            'rsi_period': random.randint(10, 20),          # RSI period between 10 and 20
            'rsi_overbought': random.randint(60, 80),      # RSI overbought threshold between 60 and 80
            'rsi_oversold': random.randint(20, 40),        # RSI oversold threshold between 20 and 40
            'macd_fast_period': random.randint(5, 15),     # MACD fast period between 5 and 15
            'macd_slow_period': random.randint(20, 40),    # MACD slow period between 20 and 40
            'macd_signal_period': random.randint(5, 15)     # MACD signal period between 5 and 15
        },
        'weights': {
            'rsi_weight': round(random.uniform(0.5, 1.5), 2),    # Weight for RSI between 0.5 and 1.5
            'macd_weight': round(random.uniform(0.5, 1.5), 2)    # Weight for MACD between 0.5 and 1.5
        },
        'preferred_regime': random.randint(0, 2)  # Preferred market regime (e.g., 0, 1, or 2)
    }
    return strategy
def evolve_strategies_by_regime(data_preprocessed, population, top_n=10, mutation_rate=0.1):
    """
    Evolves strategies based on the current market regime.

    Parameters:
        data_preprocessed (pd.DataFrame): The preprocessed market data.
        population (list): Current population of strategies.
        top_n (int): Number of top strategies to select for evolution.
        mutation_rate (float): Mutation rate for evolving strategies.

    Returns:
        list: A list of evolved strategy dictionaries.
    """
    strategy_results = []
    performances = []

    for strategy in population:
        result = run_evolved_strategy(data_preprocessed, strategy)
        strategy_results.append(result)
        performances.append(result.metrics)

    best_strategies = select_best_strategies(strategy_results, performances, top_n=top_n)
    
    # Evolve strategies
    new_population = evolve_strategies(best_strategies, mutation_rate=mutation_rate)

    return new_population


def evaluate_and_swap_strategies(population, performances, thresholds=PERFORMANCE_THRESHOLDS):
    """
    Evaluates strategies against performance thresholds and swaps out poor performers.

    Parameters:
        population (list): Current population of strategies.
        performances (list): Corresponding performance metrics for each strategy.
        thresholds (dict): Performance thresholds.

    Returns:
        list: Updated population with poor-performing strategies replaced.
    """
    new_population = []
    for strategy, metrics in zip(population, performances):
        is_poor = False
        for metric, threshold in thresholds.items():
            if metrics.get(metric, 0) < threshold:
                is_poor = True
                break
        if not is_poor:
            new_population.append(strategy)
        else:
            # Replace with a new random strategy
            new_strategy = generate_random_strategy()
            new_population.append(new_strategy)
            swap_message = f"Strategy {strategy['id']} has been replaced due to underperformance. New Strategy ID: {new_strategy['id']}."
            logging.info(swap_message)
            # Removed conversation history update to prevent unnecessary API calls
    return new_population

def determine_current_market_regime():
    """
    Determines the current market regime based on the latest data.

    Returns:
        int: Current market regime.
    """
    if app_mode == 'paper':
        data = data_paper.copy()
    elif app_mode == 'backtest':
        data = data_backtest.copy()
    else:
        return 0  # Default regime

    if not data.empty and 'market_regime' in data.columns:
        current_regime = int(data['market_regime'].iloc[-1])
        logging.debug(f"Current market regime determined: {current_regime}")
        return current_regime
    else:
        logging.warning("Market regime data unavailable, defaulting to regime 0.")
        return 0  # Default regime

def get_strategy_by_id(strategy_id):
    """
    Retrieves strategy parameters based on strategy ID.

    Parameters:
        strategy_id (str): Unique identifier for the strategy.

    Returns:
        dict: Strategy parameters.
    """
    for strategy in strategy_params.get('best_evolved_strategies', []):
        if strategy['id'] == strategy_id:
            return strategy
    return {}

def visualize_market_regimes(data):
    """
    Visualizes market regimes using PCA components.

    Parameters:
        data (pd.DataFrame): Preprocessed market data with PCA and regime labels.
    """
    if 'pc1' in data.columns and 'pc2' in data.columns and 'market_regime' in data.columns:
        fig = px.scatter(
            data, x='pc1', y='pc2',
            color='market_regime',
            title='Market Regimes Clustering',
            labels={'pc1': 'Principal Component 1', 'pc2': 'Principal Component 2'}
        )
        fig.show()
    else:
        logging.warning("Insufficient data for market regimes visualization.")

def backtest_multiple_strategies(data, strategies):
    results = run_strategy(data, strategies)
    # Aggregate metrics
    aggregated_metrics = {
        'Net PnL': np.mean([metrics['Net PnL'] for _, metrics, _ in results]),
        'Sharpe Ratio': np.mean([metrics['Sharpe Ratio'] for _, metrics, _ in results]),
        # Add other metrics as needed
    }
    return aggregated_metrics, results

def combine_strategy_signals(strategy_results):
    """
    Combines signals from multiple strategies based on their weights and current market regime.

    Parameters:
        strategy_results (list): List of StrategyResult namedtuples.

    Returns:
        pd.Series: Final combined signal.
    """
    combined_signals = pd.DataFrame()

    for result in strategy_results:
        if result is not None and isinstance(result, pd.Series):
            strategy_id = result.name  # Assuming the series is named with strategy ID
            combined_signals[strategy_id] = result
            logging.debug(f"Strategy {strategy_id} signals added to combined_signals.")
        else:
            logging.warning(f"Invalid strategy result: {result}")

    if combined_signals.empty:
        logging.warning("No signals to combine. Returning zero signal.")
        return pd.Series([0]*len(data_paper), index=data_paper.index)

    # Assign weights (using equal weights or predefined strategy weights)
    weights = {col: 1.0 for col in combined_signals.columns}
    logging.debug(f"Signal weights: {weights}")

    # Apply weights to signals
    weighted_signals = combined_signals.mul(pd.Series(weights))
    combined_signal = weighted_signals.sum(axis=1)

    logging.debug(f"Combined Signal sum: {combined_signal.tail()}")

    # Define threshold to generate final signal
    final_signal = combined_signal.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    logging.debug(f"Final combined signal: {final_signal.tail()}")

    return final_signal
def run_evolved_strategy(data, strategy):
    """
    Executes an evolved trading strategy on the provided data.

    Parameters:
        data (pd.DataFrame): Preprocessed market data.
        strategy (dict): Strategy parameters.

    Returns:
        pd.Series: Trading signals generated by the strategy.
    """
    try:
        # Indicator Calculations
        logging.info(f"Running strategy {strategy['id']}")
        data['RSI'] = calculate_rsi(data['close'], strategy['indicators']['rsi_period'])
        data['MACD'], data['MACD_Signal'], _ = calculate_macd(
            data['close'],
            strategy['indicators']['macd_fast_period'],
            strategy['indicators']['macd_slow_period'],
            strategy['indicators']['macd_signal_period']
        )
        
        # Log indicator lengths
        rsi_length = len(data['RSI'])
        macd_length = len(data['MACD'])
        data_length = len(data)
        logging.debug(f"RSI Length: {rsi_length}, MACD Length: {macd_length}, Data Length: {data_length}")
        
        # Handle NaN Values by Forward Filling (alternative to Dropping)
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)  # Replace any remaining NaNs with 0
        
        # Verify indicators align with data
        if len(data['RSI']) != len(data) or len(data['MACD']) != len(data):
            logging.error(f"Shape mismatch after dropping NaNs: RSI {len(data['RSI'])}, MACD {len(data['MACD'])}, Data {len(data)}")
            return pd.Series()
        
        # Combine indicators with weights
        rsi_weight = strategy['weights']['rsi_weight']
        macd_weight = strategy['weights']['macd_weight']
        combined_signal = (data['RSI'] * rsi_weight) + (data['MACD'] * macd_weight)
        
        logging.debug(f"Combined Signal shape: {combined_signal.shape}")
        logging.debug(f"Data shape: {data.shape}")
        
        if combined_signal.shape[0] != data.shape[0]:
            logging.error(f"Shape mismatch: combined_signal shape {combined_signal.shape} vs data shape {data.shape}")
            return pd.Series()
        
        # Generate trading signals
        data['Trading_Signal'] = combined_signal.apply(generate_signal)
        logging.debug(f"Trading Signals generated. Shape: {data['Trading_Signal'].shape}")
        
        return data['Trading_Signal']
        
    except Exception as e:
        logging.error(f"Exception in strategy {strategy['id']}: {e}", exc_info=True)
        return pd.Series()

def generate_signal(data, strategy):
    """
    Generates buy/sell signals based on strategy indicators.

    Parameters:
        data (pd.DataFrame): Preprocessed market data.
        strategy (dict): Strategy parameters.

    Returns:
        pd.Series: Signals where 1 = Buy, -1 = Sell, 0 = Hold.
    """
    # Example: RSI-based signal generation
    rsi_period = strategy['indicators']['rsi_period']
    overbought = strategy['indicators']['rsi_overbought']
    oversold = strategy['indicators']['rsi_oversold']

    data['RSI'] = calculate_rsi(data['close'], rsi_period)

    data['signal'] = 0
    data.loc[data['RSI'] > overbought, 'signal'] = -1  # Sell signal
    data.loc[data['RSI'] < oversold, 'signal'] = 1    # Buy signal

    logging.debug(f"Strategy {strategy['id']} RSI: {data['RSI'].iloc[-5:].to_dict()}")
    logging.debug(f"Strategy {strategy['id']} Signals: {data['signal'].iloc[-5:].to_dict()}")

    return data['signal']


def execute_signals(data, strategy, available_capital, risk_percentage=0.01):
    """
    Executes trades based on generated signals with capital-based sizing and shorting capabilities.

    Parameters:
        data (pd.DataFrame): Market data with signals.
        strategy (dict): Strategy parameters.
        available_capital (float): Total capital available for trading.
        risk_percentage (float): Percentage of capital to risk per trade.

    Returns:
        list: Executed trades.
    """
    trades = []
    current_position = None  # None = No position, 'Long' = Long position, 'Short' = Short position
    fixed_risk = available_capital * risk_percentage  # Capital allocated per trade

    for index, row in data.iterrows():
        signal = row['signal']
        price = row['close']

        if signal == 1 and current_position is None:
            # Execute Buy (Long)
            trade_size = fixed_risk / price
            trade = {
                'transaction_id': str(uuid.uuid4()),
                'trade_time': index,
                'trade_action': 'Buy',
                'trade_price': price,
                'trade_size': trade_size,
                'strategy': strategy['id'],
                'strategy_returns': 0  # To be calculated on Sell
            }
            trades.append(trade)
            current_position = 'Long'
            logging.info(f"Executed Buy - ID: {trade['transaction_id']} at ${price:.2f} with size {trade_size:.4f} on {index}")

        elif signal == -1 and current_position is None:
            # Execute Sell (Short)
            trade_size = fixed_risk / price
            trade = {
                'transaction_id': str(uuid.uuid4()),
                'trade_time': index,
                'trade_action': 'Sell',
                'trade_price': price,
                'trade_size': trade_size,
                'strategy': strategy['id'],
                'strategy_returns': 0  # To be calculated on Buy to Cover
            }
            trades.append(trade)
            current_position = 'Short'
            logging.info(f"Executed Sell - ID: {trade['transaction_id']} at ${price:.2f} with size {trade_size:.4f} on {index}")

        elif signal == -1 and current_position == 'Long':
            # Execute Sell to Close Long Position
            trade_size = next((trade['trade_size'] for trade in reversed(trades) if trade['trade_action'] == 'Buy' and trade['strategy'] == strategy['id']), 0)
            trade_return = (price - trade['trade_price']) / trade['trade_price']
            trade = {
                'transaction_id': str(uuid.uuid4()),
                'trade_time': index,
                'trade_action': 'Sell',
                'trade_price': price,
                'trade_size': trade_size,
                'strategy': strategy['id'],
                'strategy_returns': trade_return
            }
            trades.append(trade)
            current_position = None
            logging.info(f"Executed Sell - ID: {trade['transaction_id']} at ${price:.2f} on {index} with return {trade_return:.2%}")

        elif signal == 1 and current_position == 'Short':
            # Execute Buy to Close Short Position
            trade_size = next((trade['trade_size'] for trade in reversed(trades) if trade['trade_action'] == 'Sell' and trade['strategy'] == strategy['id']), 0)
            trade_return = (trade['trade_price'] - price) / trade['trade_price']
            trade = {
                'transaction_id': str(uuid.uuid4()),
                'trade_time': index,
                'trade_action': 'Buy',
                'trade_price': price,
                'trade_size': trade_size,
                'strategy': strategy['id'],
                'strategy_returns': trade_return
            }
            trades.append(trade)
            current_position = None
            logging.info(f"Executed Buy - ID: {trade['transaction_id']} at ${price:.2f} on {index} with return {trade_return:.2%}")

    logging.debug(f"Total trades executed for strategy {strategy['id']}: {len(trades)}")
    return trades

def calculate_macd(close_prices, fast_period, slow_period, signal_period):
    """
    Calculates the MACD and MACD Signal line.

    Parameters:
        close_prices (pd.Series): Series of close prices.
        fast_period (int): Fast EMA period.
        slow_period (int): Slow EMA period.
        signal_period (int): Signal line period.

    Returns:
        tuple: MACD, MACD_Signal, MACD_Histogram
    """
    macd = close_prices.ewm(span=fast_period, adjust=False).mean() - close_prices.ewm(span=slow_period, adjust=False).mean()
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    logging.debug(f"MACD calculated. Length: {len(macd)}")
    
    return macd, macd_signal, macd_hist
def calculate_performance_metrics(trades):
    """
    Calculates performance metrics based on executed trades.

    Parameters:
        trades (list): List of executed trades.

    Returns:
        dict: Performance metrics.
    """
    if not trades:
        return {
            'Net PnL': 0,
            'Max Drawdown %': 0,
            'Sharpe Ratio': 0,
            'Profit Factor': 0,
            'Win Rate': 0,
            'Total Trades': 0,
            'Profitable Trades': 0
        }

    net_pnl = 0
    profitable_trades = 0
    total_trades = 0
    returns = []

    buy_price = None

    for trade in trades:
        total_trades += 1
        if trade['trade_action'] == 'Buy':
            buy_price = trade['trade_price']
        elif trade['trade_action'] == 'Sell' and buy_price is not None:
            trade_return = (trade['trade_price'] - buy_price) / buy_price
            trade['strategy_returns'] = trade_return
            net_pnl += trade_return
            if trade_return > 0:
                profitable_trades += 1
            returns.append(trade_return)
            buy_price = None

    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0

    # Calculate Sharpe Ratio safely
    if len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    else:
        sharpe_ratio = 0

    # Calculate Profit Factor safely
    gross_profit = sum([r for r in returns if r > 0])
    gross_loss = sum([abs(r) for r in returns if r < 0])
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

    # Calculate Max Drawdown safely
    max_drawdown_pct = min(returns) * 100 if returns else 0

    metrics = {
        'Net PnL': net_pnl,
        'Max Drawdown %': max_drawdown_pct,
        'Sharpe Ratio': sharpe_ratio,
        'Profit Factor': profit_factor,
        'Win Rate': win_rate,
        'Total Trades': total_trades,
        'Profitable Trades': profitable_trades
    }

    logging.debug(f"Performance metrics calculated: {metrics}")
    return metrics

def select_best_strategies(strategy_results, performances, top_n=10):
    combined = list(zip(strategy_results, performances))
    combined_sorted = sorted(combined, key=lambda x: x[1].get('Sharpe Ratio', 0), reverse=True)
    top_combined = combined_sorted[:top_n]
    
    best_strategies = []
    for idx, (result, metrics) in enumerate(top_combined):
        try:
            strategy = result.strategy
            if not isinstance(strategy, dict):
                logging.warning(f"Strategy at index {idx} is not a dictionary. Type: {type(strategy)}")
                continue
            best_strategies.append(strategy)
        except AttributeError:
            logging.error(f"StrategyResult at index {idx} does not have a 'strategy' attribute.")
            logging.error(f"StrategyResult attributes: {dir(result)}")
            continue
    
    return best_strategies

def crossover(parent1, parent2):
    """
    Combines two parent strategies to produce a child strategy.

    Parameters:
        parent1 (dict): The first parent strategy.
        parent2 (dict): The second parent strategy.

    Returns:
        dict: The child strategy resulting from crossover.
    """
    child = {'id': str(uuid.uuid4()), 'indicators': {}, 'weights': {}}
    
    # Crossover indicators
    for key in parent1['indicators']:
        child['indicators'][key] = random.choice([parent1['indicators'][key], parent2['indicators'][key]])
    
    # Crossover weights
    for key in parent1['weights']:
        child['weights'][key] = random.choice([parent1['weights'][key], parent2['weights'][key]])
    
    return child

def mutate_strategy(strategy, mutation_rate):
    """
    Applies random mutations to a strategy's parameters based on the mutation rate.

    Parameters:
        strategy (dict): The strategy to mutate.
        mutation_rate (float): Probability of mutation occurring in a parameter.

    Returns:
        dict: The mutated strategy.
    """
    # Mutate indicators
    for key in strategy['indicators']:
        if random.random() < mutation_rate:
            # Example mutation: Add or subtract a small integer value
            mutation_value = random.randint(-2, 2)
            strategy['indicators'][key] += mutation_value
            logging.debug(
                f"Mutated indicator '{key}' by {mutation_value} to {strategy['indicators'][key]}."
            )

    # Mutate weights
    for key in strategy['weights']:
        if random.random() < mutation_rate:
            # Example mutation: Add a small float value
            mutation_value = random.uniform(-0.1, 0.1)
            strategy['weights'][key] += mutation_value
            # Ensure weights remain within reasonable bounds
            strategy['weights'][key] = max(0.1, min(strategy['weights'][key], 2.0))
            logging.debug(
                f"Mutated weight '{key}' by {mutation_value:.2f} to {strategy['weights'][key]:.2f}."
            )

    return strategy
def process_live_data(message):
    """
    Processes incoming WebSocket messages and extracts relevant trading data.
    
    Parameters:
        message (dict or list): The raw message received from the WebSocket.
    
    Returns:
        pd.DataFrame: Extracted data containing timestamp, close price, and volume.
    """
    try:
        if isinstance(message, list):
            # Expected format for ticker data
            channel_id, data, channel_name, pair = message
            timestamp = datetime.utcnow()
            close_price = float(data['c'][0])
            volume = float(data['v'][1])
            
            extracted_data = {
                'timestamp': timestamp,
                'close': close_price,
                'volume': volume
            }
            logging.debug(f"Extracted Data: {extracted_data}")
            return pd.DataFrame([extracted_data])
        
        elif isinstance(message, dict):
            event = message.get('event')
            if event in ['subscriptionStatus', 'heartbeat', 'systemStatus']:
                # Non-data messages
                logging.debug(f"Non-data event received: {event}")
                return pd.DataFrame()  # Return empty DataFrame
            else:
                logging.warning(f"Unhandled event type: {event}")
                return pd.DataFrame()
        
        else:
            logging.warning(f"Unknown message format: {message}")
            return pd.DataFrame()
    
    except Exception as e:
        logging.error(f"Error processing live data: {e}", exc_info=True)
        return pd.DataFrame()
def is_data_sufficient():
    """
    Checks if the accumulated data meets the minimum required records for ML processing.

    Returns:
        bool: True if sufficient, False otherwise.
    """
    global data_paper
    with data_lock:
        if isinstance(data_paper, deque):
            return len(data_paper) >= MIN_RECORDS
        elif isinstance(data_paper, pd.DataFrame):
            return len(data_paper) >= MIN_RECORDS
    return False

def paper_trading_loop(model, scaler):
    """
    Main loop for paper trading that processes data and executes trades based on ML predictions.

    Parameters:
        model: Trained ML model used for predictions.
        scaler: Fitted MinMaxScaler for data normalization.

    Returns:
        None
    """
    while True:
        if is_data_sufficient():
            try:
                # Convert data_window to DataFrame
                data_paper = pd.DataFrame(data_window).set_index('timestamp')

                logging.info(f"Proceeding with ML processing. Data size: {len(data_paper)}")

                # Feature Engineering
                data_paper = add_technical_indicators(data_paper)

                # Prepare features for ML model
                features = data_paper[['close', 'rsi', 'macd']].values  # Adjust features as needed

                if features.size == 0:
                    logging.warning("Features array is empty. Skipping ML processing.")
                    raise ValueError("Empty features array.")

                # Scale features
                scaled_features = scaler.transform(features)

                # Make predictions
                predictions = model.predict(scaled_features)

                # Execute trades based on predictions
                execute_trades(predictions, data_paper)

            except Exception as e:
                logging.error(f"Exception in paper trading loop: {e}")
        else:
            logging.info(f"Insufficient data: {len(data_window)} records. Waiting for more data.")

        # Wait before next iteration
        time.sleep(5)  # Adjust sleep duration as needed

def add_technical_indicators(data):
    """
    Adds technical indicators to the data for feature engineering.

    Parameters:
        data (pd.DataFrame): DataFrame containing market data.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    # Example: Calculate RSI
    data['delta'] = data['close'].diff()
    gain = data['delta'].clip(lower=0)
    loss = -data['delta'].clip(upper=0)
    roll_up = gain.rolling(window=14, min_periods=1).mean()
    roll_down = loss.rolling(window=14, min_periods=1).mean()
    rs = roll_up / roll_down
    data['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # Example: Calculate MACD
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2

    # Drop intermediate columns
    data.drop(['delta'], axis=1, inplace=True)

    logging.debug(f"Technical indicators added. Data columns now: {data.columns.tolist()}")

    return data

def execute_trades(predictions, data_paper):
    """
    Executes trades based on ML model predictions.

    Parameters:
        predictions (array-like): Predictions from the ML model.
        data_paper (pd.DataFrame): DataFrame containing the latest market data.

    Returns:
        None
    """
    for idx, prediction in enumerate(predictions):
        signal = prediction  # Assuming predictions are already in the desired format (e.g., 1, -1, 0)
        timestamp = data_paper.index[idx]
        close_price = data_paper['close'].iloc[idx]

        if signal == 1:
            # Execute Buy
            trade = {
                'transaction_id': str(uuid.uuid4()),
                'trade_time': timestamp,
                'trade_action': 'Buy',
                'trade_price': close_price,
                'strategy': 'ML_Model',
                'strategy_returns': 0  # To be updated on Sell
            }
            # Append to trade history, execute order, etc.
            logging.info(f"Executed Buy - ID: {trade['transaction_id']} at ${trade['trade_price']:.2f} on {trade['trade_time']}")
            # Implement trade appending logic here

        elif signal == -1:
            # Execute Sell
            trade = {
                'transaction_id': str(uuid.uuid4()),
                'trade_time': timestamp,
                'trade_action': 'Sell',
                'trade_price': close_price,
                'strategy': 'ML_Model',
                'strategy_returns': 0  # To be calculated
            }
            # Append to trade history, execute order, etc.
            logging.info(f"Executed Sell - ID: {trade['transaction_id']} at ${trade['trade_price']:.2f} on {trade['trade_time']}")
            # Implement trade appending logic here

        # If signal == 0, Hold (no action)

def evolve_strategies(best_strategies, mutation_rate=0.1):
    """
    Evolves the current set of best strategies to produce a new generation.

    Parameters:
        best_strategies (list): A list of top-performing strategy dictionaries.
        mutation_rate (float): Probability of mutation occurring in a strategy.

    Returns:
        list: A new list of evolved strategy dictionaries.
    """
    new_generation = []
    population_size = len(best_strategies) * 2  # Example: Double the number of best strategies

    logging.debug(f"Starting to evolve strategies. Population size to achieve: {population_size}")
    logging.debug(f"Number of best strategies provided: {len(best_strategies)}")

    while len(new_generation) < population_size:
        try:
            # Select two parents randomly from the best strategies
            parent1, parent2 = random.sample(best_strategies, 2)
        except ValueError as e:
            logging.error(f"Not enough strategies to sample parents: {e}")
            break  # Exit the loop if not enough parents

        # Ensure parents are dictionaries
        if not isinstance(parent1, dict) or not isinstance(parent2, dict):
            logging.error("Parents must be dictionaries representing strategies.")
            continue  # Skip to next iteration

        logging.debug(f"Selected parents: {parent1['id']}, {parent2['id']}")

        # Crossover: Combine parameters from both parents to create a child strategy
        child = {
            'id': str(uuid.uuid4()),
            'indicators': {},
            'weights': {},
            'preferred_regime': random.choice([parent1['preferred_regime'], parent2['preferred_regime']])
        }

        # Crossover indicators
        for key in parent1['indicators']:
            child['indicators'][key] = random.choice([
                parent1['indicators'][key], 
                parent2['indicators'][key]
            ])

        # Crossover weights
        for key in parent1['weights']:
            child['weights'][key] = random.choice([
                parent1['weights'][key], 
                parent2['weights'][key]
            ])

        # Mutation
        child = mutate_strategy(child, mutation_rate)

        new_generation.append(child)
        logging.debug(f"Generated new strategy: {child['id']}")

    logging.info(f"Evolved new generation with {len(new_generation)} strategies.")
    return new_generation

def mutate(strategy, mutation_rate=0.1):
    """
    Mutates a strategy's parameters based on the mutation rate.

    Parameters:
        strategy (dict): Strategy to mutate.
        mutation_rate (float): Probability of mutation.

    Returns:
        dict: Mutated strategy.
    """
    # Mutate indicators
    for key in strategy['indicators']:
        if random.random() < mutation_rate:
            change = random.choice([-1, 1]) * random.randint(1, 5)
            strategy['indicators'][key] += change
            # Ensure indicators stay within realistic bounds
            if 'period' in key:
                strategy['indicators'][key] = max(1, strategy['indicators'][key])
            elif 'overbought' in key or 'oversold' in key:
                strategy['indicators'][key] = min(max(0, strategy['indicators'][key]), 100)
    # Mutate weights
    for key in strategy['weights']:
        if random.random() < mutation_rate:
            strategy['weights'][key] += random.uniform(-0.1, 0.1)
    # Normalize weights
    total_weight = sum(strategy['weights'].values())
    for key in strategy['weights']:
        strategy['weights'][key] /= total_weight
    # Mutate preferred regime
    if random.random() < mutation_rate:
        strategy['preferred_regime'] = random.randint(0, 2)
    return strategy

def initialize_population(pop_size=500):
    """
    Initializes a population of random strategies.

    Parameters:
        pop_size (int): Number of strategies in the population.

    Returns:
        list: List of strategy dictionaries.
    """
    population = []
    for _ in range(pop_size):
        strategy = generate_random_strategy()
        population.append(strategy)
    return population

def save_strategy_params():
    """
    Saves the current strategy parameters to a JSON file.
    """
    try:
        with open(PARAMS_FILE, 'w') as file:
            json.dump(strategy_params, file, indent=4)
        logging.info(f"Strategy parameters saved to {PARAMS_FILE}.")
    except Exception as e:
        logging.error(f"Failed to save strategy parameters: {e}")

def load_strategy_params():
    """
    Loads strategy parameters from a JSON file if it exists.
    """
    global strategy_params
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r') as file:
                strategy_params = json.load(file)
            logging.info(f"Strategy parameters loaded from {PARAMS_FILE}.")
        except Exception as e:
            logging.error(f"Failed to load strategy parameters: {e}")
    else:
        logging.info(f"No existing strategy parameters found. Using defaults.")

def fetch_ohlc_data(pair='SOLUSD', interval=15, lookback=8640):
    """
    Fetches OHLC data from Kraken API.

    Parameters:
        pair (str): Trading pair.
        interval (int): Time interval in minutes.
        lookback (int): Number of past intervals to fetch.

    Returns:
        pd.DataFrame or None: DataFrame containing OHLC data or None if failed.
    """
    try:
        end_time = int(time.time())
        start_time = end_time - (lookback * interval * 60)
        resp = api.query_public('OHLC', data={
            'pair': pair,
            'interval': interval,
            'since': start_time
        })
        if resp.get('error'):
            logging.error(f"Error fetching data: {resp['error']}")
            return None
        ohlc_data = resp['result'][pair]
        df = pd.DataFrame(ohlc_data, columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df.astype(float)
        logging.info(f"Fetched {len(df)} rows of OHLC data.")
        return df
    except Exception as e:
        logging.error(f"Exception processing data: {e}")
        return None

def rsi_strategy(data, params, current_position):
    """
    Implements an RSI-based trading strategy.

    Parameters:
        data (pd.DataFrame): OHLC data.
        params (dict): Strategy parameters.
        current_position (int): Current position (1 for long, 0 for no position).

    Returns:
        pd.DataFrame: DataFrame with RSI and updated signals.
        int: Updated position.
    """
    periods = params['periods']
    overbought = params['overbought']
    oversold = params['oversold']

    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=periods, min_periods=1).mean()
    avg_loss = loss.rolling(window=periods, min_periods=1).mean()

    rs = avg_gain / avg_loss
    data.loc[:, 'RSI'] = 100 - (100 / (1 + rs))

    # Generate signals
    data['signal_rsi'] = 0
    # Buy signal only if not in a position
    buy_signals = (data['RSI'] < oversold) & (current_position == 0)
    data.loc[buy_signals, 'signal_rsi'] = 1  # Buy signal
    # Sell signal only if in a position
    sell_signals = (data['RSI'] > overbought) & (current_position == 1)
    data.loc[sell_signals, 'signal_rsi'] = -1  # Sell signal

    return data, current_position

def get_bot_state():
    """
    Retrieves the bot's current trading state, including ML model details and reward mechanisms.

    Returns:
        str: A formatted string containing the bot's current trading information.
    """
    try:
        # Get the latest trade and current position
        if app_mode == 'paper':
            trades = trades_log_paper.copy()
            metrics = metrics_paper.copy()
        elif app_mode == 'backtest':
            trades = trades_log_backtest.copy()
            metrics = metrics_backtest.copy()
        else:
            trades = pd.DataFrame()
            metrics = {}

        if not trades.empty:
            last_trade = trades.iloc[-1]
            current_position = 'Long' if last_trade['trade_action'] == 'Buy' else 'Flat'
            trade_time = last_trade['trade_time'].strftime('%Y-%m-%d %H:%M:%S')
            trade_price = f"${last_trade['trade_price']:.2f}"
            bot_state = (
                f"Current Position: {current_position}\n"
                f"Last Trade: {last_trade['trade_action']} at {trade_price} on {trade_time}\n"
            )
        else:
            bot_state = "The bot has not executed any trades yet.\n"

        # Fetch the latest price
        current_price = fetch_current_price()
        if current_price is not None:
            bot_state += f"Latest SOL/USD Price: ${current_price:.2f}\n"
        else:
            bot_state += "Latest SOL/USD Price: Price unavailable\n"

        # Include the latest RSI value
        if latest_rsi_value is not None:
            bot_state += f"Latest RSI Value: {latest_rsi_value:.2f}\n"

        # Include performance metrics if available
        if metrics:
            net_pnl = metrics.get('Net PnL', 0.0)
            sharpe_ratio = metrics.get('Sharpe Ratio', 0.0)
            profit_factor = metrics.get('Profit Factor', 0.0)
            max_drawdown_pct = metrics.get('Max Drawdown %', 0.0)

            bot_state += (
                f"Net PnL: ${net_pnl:.2f}\n"
                f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"Profit Factor: {profit_factor:.2f}\n"
                f"Max Drawdown %: {max_drawdown_pct:.2f}%\n"
            )

            # Include reinforcement metrics
            bot_state += (
                f"Reward Function: Net PnL - Max Drawdown % + (Sharpe Ratio * 100) + (Profit Factor * 50)\n"
                f"Current Reward: {calculate_reward([metrics])[0]:.2f}\n"
            )

        # Include information about multiple strategies
        current_strategies = strategy_params.get('best_evolved_strategies', [])
        bot_state += "The bot is currently evaluating multiple strategies simultaneously to improve learning speeds.\n"
        bot_state += f"Number of Strategies: {len(current_strategies)}\n"

        # List top strategies
        bot_state += "Top Strategies:\n"
        for strategy in current_strategies[:5]:  # Display top 5
            bot_state += f"- Strategy ID: {strategy['id']}\n"

        # Include information about the reward mechanisms
        reward_info = (
            "The bot uses a reward function to adjust its strategies based on performance metrics. "
            "The reward is calculated as follows:\n"
            "Reward = Net PnL - Max Drawdown % + (Sharpe Ratio * 100) + (Profit Factor * 50)\n"
            "This reward function helps the bot to continuously learn and improve its trading strategies through reinforcement learning."
        )
        bot_state += f"\n{reward_info}\n"

        # Include ML model performance
        if app_mode == 'paper':
            model_performance = metrics_paper.get('model_performance', {})
        elif app_mode == 'backtest':
            model_performance = metrics_backtest.get('model_performance', {})
        else:
            model_performance = {}

        if model_performance:
            mse = model_performance.get('MSE', 'N/A')
            r2 = model_performance.get('R2 Score', 'N/A')
            bot_state += (
                f"ML Model Performance:\n"
                f"MSE: {mse}\n"
                f"R2 Score: {r2}\n"
            )

        # Include strategy parameters
        bot_state += "Current Strategy Parameters:\n"
        for strategy_name, params in strategy_params.items():
            bot_state += f"{strategy_name}: {params}\n"

        # Include current market regime
        current_regime = determine_current_market_regime()
        bot_state += f"Current Market Regime: {current_regime}\n"

        # Include latest features or signals
        if app_mode == 'paper' and not data_paper.empty:
            latest_features = data_paper.iloc[-1].to_dict()
            bot_state += f"Latest Data Features:\n{latest_features}\n"
        elif app_mode == 'backtest' and not data_backtest.empty:
            latest_features = data_backtest.iloc[-1].to_dict()
            bot_state += f"Latest Data Features:\n{latest_features}\n"

        return bot_state
    except Exception as e:
        logging.error(f"Error retrieving bot state: {e}")
        return "Unable to retrieve bot state at the moment."

def ml_strategy(data, params):
    """
    Implements an advanced pattern recognition strategy using ML.

    Parameters:
        data (pd.DataFrame): OHLC data.
        params (dict): Strategy parameters.

    Returns:
        pd.DataFrame: DataFrame with ML predictions and signals.
    """
    # Existing feature engineering
    data_ml = data.copy()
    data_ml['return'] = data_ml['close'].pct_change()
    data_ml['volatility'] = data_ml['return'].rolling(window=10).std()
    data_ml['momentum'] = data_ml['close'] - data_ml['close'].shift(10)
    data_ml = data_ml.dropna()

    # Apply pre-initialized models
    if autoencoder_model is None or pca_model is None or kmeans_model is None or anomaly_detector is None:
        initialize_models(data_ml)

    # Apply PCA
    principal_components = pca_model.transform(scaler.transform(data_ml[['return', 'volatility', 'momentum']]))
    data_ml['pc1'] = principal_components[:, 0]
    data_ml['pc2'] = principal_components[:, 1]

    # Apply Autoencoder Feature Extraction
    encoded_features = autoencoder_model.predict(scaler.transform(data_ml[['return', 'volatility', 'momentum']]))
    for i in range(encoded_features.shape[1]):
        data_ml[f'ae_{i+1}'] = encoded_features[:, i]

    # Apply market regime identification
    regimes = kmeans_model.predict(scaler.transform(data_ml[['return', 'volatility', 'momentum']]))
    data_ml['market_regime'] = regimes

    # Apply anomaly detection
    anomalies = anomaly_detector.predict(scaler.transform(data_ml[['return', 'volatility', 'momentum']]))
    data_ml['anomaly'] = anomalies

    # Exclude Anomalies
    data_ml = data_ml[data_ml['anomaly'] == 1]  # Adjust based on IsolationForest's labeling

    # Prepare features and target
    # Include the new features: principal components and market regime
    X = data_ml[['pc1', 'pc2', 'market_regime']]
    y = data_ml['return'].shift(-1).fillna(0)

    # One-hot encode 'market_regime'
    X = pd.get_dummies(X, columns=['market_regime'], prefix='regime')

    # For simplicity, we'll use original data without augmentation
    X_combined = X.copy()
    y_combined = y.copy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, shuffle=False)

    # Scaling
    scaler_ml = MinMaxScaler()
    X_train_scaled = scaler_ml.fit_transform(X_train)
    X_test_scaled = scaler_ml.transform(X_test)

    # Train ML model
    model = SVR(**params.get('model_params', {}))
    model.fit(X_train_scaled, y_train)

    # Predict
    predictions = model.predict(X_test_scaled)
    data_ml.loc[X_test.index, 'prediction'] = predictions

    # Evaluate model performance
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    model_performance = {'MSE': mse, 'R2 Score': r2}

    # Store model performance metrics
    if app_mode == 'paper':
        metrics_paper['model_performance'] = model_performance
    elif app_mode == 'backtest':
        metrics_backtest['model_performance'] = model_performance

    # Generate signals
    data_ml['signal_ml'] = 0
    data_ml.loc[data_ml['prediction'] > 0, 'signal_ml'] = 1  # Buy signal
    data_ml.loc[data_ml['prediction'] < 0, 'signal_ml'] = -1  # Sell signal

    return data_ml

def data_augmentation(X, y):
    """
    Performs data augmentation using OpenAI's ChatCompletion API.

    Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.

    Returns:
        tuple: Augmented features and targets.
    """
    augmented_X = []
    augmented_y = []

    for index, row in X.iterrows():
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant that generates synthetic market data."
                },
                {
                    "role": "user",
                    "content": (
                        "Given the following market data features, generate a plausible synthetic data point as a JSON object.\n\n"
                        f"Features: {row.to_dict()}\n\nSynthetic Features:"
                    )
                }
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
                temperature=0.3
            )
            synthetic_data = response['choices'][0]['message']['content'].strip()
            synthetic_features = parse_synthetic_data(synthetic_data)
            if synthetic_features:
                augmented_X.append(synthetic_features)
                augmented_y.append(y.loc[index])
        except Exception as e:
            logging.error(f"Error during data augmentation: {e}")
            continue
    if augmented_X:
        augmented_X_df = pd.DataFrame(augmented_X)
        augmented_y_series = pd.Series(augmented_y)
        return augmented_X_df, augmented_y_series
    else:
        return pd.DataFrame(), pd.Series()

def parse_synthetic_data(synthetic_data):
    """
    Parses the synthetic data generated by OpenAI into a feature dictionary.

    Parameters:
        synthetic_data (str): The synthetic data as a string.

    Returns:
        dict: Parsed features as a dictionary.
    """
    try:
        # Attempt to parse as JSON
        synthetic_features = json.loads(synthetic_data)
        return synthetic_features
    except json.JSONDecodeError:
        # If not JSON, attempt to parse key-value pairs
        try:
            synthetic_features = {}
            for line in synthetic_data.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    synthetic_features[key.strip().lower()] = float(value.strip())
            return synthetic_features
        except Exception as e:
            logging.error(f"Error parsing synthetic data: {e}")
            return None

def backtest_strategy(data, signal_column='signal', initial_capital=10000.0, trade_cost=0.0006):
    """
    Backtests the strategy on new data and calculates performance metrics.

    Parameters:
        data (pd.DataFrame): OHLC data with signals (new data only).
        signal_column (str): Column name for strategy signals.
        initial_capital (float): Starting capital.
        trade_cost (float): Cost per trade.

    Returns:
        tuple: (data with performance, metrics dict, trades DataFrame)
    """
    data = data.copy()
    data['positions'] = 0
    current_position = 0

    signals = data[signal_column].fillna(0).tolist()
    positions = []
    for signal in signals:
        if signal == 1 and current_position == 0:
            current_position = 1
        elif signal == -1 and current_position == 1:
            current_position = 0
        positions.append(current_position)
    data['positions'] = positions

    data['market_returns'] = data['close'].pct_change().fillna(0)
    data['strategy_returns'] = data['positions'].shift(1).fillna(0) * data['market_returns']

    # Account for trade costs
    data['trade_costs'] = data['positions'].diff().abs() * trade_cost
    data['trade_costs'] = data['trade_costs'].fillna(0)
    data['strategy_returns'] = data['strategy_returns'] - data['trade_costs']

    data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod() * initial_capital

    # Log trades
    data['trade'] = data['positions'].diff()
    trades = data[data['trade'] != 0].copy()
    trades['trade_action'] = trades['trade'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    trades['trade_price'] = trades['close']
    trades['trade_time'] = trades.index
    trades['strategy'] = signal_column

    # Calculate metrics based on new trades
    total_pnl = data['strategy_returns'].sum()
    max_drawdown = (data['cumulative_returns'].cummax() - data['cumulative_returns']).max()
    max_cummax = data['cumulative_returns'].cummax().max()
    max_drawdown_pct = (max_drawdown / max_cummax * 100) if max_cummax != 0 else np.nan
    total_volume = data['positions'].diff().abs().sum()
    sharpe_ratio = (data['strategy_returns'].mean() / data['strategy_returns'].std()) * np.sqrt(252 * (60 / 15)) if data['strategy_returns'].std() != 0 else 0  # Adjusted for 15-minute intervals
    profit_factor = (data['strategy_returns'][data['strategy_returns'] > 0].sum() / -data['strategy_returns'][data['strategy_returns'] < 0].sum()) if data['strategy_returns'][data['strategy_returns'] < 0].sum() != 0 else np.nan

    metrics = {
        'Net PnL': total_pnl,
        'Net PnL %': (total_pnl / initial_capital) * 100,
        'Max Drawdown': max_drawdown,
        'Max Drawdown %': max_drawdown_pct,
        'Total Volume': total_volume,
        'Sharpe Ratio': sharpe_ratio,
        'Profit Factor': profit_factor,
    }

    return data, metrics, trades

def run_strategy(data, strategies, sentiment_score=0):
    """
    Runs multiple strategies on the provided data.

    Parameters:
        data (pd.DataFrame): Preprocessed market data.
        strategies (list): List of strategy dictionaries.
        sentiment_score (int): Aggregated sentiment score.

    Returns:
        list: List of tuples containing strategy data, metrics, trades, and strategy ID.
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all strategies to the executor
        future_to_strategy = {
            executor.submit(run_evolved_strategy, data.copy(), strategy): strategy
            for strategy in strategies
        }

        for future in concurrent.futures.as_completed(future_to_strategy):
            strategy = future_to_strategy[future]
            try:
                result = future.result()
                if result.data is None:
                    logging.info(f"Strategy {result.id} skipped due to insufficient data.")
                    continue
                results.append((result.data, result.metrics, result.trades, result.id))
            except Exception as e:
                logging.error(f"Error running strategy {strategy['id']}: {e}")

    return results

def calculate_accuracy(trades, metrics):
    """
    Calculates accuracy metrics for the strategy.

    Parameters:
        trades (pd.DataFrame): DataFrame containing trade history.
        metrics (dict): Performance metrics.

    Returns:
        dict: Accuracy metrics.
    """
    analysis = {}
    if not trades.empty:
        analysis['Total Trades'] = len(trades)
        if 'strategy_returns' in trades.columns:
            analysis['Profitable Trades'] = len(trades[trades['strategy_returns'] > 0])
            analysis['Win Rate'] = (len(trades[trades['strategy_returns'] > 0]) / len(trades)) * 100 if len(trades) > 0 else 0
            analysis['Average Profit'] = trades['strategy_returns'].mean()
            analysis['Max Profit'] = trades['strategy_returns'].max()
            analysis['Max Loss'] = trades['strategy_returns'].min()
        else:
            analysis['Profitable Trades'] = 0
            analysis['Win Rate'] = 0
            analysis['Average Profit'] = 0
            analysis['Max Profit'] = 0
            analysis['Max Loss'] = 0
    else:
        analysis = {
            'Total Trades': 0,
            'Profitable Trades': 0,
            'Win Rate': 0,
            'Average Profit': 0,
            'Max Profit': 0,
            'Max Loss': 0
        }
    logging.info(f"Trade Analysis: {analysis}")
    return analysis

def load_trade_history(mode='backtest'):
    """
    Loads trade history from a CSV file.

    Parameters:
        mode (str): Mode of operation ('backtest' or 'paper').

    Returns:
        pd.DataFrame: DataFrame containing trade history.
    """
    file_path = TRADES_HISTORY_BACKTEST if mode == 'backtest' else TRADES_HISTORY_PAPER
    if os.path.exists(file_path):
        try:
            trades = pd.read_csv(file_path, parse_dates=['trade_time'])
            logging.info(f"Loaded {len(trades)} trades from {file_path}.")
            return trades
        except Exception as e:
            logging.error(f"Error loading trade history from {file_path}: {e}")
            return pd.DataFrame()
    else:
        logging.info(f"No trade history file found for {mode} mode.")
        return pd.DataFrame()

def analyze_trade_history(trades):
    """
    Analyzes trade history to extract insights.

    Parameters:
        trades (pd.DataFrame): DataFrame containing trade history.

    Returns:
        dict: Analysis results.
    """
    analysis = {}
    if not trades.empty:
        analysis['Total Trades'] = len(trades)
        if 'strategy_returns' in trades.columns:
            analysis['Profitable Trades'] = len(trades[trades['strategy_returns'] > 0])
            analysis['Win Rate'] = (len(trades[trades['strategy_returns'] > 0]) / len(trades)) * 100 if len(trades) > 0 else 0
            analysis['Average Profit'] = trades['strategy_returns'].mean()
            analysis['Max Profit'] = trades['strategy_returns'].max()
            analysis['Max Loss'] = trades['strategy_returns'].min()
        else:
            analysis['Profitable Trades'] = 0
            analysis['Win Rate'] = 0
            analysis['Average Profit'] = 0
            analysis['Max Profit'] = 0
            analysis['Max Loss'] = 0
    else:
        analysis = {
            'Total Trades': 0,
            'Profitable Trades': 0,
            'Win Rate': 0,
            'Average Profit': 0,
            'Max Profit': 0,
            'Max Loss': 0
        }
    logging.info(f"Trade Analysis: {analysis}")
    return analysis

def optimize_parameters_with_history(strategy_name='rsi_strategy', mode='backtest'):
    """
    Optimizes strategy parameters based on historical trade performance.

    Parameters:
        strategy_name (str): Name of the strategy to optimize.
        mode (str): Mode of operation ('backtest' or 'paper').
    """
    trades = load_trade_history(mode=mode)
    analysis = analyze_trade_history(trades)

    if analysis['Win Rate'] == 0:
        logging.warning(f"No profitable trades to inform optimization for {mode} mode.")
        return

    # Define parameter ranges
    if strategy_name == 'rsi_strategy':
        periods_range = range(10, 21)        # Periods from 10 to 20
        overbought_range = range(65, 76)     # Overbought threshold from 65 to 75
        oversold_range = range(25, 36)       # Oversold threshold from 25 to 35
        param_combinations = [
            {'periods': p, 'overbought': ob, 'oversold': os}
            for p in periods_range
            for ob in overbought_range
            for os in oversold_range
        ]
    elif strategy_name == 'ml_strategy':
        c_range = [0.1, 1, 10]
        epsilon_range = [0.01, 0.1, 1]
        param_combinations = [
            {'model_params': {'C': c, 'epsilon': e}}
            for c in c_range
            for e in epsilon_range
        ]
    else:
        logging.error(f"Unknown strategy {strategy_name} for optimization.")
        return

    best_params = strategy_params[strategy_name]
    best_reward = calculate_reward([analysis])[0]  # Assuming one metrics dict

    # Iterate through a subset of possible combinations for optimization
    for temp_params in random.sample(param_combinations, min(100, len(param_combinations))):
        if mode == 'backtest':
            data_copy = data_backtest.copy()
        else:
            data_copy = data_paper.copy()

        if data_copy.empty:
            logging.warning(f"No data available for optimization in {mode} mode.")
            continue

        # Update temporary parameters
        strategy_params[strategy_name] = temp_params

        # Run strategy with temporary parameters
        if strategy_name == 'rsi_strategy':
            data_temp, metrics_temp = rsi_strategy(data_copy.copy(), temp_params, current_position=0)
            # You might need to backtest here similar to run_evolved_strategy
            data_temp, metrics_temp, trades_temp = backtest_strategy(data_temp, 'signal_rsi')
        elif strategy_name == 'ml_strategy':
            data_temp = ml_strategy(data_copy.copy(), temp_params)
            data_temp, metrics_temp, trades_temp = backtest_strategy(data_temp, 'signal_ml')
        else:
            continue

        temp_reward = calculate_reward([metrics_temp])[0]

        if temp_reward > best_reward:
            best_reward = temp_reward
            best_params = temp_params
            logging.info(f"Found better params for {mode} mode: {best_params} with reward {temp_reward}")

    if best_params != strategy_params[strategy_name]:
        strategy_params[strategy_name] = best_params
        logging.info(f"Optimized strategy parameters for {mode} mode: {best_params}")
        save_strategy_params()
    else:
        logging.info(f"No better parameters found during optimization for {mode} mode.")

def optimize_parameters(strategy_name='rsi_strategy', mode='backtest'):
    """
    Adjusts strategy parameters to optimize performance metrics based on historical data.

    Parameters:
        strategy_name (str): Name of the strategy to optimize.
        mode (str): Mode of operation ('backtest' or 'paper').
    """
    # Utilize historical trade analysis for optimization
    optimize_parameters_with_history(strategy_name, mode)

def calculate_reward(metrics_list):
    """
    Calculates the reward based on performance metrics.

    Parameters:
        metrics_list (list): List of performance metrics dictionaries.

    Returns:
        list: List of reward values.
    """
    rewards = []
    for metrics in metrics_list:
        pnl = metrics.get('Net PnL', 0)
        sharpe = metrics.get('Sharpe Ratio', 0)
        drawdown = metrics.get('Max Drawdown %', 0)
        profit_factor = metrics.get('Profit Factor', 0)
        reward = pnl - drawdown + (sharpe * 100) + (profit_factor * 50)
        rewards.append(reward)
    return rewards

def save_trade_history(trades_log, mode='backtest'):
    """
    Saves trade history to a CSV file without duplicating existing trades.

    Parameters:
        trades_log (pd.DataFrame): Trades log.
        mode (str): Mode of operation ('backtest' or 'paper').
    """
    file_path = TRADES_HISTORY_BACKTEST if mode == 'backtest' else TRADES_HISTORY_PAPER
    if not trades_log.empty:
        if os.path.exists(file_path):
            try:
                existing_trades = pd.read_csv(file_path, parse_dates=['trade_time'])
                # Define subset of columns to identify unique trades
                subset_cols = ['trade_time', 'trade_action', 'trade_price', 'strategy']
                # Concatenate and drop duplicates
                combined_trades = pd.concat([existing_trades, trades_log], ignore_index=True)
                combined_trades.drop_duplicates(subset=subset_cols, inplace=True)
                combined_trades.to_csv(file_path, index=False)
                logging.info(f"Trade history saved for {mode} mode. Total trades: {len(combined_trades)}")
            except Exception as e:
                logging.error(f"Error loading or saving trade history from {file_path}: {e}")
        else:
            try:
                trades_log.to_csv(file_path, mode='w', header=True, index=False)
                logging.info(f"Trade history saved for {mode} mode. Total trades: {len(trades_log)}")
            except Exception as e:
                logging.error(f"Error saving trade history to {file_path}: {e}")
    else:
        logging.info(f"No new trades to save for {mode} mode.")

def save_performance_metrics(metrics, mode='backtest'):
    """
    Saves performance metrics to a JSON file after converting numpy types to native Python types.
    
    Parameters:
        metrics (dict): Performance metrics.
        mode (str): Mode of operation ('backtest' or 'paper').
    """
    file_path = PERFORMANCE_METRICS_BACKTEST if mode == 'backtest' else PERFORMANCE_METRICS_PAPER
    
    # Convert numpy types to native Python types
    metrics_native = {}
    for key, value in metrics.items():
        if isinstance(value, np.integer):
            metrics_native[key] = int(value)
        elif isinstance(value, np.floating):
            metrics_native[key] = float(value)
        elif isinstance(value, np.bool_):
            metrics_native[key] = bool(value)
        else:
            metrics_native[key] = value
    
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics_native, f, indent=4)
        logging.info(f"Performance metrics saved for {mode} mode.")
    except Exception as e:
        logging.error(f"Error saving performance metrics to {file_path}: {e}")


# ============================
# OpenAI Integration Functions
# ============================

def fetch_news(query='Solana', max_articles=5):
    """
    Fetches news articles related to the query.

    Parameters:
        query (str): The search query for news.
        max_articles (int): Maximum number of articles to fetch.

    Returns:
        list: List of article texts.
    """
    try:
        # Using NewsAPI for fetching news articles
        # Ensure 'NEWS_API_KEY' is set as an environment variable
        NEWS_API_KEY = os.getenv('NEWS_API_KEY')
        if not NEWS_API_KEY:
            logging.error("News API key not found. Please set the 'NEWS_API_KEY' environment variable.")
            return []

        url = ('https://newsapi.org/v2/everything?'
               f'q={query}&'
               'language=en&'
               'sortBy=publishedAt&'
               f'pageSize={max_articles}&'
               f'apiKey={NEWS_API_KEY}')

        response = requests.get(url)
        data = response.json()

        if data.get('status') != 'ok':
            logging.error(f"News API error: {data.get('message', 'Unknown error')}")
            return []

        articles = []
        for article in data.get('articles', []):
            content = article.get('title', '') + '. ' + article.get('description', '')
            articles.append(content)

        logging.info(f"Fetched {len(articles)} news articles for sentiment analysis.")
        return articles
    except Exception as e:
        logging.error(f"Error fetching news articles: {e}")
        return []

def analyze_sentiment(article_text):
    """
    Analyzes the sentiment of the given text using OpenAI's ChatCompletion API.

    Parameters:
        article_text (str): The text of the article.

    Returns:
        str: Sentiment result ('Positive', 'Negative', 'Neutral').
    """
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that classifies sentiment as Positive, Negative, or Neutral."
            },
            {
                "role": "user",
                "content": (
                    "Analyze the sentiment of the following news article and classify it as Positive, Negative, or Neutral.\n\n"
                    f"Article: {article_text}\n\nSentiment:"
                )
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0
        )
        # Extract the assistant's reply
        sentiment = response['choices'][0]['message']['content'].strip()
        # Ensure sentiment is one of the expected values
        if sentiment not in ['Positive', 'Negative', 'Neutral']:
            sentiment = 'Neutral'
        return sentiment
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return "Neutral"

def aggregate_sentiment(sentiments):
    """
    Aggregates sentiment scores.

    Parameters:
        sentiments (list): List of sentiment strings.

    Returns:
        float: Aggregated sentiment score (-1 to 1).
    """
    sentiment_mapping = {
        'Positive': 1,
        'Negative': -1,
        'Neutral': 0
    }

    scores = [sentiment_mapping.get(sentiment, 0) for sentiment in sentiments]
    if scores:
        aggregated_score = sum(scores) / len(scores)
    else:
        aggregated_score = 0
    return aggregated_score

def get_market_insights():
    """
    Generates market insights using ChatGPT based on the latest price data and sentiment.

    Returns:
        str: Market insights generated by the assistant.
    """
    try:
        # Fetch the latest price
        current_price = fetch_current_price()
        if current_price is None:
            return "Unable to fetch the latest price data."

        # Get aggregated sentiment
        sentiment_score = aggregated_sentiment_paper if app_mode == 'paper' else aggregated_sentiment_backtest

        # Prepare the input for ChatGPT
        messages = [
            {
                "role": "system",
                "content": "You are a market analyst providing insights based on price data and sentiment scores."
            },
            {
                "role": "user",
                "content": (
                    f"The latest SOL/USD price is ${current_price:.2f}.\n"
                    f"The aggregated sentiment score is {sentiment_score:.2f}.\n"
                    "Based on this information, please provide market insights and potential implications for trading."
                )
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        insights = response['choices'][0]['message']['content'].strip()
        return insights
    except Exception as e:
        logging.error(f"Error generating market insights: {e}")
        return "Unable to generate market insights at the moment."

def get_strategy_suggestions():
    """
    Gets strategy suggestions from ChatGPT based on current performance metrics.

    Returns:
        str: Strategy suggestions from the assistant.
    """
    try:
        metrics = metrics_paper if app_mode == 'paper' else metrics_backtest
        if not metrics:
            return "Performance metrics are not available."

        metrics_summary = (
            f"Net PnL: {metrics.get('Net PnL', 'N/A')}\n"
            f"Net PnL %: {metrics.get('Net PnL %', 'N/A')}\n"
            f"Max Drawdown: {metrics.get('Max Drawdown', 'N/A')}\n"
            f"Max Drawdown %: {metrics.get('Max Drawdown %', 'N/A')}\n"
            f"Total Volume: {metrics.get('Total Volume', 'N/A')}\n"
            f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 'N/A')}\n"
            f"Profit Factor: {metrics.get('Profit Factor', 'N/A')}\n"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading expert providing suggestions to improve trading strategies based on performance metrics."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Given the following performance metrics:\n{metrics_summary}\n"
                    "What suggestions do you have to improve the trading strategy?"
                )
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )

        suggestions = response['choices'][0]['message']['content'].strip()
        return suggestions
    except Exception as e:
        logging.error(f"Error getting strategy suggestions: {e}")
        return "Unable to retrieve strategy suggestions at the moment."

def generate_features_from_text(text):
    """
    Generates additional features for ML models based on textual data using ChatGPT.

    Parameters:
        text (str): The input text to generate features from.

    Returns:
        dict: A dictionary of generated features.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an assistant that extracts numerical features from textual market summaries for use in ML models."
            },
            {
                "role": "user",
                "content": (
                    f"Given the following market summary:\n{text}\n\n"
                    "Please extract relevant numerical features as key-value pairs in JSON format for use in an ML model."
                )
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        features_text = response['choices'][0]['message']['content'].strip()
        features = json.loads(features_text)
        return features
    except Exception as e:
        logging.error(f"Error generating features from text: {e}")
        return {}

def retrain_ml_model():
    """
    Retrains the ML model with the latest data.
    """
    global data_paper, strategy_params, app_mode

    logging.info("Retraining ML model with the latest data.")
    try:
        with data_lock:
            if app_mode == 'paper' and not data_paper.empty:
                data = data_paper.copy()
            elif app_mode == 'backtest' and not data_backtest.empty:
                data = data_backtest.copy()
            else:
                logging.warning("No data available for retraining.")
                return

        # Preprocess data
        data_ml = preprocess_data(data)

        # Retrain the ML model using the ml_strategy function
        data_ml = ml_strategy(data_ml, strategy_params['ml_strategy'])

        logging.info("ML model retrained successfully.")
    except Exception as e:
        logging.error(f"Error retraining ML model: {e}")

def parse_user_command(user_message):
    """
    Parses the user message for commands and executes them.

    Parameters:
        user_message (str): The message from the user.

    Returns:
        str or None: Response message after executing the command, or None if no command found.
    """
    global selected_strategy, app_mode, stop_event, strategy_params

    user_message_lower = user_message.lower()

    if "try new strategy" in user_message_lower or "switch strategy" in user_message_lower:
        # Implement strategy switching logic here
        return "Switching to a new strategy is currently not implemented."
    elif "what are you currently doing" in user_message_lower or "current activity" in user_message_lower:
        return "I am currently monitoring the market and executing trading strategies based on the latest signals."
    elif "stop trading" in user_message_lower or "halt operations" in user_message_lower:
        stop_event.set()
        return "Trading operations have been halted."
    elif "start trading" in user_message_lower and app_mode == 'paper':
        # Implement logic to start trading
        return "Trading operations have been started."
    elif "provide market conditions" in user_message_lower or "market conditions" in user_message_lower:
        return get_market_insights()
    elif "set rsi periods to" in user_message_lower:
        # Extract the new period value
        match = re.search(r"set rsi periods to (\d+)", user_message_lower)
        if match:
            new_period = int(match.group(1))
            strategy_params['rsi_strategy']['periods'] = new_period
            save_strategy_params()
            response = f"RSI periods updated to {new_period}."
            return response
        else:
            return "Could not parse the new RSI period value."
    elif "adjust oversold threshold to" in user_message_lower:
        match = re.search(r"adjust oversold threshold to (\d+)", user_message_lower)
        if match:
            new_threshold = int(match.group(1))
            strategy_params['rsi_strategy']['oversold'] = new_threshold
            save_strategy_params()
            response = f"Oversold threshold updated to {new_threshold}."
            return response
        else:
            return "Could not parse the new oversold threshold value."
    elif "retrain the ml model" in user_message_lower:
        # Code to retrain the ML model
        threading.Thread(target=retrain_ml_model, daemon=True).start()
        response = "The ML model is being retrained with the latest data."
        return response
    elif "show me the latest features" in user_message_lower:
        # Provide latest features
        if app_mode == 'paper' and not data_paper.empty:
            latest_features = data_paper.iloc[-1].to_dict()
            response = f"Latest data features: {latest_features}"
            return response
        elif app_mode == 'backtest' and not data_backtest.empty:
            latest_features = data_backtest.iloc[-1].to_dict()
            response = f"Latest data features: {latest_features}"
            return response
        else:
            return "No data available to show."
    elif "explain the current market regime" in user_message_lower:
        current_regime = determine_current_market_regime()
        response = f"The current market regime is {current_regime}."
        return response
    # Add more commands as needed...

    # If no command is recognized, return None
    return None

def chat_with_ml_bot(user_message):
    """
    Sends a message to the ML bot and gets the response using OpenAI's ChatCompletion API.

    Parameters:
        user_message (str): The message from the user.

    Returns:
        str: The response from the ML bot.
    """
    global conversation_history

    try:
        # Include bot's current state
        bot_state = get_bot_state()

        # Check for commands in the user message
        command_response = parse_user_command(user_message)
        if command_response:
            return command_response  # Return the response after executing the command

        # Prepare the messages for the conversation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Jeff, a top-notch coder and crypto trader with a singular focus on making money. "
                    "You have full visibility into the trading system and can interact with all components. "
                    "Provide concise and to-the-point information, emphasizing strategies and insights that maximize profitability. "
                    "Your responses should include any relevant details from the ML models, strategies, and market conditions. "
                    "You can execute commands and make adjustments as needed."
                    f"\n\nCurrent Bot State:\n{bot_state}"
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        # Make the API call with the conversation history
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )

        # Get the assistant's reply
        bot_response = response['choices'][0]['message']['content'].strip()

        # Add the user's message and assistant's reply to the conversation history
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        conversation_history.append({
            "role": "assistant",
            "content": bot_response
        })

        return bot_response
    except Exception as e:
        logging.error(f"Error during chat with ML bot: {e}")
        return "Sorry, I couldn't process your request at the moment."

# ============================
# Dash Application Setup
# ============================

# Initialize Dash app
app = Dash(__name__)
server = app.server  # For deployment

def create_app_layout():
       """
       Creates the layout for the Dash app.
       Returns:
           html.Div: Dash app layout.
       """
       return html.Div([
           html.H1('Trading Slave'),
           
           html.Div(id='live-update-text'),
           html.Button('Save and Quit', id='save-quit-button'),
           html.Div([
               html.H3('Current Strategy Parameters'),
               html.Pre(id='strategy-params-display'),
           ]),
           html.Div(id='current-strategy-display'),
           html.Div(id='current-price-display'),
           html.Div(id='metrics-display'),
           
           # Corrected Trade Signals Graph
           html.H3('Trade Signals'),
           dcc.Graph(id='trade-signals-graph'),
           
           # Corrected Trades Executions Table
           html.H3('Trade Executions'),
           html.Div(id='trades-executions-table'),
           
           # Existing Graphs
           dcc.Graph(id='equity-curve'),
           dcc.Graph(id='price-chart'),
           
           # Chat Interface
           html.H3('Chat with Jeff'),
           html.Div([
               dcc.Input(
                   id='user-input',
                   type='text',
                   placeholder='Type your message here...',
                   style={'width': '80%'}
               ),
               html.Button('Send', id='send-button', n_clicks=0)
           ]),
           html.Div(
               id='chat-output',
               style={
                   'whiteSpace': 'pre-line',
                   'border': '1px solid black',
                   'padding': '10px',
                   'height': '200px',
                   'overflowY': 'scroll'
               }
           ),
           
           # Update Interval
           dcc.Interval(
               id='interval-component',
               interval=1 * 60 * 1000,  # Update every 1 minute
               n_intervals=0
           )
       ])

@app.callback(
    DashOutput('strategy-params-display', 'children'),
    DashInput('interval-component', 'n_intervals')
)
def update_strategy_params_display(n_intervals):
    return json.dumps(strategy_params, indent=4)

def fetch_current_price(pair='SOLUSD'):
    """
    Fetches the current market price from Kraken API.

    Parameters:
        pair (str): Trading pair.

    Returns:
        float or None: Current market price or None if failed.
    """
    try:
        resp = api.query_public('Ticker', data={'pair': pair})
        if resp.get('error'):
            logging.error(f"Error fetching ticker data: {resp['error']}")
            return None
        else:
            # The result is a nested dictionary with the pair as the key
            ticker_data = resp['result'][pair]
            # 'c' is the last trade closed price [price, lot volume]
            current_price = float(ticker_data['c'][0])
            logging.info(f"Fetched current price: {current_price} for {pair}")
            return current_price
    except Exception as e:
        logging.error(f"Exception fetching current price: {e}")
        return None


def stop_threads():
    """
    Sets the stop event to signal all threads to stop.
    """
    stop_event.set()
    logging.info("Threads have been signaled to stop.")

@app.callback(
    DashOutput('save-quit-button', 'children'),
    DashInput('save-quit-button', 'n_clicks'),
    prevent_initial_call=True
)
def save_and_quit(n_clicks):
    """
    Callback to handle the 'Save and Quit' button.

    Parameters:
        n_clicks (int): Number of times the button has been clicked.

    Returns:
        str: Updated button text.
    """
    if n_clicks:
        save_strategy_params()
        stop_threads()
        logging.info("Save and Quit button clicked. Exiting.")
        os._exit(0)  # Force exit the program
    return 'Save and Quit'

@app.callback(
    [
        DashOutput('equity-curve', 'figure'),
        DashOutput('price-chart', 'figure'),
        DashOutput('metrics-display', 'children'),
        DashOutput('current-price-display', 'children'),
        DashOutput('current-strategy-display', 'children'),
        DashOutput('trade-signals-graph', 'figure'),  # Correct: Separate arguments
        DashOutput('trades-executions-table', 'children'),
    ],
    [
        DashInput('interval-component', 'n_intervals'),
    ]
)
def update_dashboard(n_intervals):
    global data_backtest, portfolio_backtest, metrics_backtest, accuracy_metrics_backtest, trades_log_backtest
    global data_paper, portfolio_paper, metrics_paper, accuracy_metrics_paper, trades_log_paper
    global app_mode
    global selected_strategy

    with data_lock:
        if app_mode == 'backtest' and not data_backtest.empty:
            df = portfolio_backtest.copy()
            metrics_to_use = metrics_backtest.copy()
            accuracy_metrics_to_use = accuracy_metrics_backtest.copy()
            trades_to_use = trades_log_backtest.copy()
            data_source = data_backtest.copy()
            title = 'Equity Curve - Backtest'
        elif app_mode == 'paper' and not data_paper.empty:
            df = portfolio_paper.copy()
            metrics_to_use = metrics_paper.copy()
            accuracy_metrics_to_use = accuracy_metrics_paper.copy()
            trades_to_use = trades_log_paper.copy()
            data_source = data_paper.copy()
            title = 'Equity Curve - Paper Trading'
        else:
            df = pd.DataFrame()
            metrics_to_use = {}
            accuracy_metrics_to_use = {}
            trades_to_use = pd.DataFrame()
            data_source = pd.DataFrame()
            title = 'No Data Available'

    # Equity Curve
    if not df.empty and 'cumulative_returns' in df.columns:
        equity_curve = px.line(df, x=df.index, y='cumulative_returns', title=title)
    else:
        equity_curve = go.Figure()

    # Price Chart with Trade Signals
    if not data_source.empty and 'close' in data_source.columns and 'signal' in data_source.columns:
        price_trace = go.Scatter(
            x=data_source.index,
            y=data_source['close'],
            mode='lines',
            name='SOL/USD Price'
        )
        
        # Extract Buy and Sell signals
        buy_signals = data_source[data_source['signal'] == 1]
        sell_signals = data_source[data_source['signal'] == -1]
        
        buy_trace = go.Scatter(
            x=buy_signals.index,
            y=buy_signals['close'],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='Buy Signal'
        )
        
        sell_trace = go.Scatter(
            x=sell_signals.index,
            y=sell_signals['close'],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='Sell Signal'
        )
        
        price_chart = go.Figure(data=[price_trace, buy_trace, sell_trace],
                                layout=go.Layout(title='SOL/USD Price with Trade Signals'))
    else:
        price_chart = go.Figure()

    # Display metrics
    def format_metric(value, suffix=''):
        if isinstance(value, str):
            return value
        elif value is not None and not np.isnan(value):
            return f"{value:.2f}{suffix}"
        else:
            return 'N/A'

    net_pnl = format_metric(metrics_to_use.get('Net PnL'), '$')
    net_pnl_pct = format_metric(metrics_to_use.get('Net PnL %'), '%')
    max_drawdown = format_metric(metrics_to_use.get('Max Drawdown'), '$')
    max_drawdown_pct = format_metric(metrics_to_use.get('Max Drawdown %'), '%')
    sharpe_ratio = format_metric(metrics_to_use.get('Sharpe Ratio'))
    profit_factor = format_metric(metrics_to_use.get('Profit Factor'))
    win_rate = format_metric(accuracy_metrics_to_use.get('Win Rate'), '%')
    total_trades = accuracy_metrics_to_use.get('Total Trades', 'N/A')
    profitable_trades = accuracy_metrics_to_use.get('Profitable Trades', 'N/A')

    metric_display = html.Div([
        html.H3('Performance Metrics'),
        html.P(f"Net PnL: {net_pnl} ({net_pnl_pct})"),
        html.P(f"Max Drawdown: {max_drawdown} ({max_drawdown_pct})"),
        html.P(f"Sharpe Ratio: {sharpe_ratio}"),
        html.P(f"Profit Factor: {profit_factor}"),
        html.H3('Accuracy Metrics'),
        html.P(f"Win Rate: {win_rate}"),
        html.P(f"Total Trades: {total_trades}"),
        html.P(f"Profitable Trades: {profitable_trades}"),
    ])

    # Fetch current market price
    current_price = fetch_current_price()
    if current_price is not None:
        current_price_display = html.Div([
            html.H3('Current SOL/USD Price'),
            html.P(f"${current_price:.2f}"),
        ])
    else:
        current_price_display = html.Div([
            html.H3('Current SOL/USD Price'),
            html.P("Price unavailable"),
        ])

    # Current strategy display
    current_strategy_display = html.Div([
        html.H3('Current Strategy'),
        html.P(f"{selected_strategy}")
    ])

    # Trades log display
    if not trades_to_use.empty:
        trades_table = html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in ['Transaction ID', 'Time', 'Action', 'Price', 'Strategy', 'Returns']])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(row['transaction_id']),
                    html.Td(row['trade_time'].strftime('%Y-%m-%d %H:%M:%S')),
                    html.Td(row['trade_action']),
                    html.Td(f"${row['trade_price']:.2f}"),
                    html.Td(row['strategy']),
                    html.Td(f"{row['strategy_returns']*100:.2f}%" if row['strategy_returns'] else 'N/A'),
                ]) for index, row in trades_to_use.iterrows()
            ])
        ])
    else:
        trades_table = html.P("No trades executed yet.")

    # Trade Signals Distribution Graph
    if not data_source.empty and 'signal' in data_source.columns:
        signals = data_source['signal'].value_counts().to_dict()
        # Map signals to labels
        signals_labels = {1: 'Buy', -1: 'Sell', 0: 'Hold'}
        signals_mapped = {signals_labels.get(k, 'Other'): v for k, v in signals.items()}
        signals_fig = px.pie(
            names=list(signals_mapped.keys()),
            values=list(signals_mapped.values()),
            title='Trade Signals Distribution'
        )
    else:
        signals_fig = px.pie(title='No Trade Signals Available')

    # Trades Executions Table
    if not trades_to_use.empty:
        trades_executions_table = html.Div([
            html.H3('Trades Executions'),
            trades_table
        ])
    else:
        trades_executions_table = html.P("No trades executed yet.")

    # Return all components
    return (
        equity_curve,
        price_chart,
        metric_display,
        current_price_display,
        current_strategy_display,
        signals_fig,                # Trade Signals Distribution
        trades_executions_table    # Trades Executions
    )

@app.callback(
    DashOutput('chat-output', 'children'),
    [
        DashInput('send-button', 'n_clicks')
    ],
    [DashState('user-input', 'value'), DashState('chat-output', 'children')],
    prevent_initial_call=True
)
def update_chat(n_clicks, user_input, existing_chat):
    if n_clicks and user_input:
        # Send the user's message to the ML bot
        response_text = chat_with_ml_bot(user_input)
        # Display the conversation
        existing_chat = existing_chat or ""
        updated_chat = f"{existing_chat}\nYou: {user_input}\nBot: {response_text}"
    else:
        updated_chat = existing_chat or ""
    return updated_chat

@app.callback(
    DashOutput('live-update-text', 'children'),
    DashInput('interval-component', 'n_intervals'),
)
def update_metrics_text(n_intervals):
    """
    Updates the live-update-text div with the current status.

    Parameters:
        n_intervals (int): Number of intervals passed.

    Returns:
        html.P: Status message.
    """
    if app_mode == 'paper':
        return html.P(f'Paper Trading... Updated at {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}')
    elif app_mode == 'live':
        return html.P('Live Trading Mode is not implemented.')
    else:
        return html.P('Backtesting Mode.')


# ============================
# Thread Functions
# ============================
def start_paper_trading(strategies):
    """
    Starts the paper trading loop, executing trades based on combined strategy signals.
    
    Parameters:
        strategies (list): List of strategy dictionaries.
    """
    global data_paper, trades_log_paper, app_mode, stop_event
    global selected_strategies
    global metrics_paper, accuracy_metrics_paper  # Ensure metrics are accessible

    trade_list = []  # Temporary list to collect trades to avoid using append in loop
    current_position = 0  # Initialize current position

    interval_minutes = 1  # Reduced interval for testing
    lookback = 60  # Fetch the last 60 intervals (e.g., 60 minutes)

    try:
        while not stop_event.is_set():
            with data_lock:
                if isinstance(data_paper, deque):
                    current_data = pd.DataFrame(list(data_paper)).set_index('timestamp')
                elif isinstance(data_paper, pd.DataFrame):
                    current_data = data_paper.copy()
                else:
                    logging.error("data_paper has an unsupported type.")
                    current_data = pd.DataFrame()

            if current_data.empty or 'close' not in current_data.columns:
                logging.warning("Current data is empty or missing 'close' column. Skipping this iteration.")
                time.sleep(interval_minutes * 60)
                continue

            if len(current_data) < MIN_RECORDS:
                logging.info(f"Data records ({len(current_data)}) less than MIN_RECORDS ({MIN_RECORDS}). Waiting for more data.")
                time.sleep(interval_minutes * 60)
                continue

            logging.info("Running evolved strategies on current data.")
            strategy_results = []
            for strategy in strategies:
                result = run_evolved_strategy(current_data, strategy)
                if not result.empty:
                    strategy_results.append(result)
                else:
                    logging.warning(f"Strategy {strategy['id']} returned empty results.")
            
            if not strategy_results:
                logging.warning("No valid strategy results to combine. Skipping trade execution.")
                time.sleep(interval_minutes * 60)
                continue

            # Generate final_signal using your strategies
            final_signal = combine_strategy_signals(strategy_results)
            logging.debug(f"Final Signal: {final_signal.iloc[-1]}, Current Position: {current_position}")

            # **Attach the final_signal to data_paper**
            with data_lock:
                if isinstance(final_signal, pd.Series) and not final_signal.empty:
                    data_paper.loc[data_paper.index[-1], 'signal'] = final_signal.iloc[-1]
                    logging.debug(f"Signal {final_signal.iloc[-1]} attached to data_paper.")
                else:
                    logging.warning("final_signal is not a valid pandas Series or is empty.")

            # Execute trade based on final_signal
            if final_signal.iloc[-1] == 1 and current_position == 0:
                # Execute Buy
                trade_action = 'Buy'
                trade_price = current_data['close'].iloc[-1]
                trade_time = current_data.index[-1]
                trade_id = str(uuid.uuid4())
                new_trade = {
                    'transaction_id': trade_id,
                    'trade_time': trade_time,
                    'trade_action': trade_action,
                    'trade_price': trade_price,
                    'strategy': 'Combined Strategy',
                    'strategy_returns': 0  # To be calculated on Sell
                }
                trade_list.append(new_trade)
                logging.info(f"Executed Buy - ID: {trade_id} at ${trade_price:.2f} on {trade_time}")

                current_position = 1  # Update position

            elif final_signal.iloc[-1] == -1 and current_position == 1:
                # Execute Sell
                trade_action = 'Sell'
                trade_price = current_data['close'].iloc[-1]
                trade_time = current_data.index[-1]

                # Find the most recent Buy trade
                buy_trades = trades_log_paper[trades_log_paper['trade_action'] == 'Buy']
                if not buy_trades.empty:
                    previous_buy = buy_trades.iloc[-1]
                    trade_return = (trade_price - previous_buy['trade_price']) / previous_buy['trade_price']
                    logging.info(f"Calculated trade return: {trade_return:.2%}")
                else:
                    trade_return = 0
                    logging.warning("Sell signal received but no corresponding Buy trade found.")

                trade_id = str(uuid.uuid4())
                new_trade = {
                    'transaction_id': trade_id,
                    'trade_time': trade_time,
                    'trade_action': trade_action,
                    'trade_price': trade_price,
                    'strategy': 'Combined Strategy',
                    'strategy_returns': trade_return
                }
                trade_list.append(new_trade)
                logging.info(f"Executed Sell - ID: {trade_id} at ${trade_price:.2f} on {trade_time} with return {trade_return:.2%}")

                current_position = 0  # Update position

            else:
                logging.debug(f"No trade executed. Final signal: {final_signal.iloc[-1]}, Current position: {current_position}")

            # Append new trades to trades_log_paper
            if trade_list:
                new_trades_df = pd.DataFrame(trade_list)
                
                # Convert all numerical types to native Python types before saving
                new_trades_df = new_trades_df.applymap(lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x)
                
                with data_lock:
                    trades_log_paper = pd.concat([trades_log_paper, new_trades_df], ignore_index=True)
                    save_trade_history(trades_log_paper, mode='paper')

                    # Update performance metrics
                    if not trades_log_paper.empty:
                        metrics_paper['Net PnL'] = float(trades_log_paper['strategy_returns'].sum())
                        metrics_paper['Net PnL %'] = (metrics_paper['Net PnL'] / 10000.0) * 100  # Assuming initial capital is 10,000
                        metrics_paper['Win Rate'] = float((trades_log_paper['strategy_returns'] > 0).mean() * 100)
                        metrics_paper['Total Trades'] = int(len(trades_log_paper))
                        metrics_paper['Profitable Trades'] = int((trades_log_paper['strategy_returns'] > 0).sum())
                        # Calculate Sharpe Ratio, Profit Factor, etc., as needed

                    save_performance_metrics(metrics_paper, mode='paper')

                # Clear trade_list after appending
                trade_list = []

            logging.info(f"Total Trades Executed: {len(trades_log_paper)}")

            # Sleep until the next data fetch
            logging.info(f"Sleeping for {interval_minutes} minute(s) before the next iteration.")
            time.sleep(interval_minutes * 60)

    except Exception as e:
        logging.error(f"Exception in paper trading loop: {e}", exc_info=True)
        if isinstance(e, IndexError):
            logging.error("IndexError encountered: Attempted to access a trade that does not exist.")
# ============================
# Main Menu and Execution
# ============================

def main_menu():
    """
    Displays the main menu and handles user input.
    """
    global data_backtest, portfolio_backtest, metrics_backtest, accuracy_metrics_backtest, trades_log_backtest, aggregated_sentiment_backtest
    global data_paper, portfolio_paper, metrics_paper, accuracy_metrics_paper, trades_log_paper, aggregated_sentiment_paper
    global app_mode
    global selected_strategies

    while True:
        print("\nSelect an option:")
        print("1. Evolve and Backtest Strategies")
        print("2. Paper Trade with Best Strategies")
        print("3. Live Trade (Not Implemented)")
        print("4. Exit")
        choice = input("Enter choice (1/2/3/4): ")

        if choice == '1':
            app_mode = 'backtest'
            # Set backtest to the past 90 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)
            interval = 15  # in minutes

            logging.info(f"Starting strategy evolution and backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")

            # Calculate lookback for 90 days of 15-minute intervals
            lookback = 90 * 24 * (60 // interval)  # 90 days * 24 hours/day * 4 intervals/hour = 8640

            # Fetch data
            data = fetch_ohlc_data(interval=interval, lookback=lookback)
            if data is None or data.empty:
                print("Failed to fetch backtest data.")
                logging.error("Failed to fetch backtest data.")
                continue

            # Preprocess data and initialize models
            data_preprocessed = preprocess_data(data)

            # Evolve strategies
            print("Evolving strategies...")
            best_strategies = evolve_strategies_by_regime(data_preprocessed)
            strategy_params['best_evolved_strategies'] = best_strategies
            save_strategy_params()
            selected_strategies = best_strategies

            # Run strategies with sentiment
            strategy_results = run_strategy(data_preprocessed.copy(), selected_strategies)
            # Combine signals
            final_signal = combine_strategy_signals(strategy_results)

            # Backtest the combined strategy
            data_backtest = data_preprocessed.copy()
            data_backtest['signal'] = final_signal
            data_backtest, metrics_backtest, trades_backtest = backtest_strategy(data_backtest, 'signal')

            # Update trades log
            trades_log_backtest = trades_backtest.copy()

            # Save trade history and performance metrics
            save_trade_history(trades_log_backtest, mode='backtest')
            save_performance_metrics(metrics_backtest, mode='backtest')

            logging.info(f"Backtest completed with {len(trades_log_backtest)} trades.")

            # Define app layout and run Dash server
            app.layout = create_app_layout()

            # Start Dash server in a separate thread to prevent blocking
            dash_thread = threading.Thread(target=lambda: app.run_server(debug=False), daemon=True)
            dash_thread.start()
            logging.info("Dash server launched for backtest.")

            print("Backtest completed. Open the dashboard at http://127.0.0.1:8050")
        
        elif choice == '2':
            app_mode = 'paper'
            # Check if evolved strategies are available
            if not strategy_params['best_evolved_strategies']:
                print("No evolved strategies found. Evolving strategies now before starting paper trading.")
                logging.info("Evolving strategies before starting paper trading.")
                # Fetch recent data to evolve strategies
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=30)  # Use last 30 days of data
                interval = 15  # in minutes
                lookback = 30 * 24 * (60 // interval)  # 30 days * 24 hours/day * 4 intervals/hour = 2880
                data = fetch_ohlc_data(interval=interval, lookback=lookback)
                if data is None or data.empty:
                    print("Failed to fetch data for evolving strategies.")
                    logging.error("Failed to fetch data for evolving strategies.")
                    continue
                # Preprocess data and initialize models
                data_preprocessed = preprocess_data(data)
                # Evolve strategies
                population = initialize_population(pop_size=500)
                logging.info(f"Initialized population with {len(population)} strategies.")
                best_strategies = evolve_strategies_by_regime(data_preprocessed, population)
                strategy_params['best_evolved_strategies'] = best_strategies
                save_strategy_params()
                selected_strategies = best_strategies
            else:
                selected_strategies = strategy_params['best_evolved_strategies']

            # Define app layout
            app.layout = create_app_layout()
            # Start Dash server in a separate thread
            dash_thread = threading.Thread(target=lambda: app.run_server(debug=False), daemon=True)
            dash_thread.start()
            logging.info("Dash server launched for paper trading.")
            # Start paper trading in a separate thread
            paper_trading_thread = threading.Thread(target=start_paper_trading, args=(selected_strategies,), daemon=True)
            paper_trading_thread.start()

            print("Paper trading started. Open the dashboard at http://127.0.0.1:8050")
            logging.info("Paper trading started.")

        elif choice == '3':
            print("Live trading functionality is under development.")
            logging.info("Live trading option selected, but not implemented.")

        elif choice == '4':
            print("Exiting program.")
            logging.info("Exiting program as per user request.")
            save_strategy_params()
            stop_threads()
            sys.exit(0)

        else:
            print("Invalid choice. Please try again.")
            logging.warning(f"Invalid menu choice entered: {choice}")


def signal_handler(sig, frame):
    """
    Handles termination signals to gracefully shutdown the bot.

    Parameters:
        sig (int): Signal number.
        frame: Current stack frame.
    """
    print('Signal received, saving parameters and shutting down...')
    logging.info("Termination signal received. Shutting down.")
    save_strategy_params()
    stop_threads()
    sys.exit(0)

# ============================
# Main Execution
# ============================

if __name__ == '__main__':
    # Load strategy parameters when starting the bot
    load_strategy_params()

    # Load and analyze trade history for continuous learning
    # This can be done for both backtest and paper trading modes
    backtest_trades = load_trade_history(mode='backtest')
    paper_trades = load_trade_history(mode='paper')

    backtest_analysis = analyze_trade_history(backtest_trades)
    paper_analysis = analyze_trade_history(paper_trades)

    # You can use these analyses to inform initial parameter settings or other strategies

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination

    # Start the main menu
    main_menu()

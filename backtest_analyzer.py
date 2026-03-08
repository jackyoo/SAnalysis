#!/usr/bin/env python3
"""
Backtesting Script for Enhanced Stock Analyzer
Tests model accuracy against actual historical data
"""

import yfinance as yf
import talib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class BacktestAnalyzer:
    def __init__(self, symbol, lookback_days=30, training_period='1y'):
        self.symbol = symbol.upper()
        self.lookback_days = lookback_days
        self.training_period = training_period
        self.data = None
        self.backtest_results = []
        
    def fetch_extended_data(self):
        """Fetch extended data for backtesting"""
        try:
            ticker = yf.Ticker(self.symbol)
            # Fetch more data to ensure we have enough for training + backtesting
            self.data = ticker.history(period='2y')
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_indicators(self, data):
        """Calculate technical indicators for given data"""
        # Price data (convert to float64 for TA-Lib)
        high = data['High'].astype('float64').values
        low = data['Low'].astype('float64').values
        close = data['Close'].astype('float64').values
        volume = data['Volume'].astype('float64').values
        open_price = data['Open'].astype('float64').values
        
        # Copy data to avoid modifying original
        df = data.copy()
        
        # === MOMENTUM INDICATORS ===
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['RSI_9'] = talib.RSI(close, timeperiod=9)
        df['RSI_21'] = talib.RSI(close, timeperiod=21)
        
        # MACD
        macd, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_histogram
        df['MACD_Slope'] = df['MACD'].diff()
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        df['Stoch_Diff'] = stoch_k - stoch_d
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Momentum and ROC
        df['Momentum'] = talib.MOM(close, timeperiod=10)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['ROC_5'] = talib.ROC(close, timeperiod=5)
        
        # === VOLATILITY INDICATORS ===
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_Ratio'] = df['ATR'] / close
        df['TRANGE'] = talib.TRANGE(high, low, close)
        
        # === TREND INDICATORS ===
        df['SMA_5'] = talib.SMA(close, timeperiod=5)
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['SMA_50'] = talib.SMA(close, timeperiod=50)
        df['EMA_5'] = talib.EMA(close, timeperiod=5)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        # === VOLUME INDICATORS ===
        df['OBV'] = talib.OBV(close, volume)
        df['OBV_EMA'] = talib.EMA(df['OBV'].astype('float64').values, timeperiod=10)
        df['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
        df['Volume_Ratio'] = volume / df['Volume_SMA']
        
        # === PRICE ACTION FEATURES ===
        df['High_Low_Ratio'] = (high - low) / close
        df['Open_Close_Ratio'] = (close - open_price) / open_price
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        df['Price_Change_1d'] = df['Close'].pct_change(1)
        df['Price_Change_2d'] = df['Close'].pct_change(2)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        
        df['Rolling_Volatility_5'] = df['Price_Change_1d'].rolling(5).std()
        df['Rolling_Volatility_10'] = df['Price_Change_1d'].rolling(10).std()
        
        # Support/Resistance
        df['High_5d'] = df['High'].rolling(5).max()
        df['Low_5d'] = df['Low'].rolling(5).min()
        df['Resistance_Distance'] = (df['High_5d'] - close) / close
        df['Support_Distance'] = (close - df['Low_5d']) / close
        
        return df
    
    def create_features(self, data):
        """Create features for modeling"""
        df = data.copy()
        
        # Target variable
        df['Next_Day_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = (df['Next_Day_Return'] > 0).astype(int)
        
        # Time features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Is_Month_End'] = (df.index.day > 25).astype(int)
        
        # Relative position features
        df['Close_vs_SMA5'] = (df['Close'] / df['SMA_5'] - 1)
        df['Close_vs_SMA10'] = (df['Close'] / df['SMA_10'] - 1)
        df['Close_vs_SMA20'] = (df['Close'] / df['SMA_20'] - 1)
        df['Close_vs_SMA50'] = (df['Close'] / df['SMA_50'] - 1)
        df['Close_vs_EMA5'] = (df['Close'] / df['EMA_5'] - 1)
        df['Close_vs_EMA12'] = (df['Close'] / df['EMA_12'] - 1)
        df['Close_vs_EMA26'] = (df['Close'] / df['EMA_26'] - 1)
        df['Close_vs_BB_Upper'] = (df['Close'] - df['BB_Upper']) / df['BB_Upper']
        df['Close_vs_BB_Lower'] = (df['Close'] - df['BB_Lower']) / df['BB_Lower']
        df['Close_vs_BB_Middle'] = (df['Close'] - df['BB_Middle']) / df['BB_Middle']
        df['Close_vs_SAR'] = (df['Close'] - df['SAR']) / df['SAR']
        df['RSI_vs_70'] = df['RSI'] - 70
        df['RSI_vs_30'] = df['RSI'] - 30
        
        # Advanced patterns
        df['Price_vs_OBV'] = (df['Close'].pct_change() * df['OBV'].pct_change()).rolling(5).mean()
        df['MACD_Above_Signal'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(20).quantile(0.2)).astype(int)
        
        # Feature columns
        feature_columns = [
            'RSI', 'RSI_9', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Slope',
            'Stoch_K', 'Stoch_D', 'Stoch_Diff', 'Williams_R', 'Momentum', 'ROC', 'ROC_5',
            'BB_Width', 'BB_Position', 'ATR', 'ATR_Ratio', 'TRANGE',
            'ADX', 'PLUS_DI', 'MINUS_DI', 'OBV', 'Volume_Ratio',
            'High_Low_Ratio', 'Open_Close_Ratio', 'Gap', 'Price_Change_1d', 'Price_Change_2d',
            'Price_Change_5d', 'Rolling_Volatility_5', 'Rolling_Volatility_10',
            'Resistance_Distance', 'Support_Distance', 'Day_of_Week', 'Month', 'Is_Month_End',
            'Close_vs_SMA5', 'Close_vs_SMA10', 'Close_vs_SMA20', 'Close_vs_SMA50',
            'Close_vs_EMA5', 'Close_vs_EMA12', 'Close_vs_EMA26',
            'Close_vs_BB_Upper', 'Close_vs_BB_Lower', 'Close_vs_BB_Middle',
            'Close_vs_SAR', 'RSI_vs_70', 'RSI_vs_30',
            'Price_vs_OBV', 'MACD_Above_Signal', 'RSI_Oversold', 'RSI_Overbought', 'BB_Squeeze'
        ]
        
        return df, feature_columns
    
    def backtest_last_month(self):
        """Backtest the model on the last month of data"""
        if not self.fetch_extended_data():
            return None
            
        # Calculate indicators for full dataset
        data_with_indicators = self.calculate_indicators(self.data)
        data_with_features, feature_columns = self.create_features(data_with_indicators)
        
        # Get the last month's trading days
        end_date = data_with_features.index[-1]
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Find actual trading days in the last month
        backtest_period = data_with_features[data_with_features.index >= start_date].copy()
        
        results = []
        
        print(f"\n{'='*60}")
        print(f"BACKTESTING {self.symbol} - LAST {self.lookback_days} DAYS")
        print(f"{'='*60}")
        print(f"Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Trading Days: {len(backtest_period)}")
        
        # Rolling window backtesting
        for i in range(len(backtest_period) - 1):  # -1 because we need next day's data
            current_date = backtest_period.index[i]
            next_date = backtest_period.index[i + 1]
            
            # Training data: everything up to current date
            train_end_idx = data_with_features.index.get_loc(current_date)
            
            # Use 1 year of training data
            train_start_idx = max(0, train_end_idx - 252)
            
            train_data = data_with_features.iloc[train_start_idx:train_end_idx + 1]
            
            # Clean training data
            train_clean = train_data[feature_columns + ['Target']].dropna()
            
            if len(train_clean) < 100:  # Need sufficient training data
                continue
                
            # Prepare training data
            X_train = train_clean[feature_columns]
            y_train = train_clean['Target']
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train ensemble model
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
            gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
            lr = LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
            
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft'
            )
            
            try:
                ensemble.fit(X_train_scaled, y_train)
                
                # Make prediction for current day (to be validated next day)
                current_features = data_with_features.loc[current_date, feature_columns].values.reshape(1, -1)
                current_features_scaled = scaler.transform(current_features)
                
                prediction_proba = ensemble.predict_proba(current_features_scaled)[0]
                prediction = ensemble.predict(current_features_scaled)[0]
                
                # Get actual result
                actual_return = (data_with_features.loc[next_date, 'Close'] / 
                               data_with_features.loc[current_date, 'Close'] - 1)
                actual_direction = 1 if actual_return > 0 else 0
                
                # Record result
                result = {
                    'date': current_date,
                    'next_date': next_date,
                    'predicted_direction': prediction,
                    'predicted_up_prob': prediction_proba[1],
                    'predicted_down_prob': prediction_proba[0],
                    'confidence': max(prediction_proba),
                    'actual_direction': actual_direction,
                    'actual_return': actual_return,
                    'correct': prediction == actual_direction,
                    'current_price': data_with_features.loc[current_date, 'Close'],
                    'next_price': data_with_features.loc[next_date, 'Close']
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error on {current_date}: {e}")
                continue
        
        return results
    
    def analyze_results(self, results):
        """Analyze backtest results"""
        if not results:
            print("No results to analyze")
            return None
            
        df_results = pd.DataFrame(results)
        
        # Calculate metrics
        total_predictions = len(df_results)
        correct_predictions = df_results['correct'].sum()
        accuracy = correct_predictions / total_predictions
        
        # Analyze by confidence levels
        high_confidence = df_results[df_results['confidence'] > 0.7]
        medium_confidence = df_results[(df_results['confidence'] > 0.5) & (df_results['confidence'] <= 0.7)]
        low_confidence = df_results[df_results['confidence'] <= 0.5]
        
        # Direction-specific accuracy
        up_predictions = df_results[df_results['predicted_direction'] == 1]
        down_predictions = df_results[df_results['predicted_direction'] == 0]
        
        up_accuracy = up_predictions['correct'].mean() if len(up_predictions) > 0 else 0
        down_accuracy = down_predictions['correct'].mean() if len(down_predictions) > 0 else 0
        
        # Trading simulation
        portfolio_value = 1000  # Starting value
        portfolio_history = [portfolio_value]
        
        for _, row in df_results.iterrows():
            if row['predicted_direction'] == 1:  # Predicted UP, buy
                portfolio_value *= (1 + row['actual_return'])
            else:  # Predicted DOWN, sell/short
                portfolio_value *= (1 - row['actual_return'])
            portfolio_history.append(portfolio_value)
        
        # Buy and hold strategy
        buy_hold_return = (df_results['next_price'].iloc[-1] / df_results['current_price'].iloc[0] - 1)
        strategy_return = (portfolio_value / 1000 - 1)
        
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Total Predictions: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.2%}")
        
        print(f"\nACCURACY BY CONFIDENCE LEVEL:")
        if len(high_confidence) > 0:
            print(f"High Confidence (>70%): {len(high_confidence)} predictions, {high_confidence['correct'].mean():.2%} accuracy")
        if len(medium_confidence) > 0:
            print(f"Medium Confidence (50-70%): {len(medium_confidence)} predictions, {medium_confidence['correct'].mean():.2%} accuracy")
        if len(low_confidence) > 0:
            print(f"Low Confidence (<50%): {len(low_confidence)} predictions, {low_confidence['correct'].mean():.2%} accuracy")
        
        print(f"\nDIRECTION-SPECIFIC ACCURACY:")
        print(f"UP Predictions: {len(up_predictions)} predictions, {up_accuracy:.2%} accuracy")
        print(f"DOWN Predictions: {len(down_predictions)} predictions, {down_accuracy:.2%} accuracy")
        
        print(f"\nTRADING SIMULATION:")
        print(f"Strategy Return: {strategy_return:+.2%}")
        print(f"Buy & Hold Return: {buy_hold_return:+.2%}")
        print(f"Excess Return: {strategy_return - buy_hold_return:+.2%}")
        
        print(f"\nDAILY PREDICTIONS BREAKDOWN:")
        print(f"{'Date':<12} {'Pred':<4} {'Actual':<6} {'Correct':<7} {'Conf':<6} {'Return':<8}")
        print(f"{'-'*50}")
        
        for _, row in df_results.tail(10).iterrows():  # Show last 10 predictions
            pred_str = "UP" if row['predicted_direction'] == 1 else "DOWN"
            actual_str = "UP" if row['actual_direction'] == 1 else "DOWN"
            correct_str = "✓" if row['correct'] else "✗"
            
            print(f"{row['date'].strftime('%Y-%m-%d'):<12} {pred_str:<4} {actual_str:<6} {correct_str:<7} {row['confidence']:<6.1%} {row['actual_return']:+6.2%}")
        
        return {
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'results_df': df_results
        }


def main():
    """Main function for backtesting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Enhanced Stock Analyzer')
    parser.add_argument('symbol', help='Stock symbol (e.g., NVDA, AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    
    args = parser.parse_args()
    
    backtester = BacktestAnalyzer(args.symbol, lookback_days=args.days)
    results = backtester.backtest_last_month()
    
    if results:
        analysis = backtester.analyze_results(results)
        if analysis:
            print(f"\n💡 Model achieved {analysis['accuracy']:.1%} accuracy over {args.days} days")
            if analysis['strategy_return'] > analysis['buy_hold_return']:
                print(f"🎯 Strategy outperformed buy & hold by {analysis['strategy_return'] - analysis['buy_hold_return']:+.1%}")
            else:
                print(f"📊 Strategy underperformed buy & hold by {analysis['buy_hold_return'] - analysis['strategy_return']:+.1%}")
    else:
        print("Backtesting failed")


if __name__ == "__main__":
    main()
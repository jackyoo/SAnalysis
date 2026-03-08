#!/usr/bin/env python3
"""
Stock Technical Analysis and Prediction Tool
Uses TA-Lib and yfinance to analyze stocks and predict short-term movements
"""

import yfinance as yf
import talib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class StockAnalyzer:
    def __init__(self, symbol, period='1y'):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
        self.model = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators using TA-Lib"""
        if self.data is None or self.data.empty:
            return False
            
        # Price data (convert to float64 for TA-Lib)
        high = self.data['High'].astype('float64').values
        low = self.data['Low'].astype('float64').values
        close = self.data['Close'].astype('float64').values
        volume = self.data['Volume'].astype('float64').values
        
        # RSI
        self.data['RSI'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.data['BB_Upper'] = bb_upper
        self.data['BB_Middle'] = bb_middle
        self.data['BB_Lower'] = bb_lower
        self.data['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        
        # Moving Averages
        self.data['SMA_20'] = talib.SMA(close, timeperiod=20)
        self.data['SMA_50'] = talib.SMA(close, timeperiod=50)
        self.data['EMA_12'] = talib.EMA(close, timeperiod=12)
        self.data['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        self.data['Stoch_K'] = stoch_k
        self.data['Stoch_D'] = stoch_d
        
        # Williams %R
        self.data['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Average True Range
        self.data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Volume indicators
        self.data['OBV'] = talib.OBV(close, volume)
        
        # Price momentum
        self.data['Momentum'] = talib.MOM(close, timeperiod=10)
        
        # Rate of Change
        self.data['ROC'] = talib.ROC(close, timeperiod=10)
        
        return True
    
    def prepare_features(self):
        """Prepare features for machine learning model"""
        # Create target variable (1 if price goes up next day, 0 if down)
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        
        # Select features
        feature_columns = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Width', 'Stoch_K', 'Stoch_D', 'Williams_R',
            'ATR', 'Momentum', 'ROC'
        ]
        
        # Add relative position features
        self.data['Close_vs_SMA20'] = self.data['Close'] / self.data['SMA_20'] - 1
        self.data['Close_vs_SMA50'] = self.data['Close'] / self.data['SMA_50'] - 1
        self.data['Close_vs_BB_Upper'] = (self.data['Close'] - self.data['BB_Upper']) / self.data['BB_Upper']
        self.data['Close_vs_BB_Lower'] = (self.data['Close'] - self.data['BB_Lower']) / self.data['BB_Lower']
        
        feature_columns.extend([
            'Close_vs_SMA20', 'Close_vs_SMA50', 
            'Close_vs_BB_Upper', 'Close_vs_BB_Lower'
        ])
        
        # Drop rows with NaN values
        self.features = self.data[feature_columns + ['Target']].dropna()
        
        return len(self.features) > 50
    
    def train_model(self):
        """Train Random Forest model for prediction"""
        if self.features is None or len(self.features) < 50:
            return False
            
        X = self.features.drop('Target', axis=1)
        y = self.features['Target']
        
        # Split data (use recent 20% for testing)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5
        )
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def get_current_prediction(self):
        """Get prediction for current market conditions"""
        if self.model is None or self.features is None:
            return None
            
        # Get the latest feature values
        latest_features = self.features.drop('Target', axis=1).iloc[-1:]
        
        # Get prediction probability
        prob = self.model.predict_proba(latest_features)[0]
        
        # Get feature importance
        feature_names = self.features.drop('Target', axis=1).columns
        importance = dict(zip(feature_names, self.model.feature_importances_))
        
        return {
            'probability_up': prob[1],
            'probability_down': prob[0],
            'prediction': 'UP' if prob[1] > 0.5 else 'DOWN',
            'confidence': max(prob),
            'feature_importance': importance
        }
    
    def get_current_indicators(self):
        """Get current technical indicator values"""
        if self.data is None:
            return None
            
        latest = self.data.iloc[-1]
        
        return {
            'RSI': latest['RSI'],
            'MACD': latest['MACD'],
            'MACD_Signal': latest['MACD_Signal'],
            'BB_Position': (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']),
            'Stoch_K': latest['Stoch_K'],
            'Williams_R': latest['Williams_R'],
            'Close_vs_SMA20': latest['Close_vs_SMA20'],
            'Close_vs_SMA50': latest['Close_vs_SMA50']
        }
    
    def analyze(self):
        """Run complete analysis"""
        print(f"Analyzing {self.symbol}...")
        
        if not self.fetch_data():
            return None
            
        if not self.calculate_technical_indicators():
            print("Error calculating technical indicators")
            return None
            
        if not self.prepare_features():
            print("Insufficient data for analysis")
            return None
            
        accuracy = self.train_model()
        if not accuracy:
            print("Error training model")
            return None
            
        prediction = self.get_current_prediction()
        indicators = self.get_current_indicators()
        
        return {
            'symbol': self.symbol,
            'model_accuracy': accuracy,
            'prediction': prediction,
            'indicators': indicators
        }


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Technical Analysis and Prediction')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--period', default='1y', help='Data period (default: 1y)')
    
    args = parser.parse_args()
    
    analyzer = StockAnalyzer(args.symbol, args.period)
    result = analyzer.analyze()
    
    if result is None:
        print("Analysis failed")
        return
    
    # Display results
    print(f"\n{'='*50}")
    print(f"STOCK ANALYSIS: {result['symbol']}")
    print(f"{'='*50}")
    
    print(f"\nModel Accuracy: {result['model_accuracy']:.2%}")
    
    pred = result['prediction']
    print(f"\nPREDICTION:")
    print(f"Direction: {pred['prediction']}")
    print(f"Probability UP: {pred['probability_up']:.2%}")
    print(f"Probability DOWN: {pred['probability_down']:.2%}")
    print(f"Confidence: {pred['confidence']:.2%}")
    
    indicators = result['indicators']
    print(f"\nTECHNICAL INDICATORS:")
    print(f"RSI (14): {indicators['RSI']:.2f}")
    print(f"MACD: {indicators['MACD']:.4f}")
    print(f"MACD Signal: {indicators['MACD_Signal']:.4f}")
    print(f"Bollinger Band Position: {indicators['BB_Position']:.2%}")
    print(f"Stochastic %K: {indicators['Stoch_K']:.2f}")
    print(f"Williams %R: {indicators['Williams_R']:.2f}")
    print(f"Price vs SMA(20): {indicators['Close_vs_SMA20']:+.2%}")
    print(f"Price vs SMA(50): {indicators['Close_vs_SMA50']:+.2%}")
    
    print(f"\nTOP FEATURES (Importance):")
    sorted_features = sorted(pred['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"{feature}: {importance:.3f}")


if __name__ == "__main__":
    main()
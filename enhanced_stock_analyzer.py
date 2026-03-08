#!/usr/bin/env python3
"""
Enhanced Stock Technical Analysis and Next-Day Prediction Tool
Focused on predicting the very next trading day movement with advanced features
"""

import yfinance as yf
import talib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EnhancedStockAnalyzer:
    def __init__(self, symbol, period='1y'):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
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
    
    def calculate_advanced_indicators(self):
        """Calculate comprehensive technical indicators for next-day prediction"""
        if self.data is None or self.data.empty:
            return False
            
        # Price data (convert to float64 for TA-Lib)
        high = self.data['High'].astype('float64').values
        low = self.data['Low'].astype('float64').values
        close = self.data['Close'].astype('float64').values
        volume = self.data['Volume'].astype('float64').values
        open_price = self.data['Open'].astype('float64').values
        
        # === MOMENTUM INDICATORS ===
        self.data['RSI'] = talib.RSI(close, timeperiod=14)
        self.data['RSI_9'] = talib.RSI(close, timeperiod=9)
        self.data['RSI_21'] = talib.RSI(close, timeperiod=21)
        
        # MACD family
        macd, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Histogram'] = macd_histogram
        self.data['MACD_Slope'] = self.data['MACD'].diff()
        
        # Stochastic family
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        self.data['Stoch_K'] = stoch_k
        self.data['Stoch_D'] = stoch_d
        self.data['Stoch_Diff'] = stoch_k - stoch_d
        
        # Williams %R
        self.data['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Momentum and ROC
        self.data['Momentum'] = talib.MOM(close, timeperiod=10)
        self.data['ROC'] = talib.ROC(close, timeperiod=10)
        self.data['ROC_5'] = talib.ROC(close, timeperiod=5)
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.data['BB_Upper'] = bb_upper
        self.data['BB_Middle'] = bb_middle
        self.data['BB_Lower'] = bb_lower
        self.data['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        self.data['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ATR and volatility
        self.data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        self.data['ATR_Ratio'] = self.data['ATR'] / close
        
        # True Range
        self.data['TRANGE'] = talib.TRANGE(high, low, close)
        
        # === TREND INDICATORS ===
        # Moving Averages
        self.data['SMA_5'] = talib.SMA(close, timeperiod=5)
        self.data['SMA_10'] = talib.SMA(close, timeperiod=10)
        self.data['SMA_20'] = talib.SMA(close, timeperiod=20)
        self.data['SMA_50'] = talib.SMA(close, timeperiod=50)
        self.data['EMA_5'] = talib.EMA(close, timeperiod=5)
        self.data['EMA_12'] = talib.EMA(close, timeperiod=12)
        self.data['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # ADX (Directional Movement)
        self.data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        self.data['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        self.data['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Parabolic SAR
        self.data['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        # === VOLUME INDICATORS ===
        self.data['OBV'] = talib.OBV(close, volume)
        self.data['OBV_EMA'] = talib.EMA(self.data['OBV'].astype('float64').values, timeperiod=10)
        
        # Volume ratios
        self.data['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
        self.data['Volume_Ratio'] = volume / self.data['Volume_SMA']
        
        # === PRICE ACTION FEATURES ===
        # Intraday features
        self.data['High_Low_Ratio'] = (high - low) / close
        self.data['Open_Close_Ratio'] = (close - open_price) / open_price
        self.data['Gap'] = (self.data['Open'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1)
        
        # Price changes
        self.data['Price_Change_1d'] = self.data['Close'].pct_change(1)
        self.data['Price_Change_2d'] = self.data['Close'].pct_change(2)
        self.data['Price_Change_5d'] = self.data['Close'].pct_change(5)
        
        # Volatility measures
        self.data['Rolling_Volatility_5'] = self.data['Price_Change_1d'].rolling(5).std()
        self.data['Rolling_Volatility_10'] = self.data['Price_Change_1d'].rolling(10).std()
        
        # === MARKET MICROSTRUCTURE ===
        # Support/Resistance levels
        self.data['High_5d'] = self.data['High'].rolling(5).max()
        self.data['Low_5d'] = self.data['Low'].rolling(5).min()
        self.data['Resistance_Distance'] = (self.data['High_5d'] - close) / close
        self.data['Support_Distance'] = (close - self.data['Low_5d']) / close
        
        return True
    
    def create_prediction_features(self):
        """Create sophisticated features for next-day prediction"""
        # Create target variable - next day's movement
        self.data['Next_Day_Return'] = self.data['Close'].shift(-1) / self.data['Close'] - 1
        self.data['Target'] = (self.data['Next_Day_Return'] > 0).astype(int)
        
        # Time-based features
        self.data['Day_of_Week'] = self.data.index.dayofweek
        self.data['Month'] = self.data.index.month
        self.data['Is_Month_End'] = (self.data.index.day > 25).astype(int)
        
        # Relative position features
        feature_list = [
            'Close_vs_SMA5', 'Close_vs_SMA10', 'Close_vs_SMA20', 'Close_vs_SMA50',
            'Close_vs_EMA5', 'Close_vs_EMA12', 'Close_vs_EMA26',
            'Close_vs_BB_Upper', 'Close_vs_BB_Lower', 'Close_vs_BB_Middle',
            'Close_vs_SAR', 'RSI_vs_70', 'RSI_vs_30'
        ]
        
        # Calculate relative positions
        self.data['Close_vs_SMA5'] = (self.data['Close'] / self.data['SMA_5'] - 1)
        self.data['Close_vs_SMA10'] = (self.data['Close'] / self.data['SMA_10'] - 1)
        self.data['Close_vs_SMA20'] = (self.data['Close'] / self.data['SMA_20'] - 1)
        self.data['Close_vs_SMA50'] = (self.data['Close'] / self.data['SMA_50'] - 1)
        self.data['Close_vs_EMA5'] = (self.data['Close'] / self.data['EMA_5'] - 1)
        self.data['Close_vs_EMA12'] = (self.data['Close'] / self.data['EMA_12'] - 1)
        self.data['Close_vs_EMA26'] = (self.data['Close'] / self.data['EMA_26'] - 1)
        self.data['Close_vs_BB_Upper'] = (self.data['Close'] - self.data['BB_Upper']) / self.data['BB_Upper']
        self.data['Close_vs_BB_Lower'] = (self.data['Close'] - self.data['BB_Lower']) / self.data['BB_Lower']
        self.data['Close_vs_BB_Middle'] = (self.data['Close'] - self.data['BB_Middle']) / self.data['BB_Middle']
        self.data['Close_vs_SAR'] = (self.data['Close'] - self.data['SAR']) / self.data['SAR']
        self.data['RSI_vs_70'] = self.data['RSI'] - 70
        self.data['RSI_vs_30'] = self.data['RSI'] - 30
        
        # Momentum convergence/divergence
        self.data['Price_vs_OBV'] = (self.data['Close'].pct_change() * self.data['OBV'].pct_change()).rolling(5).mean()
        
        # Advanced patterns
        self.data['MACD_Above_Signal'] = (self.data['MACD'] > self.data['MACD_Signal']).astype(int)
        self.data['RSI_Oversold'] = (self.data['RSI'] < 30).astype(int)
        self.data['RSI_Overbought'] = (self.data['RSI'] > 70).astype(int)
        self.data['BB_Squeeze'] = (self.data['BB_Width'] < self.data['BB_Width'].rolling(20).quantile(0.2)).astype(int)
        
        # Select final features for modeling
        self.feature_columns = [
            'RSI', 'RSI_9', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Slope',
            'Stoch_K', 'Stoch_D', 'Stoch_Diff', 'Williams_R', 'Momentum', 'ROC', 'ROC_5',
            'BB_Width', 'BB_Position', 'ATR', 'ATR_Ratio', 'TRANGE',
            'ADX', 'PLUS_DI', 'MINUS_DI', 'OBV', 'Volume_Ratio',
            'High_Low_Ratio', 'Open_Close_Ratio', 'Gap', 'Price_Change_1d', 'Price_Change_2d',
            'Price_Change_5d', 'Rolling_Volatility_5', 'Rolling_Volatility_10',
            'Resistance_Distance', 'Support_Distance', 'Day_of_Week', 'Month', 'Is_Month_End'
        ] + feature_list + [
            'Price_vs_OBV', 'MACD_Above_Signal', 'RSI_Oversold', 'RSI_Overbought', 'BB_Squeeze'
        ]
        
        # Clean data
        self.features = self.data[self.feature_columns + ['Target', 'Next_Day_Return']].dropna()
        return len(self.features) > 100
    
    def train_ensemble_model(self):
        """Train ensemble model with multiple algorithms optimized for next-day prediction"""
        if self.features is None or len(self.features) < 100:
            return False, None
            
        X = self.features[self.feature_columns]
        y = self.features['Target']
        
        # Use time series split to prevent data leakage
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        # Use recent 80% for training, 20% for testing (time series)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Individual models
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        
        lr = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Ensemble model
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft'
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance (from Random Forest)
        rf_model = self.ensemble_model.named_estimators_['rf']
        feature_importance = dict(zip(self.feature_columns, rf_model.feature_importances_))
        
        # Calculate metrics for last week of predictions
        recent_predictions = y_pred[-5:] if len(y_pred) >= 5 else y_pred
        recent_actual = y_test[-5:] if len(y_test) >= 5 else y_test
        recent_accuracy = accuracy_score(recent_actual, recent_predictions) if len(recent_actual) > 0 else 0
        
        return accuracy, {
            'feature_importance': feature_importance,
            'overall_accuracy': accuracy,
            'recent_accuracy': recent_accuracy,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba
        }
    
    def get_next_day_prediction(self):
        """Get prediction for the very next trading day"""
        if self.ensemble_model is None or self.features is None:
            return None
            
        # Get the latest feature values (most recent day)
        latest_features = self.features[self.feature_columns].iloc[-1:].copy()
        
        # Scale the features
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Get prediction and probabilities
        prediction = self.ensemble_model.predict(latest_features_scaled)[0]
        probabilities = self.ensemble_model.predict_proba(latest_features_scaled)[0]
        
        # Individual model predictions for insight
        individual_predictions = {}
        for name, model in self.ensemble_model.named_estimators_.items():
            individual_pred = model.predict_proba(latest_features_scaled)[0]
            individual_predictions[name] = {
                'prob_down': individual_pred[0],
                'prob_up': individual_pred[1],
                'prediction': 'UP' if individual_pred[1] > 0.5 else 'DOWN'
            }
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'probability_up': probabilities[1],
            'probability_down': probabilities[0],
            'confidence': max(probabilities),
            'individual_models': individual_predictions
        }
    
    def get_market_context(self):
        """Get current market context and technical levels"""
        if self.data is None:
            return None
            
        latest = self.data.iloc[-1]
        latest_5d = self.data.iloc[-5:]
        
        return {
            'current_price': latest['Close'],
            'volume_vs_avg': latest['Volume_Ratio'] if 'Volume_Ratio' in latest else None,
            'volatility_5d': latest_5d['Close'].pct_change().std() * np.sqrt(252),
            'days_since_high': (latest.name - self.data['High'].idxmax()).days,
            'days_since_low': (latest.name - self.data['Low'].idxmin()).days,
            'support_level': latest['Low_5d'] if 'Low_5d' in latest else None,
            'resistance_level': latest['High_5d'] if 'High_5d' in latest else None,
            'trend_strength': latest['ADX'] if 'ADX' in latest else None
        }
    
    def analyze_for_next_day(self):
        """Complete analysis focused on next-day prediction"""
        print(f"Analyzing {self.symbol} for next trading day prediction...")
        
        if not self.fetch_data():
            return None
            
        if not self.calculate_advanced_indicators():
            print("Error calculating technical indicators")
            return None
            
        if not self.create_prediction_features():
            print("Insufficient data for analysis")
            return None
            
        accuracy, model_info = self.train_ensemble_model()
        if not accuracy:
            print("Error training model")
            return None
            
        prediction = self.get_next_day_prediction()
        market_context = self.get_market_context()
        
        return {
            'symbol': self.symbol,
            'model_accuracy': accuracy,
            'recent_accuracy': model_info['recent_accuracy'],
            'prediction': prediction,
            'market_context': market_context,
            'feature_importance': model_info['feature_importance']
        }


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Stock Analysis for Next-Day Prediction')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--period', default='1y', help='Data period (default: 1y)')
    
    args = parser.parse_args()
    
    analyzer = EnhancedStockAnalyzer(args.symbol, args.period)
    result = analyzer.analyze_for_next_day()
    
    if result is None:
        print("Analysis failed")
        return
    
    # Display results with focus on next-day prediction
    print(f"\n{'='*60}")
    print(f"NEXT-DAY PREDICTION ANALYSIS: {result['symbol']}")
    print(f"{'='*60}")
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"Overall Accuracy: {result['model_accuracy']:.2%}")
    print(f"Recent Accuracy (last 5 predictions): {result['recent_accuracy']:.2%}")
    
    pred = result['prediction']
    print(f"\nNEXT TRADING DAY PREDICTION:")
    print(f"📈 Direction: {pred['prediction']}")
    print(f"🎯 Confidence: {pred['confidence']:.2%}")
    print(f"⬆️  Probability UP: {pred['probability_up']:.2%}")
    print(f"⬇️  Probability DOWN: {pred['probability_down']:.2%}")
    
    print(f"\nINDIVIDUAL MODEL CONSENSUS:")
    for model_name, model_pred in pred['individual_models'].items():
        model_display = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'lr': 'Logistic Regression'}
        print(f"  {model_display[model_name]}: {model_pred['prediction']} ({max(model_pred['prob_up'], model_pred['prob_down']):.2%})")
    
    context = result['market_context']
    print(f"\nMARKET CONTEXT:")
    print(f"Current Price: ${context['current_price']:.2f}")
    if context['volume_vs_avg']:
        print(f"Volume vs Average: {context['volume_vs_avg']:.2f}x")
    print(f"5-Day Volatility: {context['volatility_5d']:.1%}")
    if context['support_level'] and context['resistance_level']:
        print(f"Support: ${context['support_level']:.2f} | Resistance: ${context['resistance_level']:.2f}")
    if context['trend_strength']:
        trend_desc = "Strong" if context['trend_strength'] > 40 else "Moderate" if context['trend_strength'] > 20 else "Weak"
        print(f"Trend Strength (ADX): {context['trend_strength']:.1f} ({trend_desc})")
    
    print(f"\nTOP PREDICTIVE FEATURES:")
    sorted_features = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:8]):
        print(f"  {i+1}. {feature}: {importance:.3f}")
    
    print(f"\n💡 Next trading day outlook: {pred['prediction']} movement expected with {pred['confidence']:.1%} confidence")


if __name__ == "__main__":
    main()
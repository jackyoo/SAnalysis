#!/usr/bin/env python3
"""
Enhanced Backtesting Script for Dual Timeframe Analysis
Tests both daily (1-day) and weekly (7-day) predictions against historical data
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


class DualBacktestAnalyzer:
    def __init__(self, symbol, lookback_days=60):
        self.symbol = symbol.upper()
        self.lookback_days = lookback_days
        self.data = None
        self.daily_results = []
        self.weekly_results = []
        
    def fetch_extended_data(self):
        """Fetch extended data for backtesting"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period='3y')  # Need more data for weekly predictions
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_indicators(self, data):
        """Calculate technical indicators for dual timeframe analysis"""
        # Price data
        high = data['High'].astype('float64').values
        low = data['Low'].astype('float64').values
        close = data['Close'].astype('float64').values
        volume = data['Volume'].astype('float64').values
        
        df = data.copy()
        
        # Basic indicators
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['RSI_9'] = talib.RSI(close, timeperiod=9)
        df['RSI_21'] = talib.RSI(close, timeperiod=21)
        df['RSI_30'] = talib.RSI(close, timeperiod=30)
        df['RSI_50'] = talib.RSI(close, timeperiod=50)
        
        # MACD
        macd, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_histogram
        df['MACD_Slope'] = df['MACD'].diff()
        
        # Weekly MACD
        macd_w, macd_signal_w, macd_histogram_w = talib.MACD(close, fastperiod=26, slowperiod=52, signalperiod=18)
        df['MACD_Weekly'] = macd_w
        df['MACD_Signal_Weekly'] = macd_signal_w
        df['MACD_Histogram_Weekly'] = macd_histogram_w
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        df['Stoch_Diff'] = stoch_k - stoch_d
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Momentum
        df['Momentum'] = talib.MOM(close, timeperiod=10)
        df['Momentum_20'] = talib.MOM(close, timeperiod=20)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        df['ROC_5'] = talib.ROC(close, timeperiod=5)
        df['ROC_20'] = talib.ROC(close, timeperiod=20)
        df['ROC_30'] = talib.ROC(close, timeperiod=30)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Weekly BB
        bb_upper_w, bb_middle_w, bb_lower_w = talib.BBANDS(close, timeperiod=50, nbdevup=2, nbdevdn=2, matype=0)
        df['BB_Upper_Weekly'] = bb_upper_w
        df['BB_Middle_Weekly'] = bb_middle_w
        df['BB_Lower_Weekly'] = bb_lower_w
        df['BB_Width_Weekly'] = (bb_upper_w - bb_lower_w) / bb_middle_w
        df['BB_Position_Weekly'] = (close - bb_lower_w) / (bb_upper_w - bb_lower_w)
        
        # ATR
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ATR_20'] = talib.ATR(high, low, close, timeperiod=20)
        df['ATR_Ratio'] = df['ATR'] / close
        df['TRANGE'] = talib.TRANGE(high, low, close)
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)
        
        for period in [5, 12, 26, 50, 100]:
            df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)
        
        # ADX
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['ADX_25'] = talib.ADX(high, low, close, timeperiod=25)
        df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # SAR
        df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        # Volume
        df['OBV'] = talib.OBV(close, volume)
        df['OBV_EMA'] = talib.EMA(df['OBV'].astype('float64').values, timeperiod=10)
        df['OBV_EMA_20'] = talib.EMA(df['OBV'].astype('float64').values, timeperiod=20)
        
        for period in [10, 20, 50]:
            df[f'Volume_SMA_{period}'] = talib.SMA(volume, timeperiod=period)
            df[f'Volume_Ratio_{period}'] = volume / df[f'Volume_SMA_{period}']
        
        # Price action
        df['High_Low_Ratio'] = (high - low) / close
        df['Open_Close_Ratio'] = (close - df['Open']) / df['Open']
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        for days in [1, 2, 3, 5, 7, 10, 20]:
            df[f'Price_Change_{days}d'] = df['Close'].pct_change(days)
        
        for window in [5, 10, 20, 50]:
            df[f'Rolling_Volatility_{window}'] = df['Price_Change_1d'].rolling(window).std()
        
        # Support/Resistance
        for window in [5, 10, 20]:
            df[f'High_{window}d'] = df['High'].rolling(window).max()
            df[f'Low_{window}d'] = df['Low'].rolling(window).min()
            df[f'Resistance_Distance_{window}d'] = (df[f'High_{window}d'] - close) / close
            df[f'Support_Distance_{window}d'] = (close - df[f'Low_{window}d']) / close
        
        return df
    
    def create_features(self, data):
        """Create features for both timeframes"""
        df = data.copy()
        
        # Target variables
        df['Next_Day_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Next_Week_Return'] = df['Close'].shift(-7) / df['Close'] - 1
        df['Daily_Target'] = (df['Next_Day_Return'] > 0).astype(int)
        df['Weekly_Target'] = (df['Next_Week_Return'] > 0).astype(int)
        
        # Time features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Is_Month_End'] = (df.index.day > 25).astype(int)
        df['Is_Quarter_End'] = (df.index.month % 3 == 0).astype(int)
        
        # Relative positions
        for period in [5, 10, 20, 50, 100, 200]:
            if f'SMA_{period}' in df.columns:
                df[f'Close_vs_SMA{period}'] = (df['Close'] / df[f'SMA_{period}'] - 1)
        
        for period in [5, 12, 26, 50, 100]:
            if f'EMA_{period}' in df.columns:
                df[f'Close_vs_EMA{period}'] = (df['Close'] / df[f'EMA_{period}'] - 1)
        
        # BB positions
        df['Close_vs_BB_Upper'] = (df['Close'] - df['BB_Upper']) / df['BB_Upper']
        df['Close_vs_BB_Lower'] = (df['Close'] - df['BB_Lower']) / df['BB_Lower']
        df['Close_vs_BB_Middle'] = (df['Close'] - df['BB_Middle']) / df['BB_Middle']
        df['Close_vs_BB_Upper_Weekly'] = (df['Close'] - df['BB_Upper_Weekly']) / df['BB_Upper_Weekly']
        df['Close_vs_BB_Lower_Weekly'] = (df['Close'] - df['BB_Lower_Weekly']) / df['BB_Lower_Weekly']
        
        # SAR
        df['Close_vs_SAR'] = (df['Close'] - df['SAR']) / df['SAR']
        
        # RSI levels
        for period in [9, 14, 21, 30, 50]:
            if f'RSI_{period}' in df.columns:
                df[f'RSI_{period}_vs_70'] = df[f'RSI_{period}'] - 70
                df[f'RSI_{period}_vs_30'] = df[f'RSI_{period}'] - 30
        
        # Advanced patterns
        df['Price_vs_OBV'] = (df['Close'].pct_change() * df['OBV'].pct_change()).rolling(5).mean()
        df['Price_vs_OBV_20'] = (df['Close'].pct_change() * df['OBV'].pct_change()).rolling(20).mean()
        df['MACD_Above_Signal'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        df['MACD_Weekly_Above_Signal'] = (df['MACD_Weekly'] > df['MACD_Signal_Weekly']).astype(int)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(20).quantile(0.2)).astype(int)
        df['BB_Squeeze_Weekly'] = (df['BB_Width_Weekly'] < df['BB_Width_Weekly'].rolling(50).quantile(0.2)).astype(int)
        df['SMA_Alignment'] = ((df['SMA_5'] > df['SMA_10']) & 
                               (df['SMA_10'] > df['SMA_20']) & 
                               (df['SMA_20'] > df['SMA_50'])).astype(int)
        
        # Feature sets
        daily_features = [
            'RSI', 'RSI_9', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Slope',
            'Stoch_K', 'Stoch_D', 'Stoch_Diff', 'Williams_R', 'Momentum', 'ROC', 'ROC_5',
            'BB_Width', 'BB_Position', 'ATR', 'ATR_Ratio', 'TRANGE',
            'ADX', 'PLUS_DI', 'MINUS_DI', 'OBV', 'Volume_Ratio_10', 'Volume_Ratio_20',
            'High_Low_Ratio', 'Open_Close_Ratio', 'Gap', 'Price_Change_1d', 'Price_Change_2d',
            'Price_Change_5d', 'Rolling_Volatility_5', 'Rolling_Volatility_10',
            'Resistance_Distance_5d', 'Support_Distance_5d', 'Day_of_Week',
            'Close_vs_SMA5', 'Close_vs_SMA10', 'Close_vs_SMA20', 'Close_vs_EMA5', 'Close_vs_EMA12',
            'Close_vs_BB_Upper', 'Close_vs_BB_Lower', 'Close_vs_SAR',
            'RSI_vs_70', 'RSI_vs_30', 'Price_vs_OBV', 'MACD_Above_Signal',
            'RSI_Oversold', 'RSI_Overbought', 'BB_Squeeze'
        ]
        
        weekly_features = [
            'RSI', 'RSI_21', 'RSI_30', 'RSI_50', 'MACD_Weekly', 'MACD_Signal_Weekly', 'MACD_Histogram_Weekly',
            'Momentum_20', 'ROC_20', 'ROC_30', 'BB_Width_Weekly', 'BB_Position_Weekly',
            'ATR_20', 'ADX', 'ADX_25', 'PLUS_DI', 'MINUS_DI', 'OBV', 'Volume_Ratio_50',
            'Price_Change_7d', 'Price_Change_10d', 'Price_Change_20d',
            'Rolling_Volatility_20', 'Rolling_Volatility_50',
            'Resistance_Distance_20d', 'Support_Distance_20d', 'Month', 'Quarter',
            'Is_Month_End', 'Is_Quarter_End',
            'Close_vs_SMA20', 'Close_vs_SMA50', 'Close_vs_SMA100', 'Close_vs_SMA200',
            'Close_vs_EMA26', 'Close_vs_EMA50', 'Close_vs_EMA100',
            'Close_vs_BB_Upper_Weekly', 'Close_vs_BB_Lower_Weekly',
            'RSI_30_vs_70', 'RSI_30_vs_30', 'RSI_50_vs_70', 'RSI_50_vs_30',
            'Price_vs_OBV_20', 'MACD_Weekly_Above_Signal', 'BB_Squeeze_Weekly', 'SMA_Alignment'
        ]
        
        # Filter available features
        daily_features = [f for f in daily_features if f in df.columns]
        weekly_features = [f for f in weekly_features if f in df.columns]
        
        return df, daily_features, weekly_features
    
    def backtest_dual_timeframe(self):
        """Backtest both daily and weekly predictions"""
        if not self.fetch_extended_data():
            return None
            
        # Prepare data
        data_with_indicators = self.calculate_indicators(self.data)
        data_with_features, daily_features, weekly_features = self.create_features(data_with_indicators)
        
        # Define backtest period
        end_date = data_with_features.index[-1]
        start_date = end_date - timedelta(days=self.lookback_days)
        backtest_period = data_with_features[data_with_features.index >= start_date].copy()
        
        print(f"\n{'='*70}")
        print(f"DUAL TIMEFRAME BACKTESTING: {self.symbol}")
        print(f"{'='*70}")
        print(f"Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Trading Days: {len(backtest_period)}")
        
        daily_results = []
        weekly_results = []
        
        # Daily predictions backtest
        for i in range(len(backtest_period) - 1):
            current_date = backtest_period.index[i]
            next_date = backtest_period.index[i + 1]
            
            train_end_idx = data_with_features.index.get_loc(current_date)
            train_start_idx = max(0, train_end_idx - 300)  # Use 300 days of training
            
            train_data = data_with_features.iloc[train_start_idx:train_end_idx + 1]
            train_clean = train_data[daily_features + ['Daily_Target']].dropna()
            
            if len(train_clean) < 100:
                continue
                
            try:
                # Train daily model
                X_train = train_clean[daily_features]
                y_train = train_clean['Daily_Target']
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                ensemble = self._create_ensemble()
                ensemble.fit(X_train_scaled, y_train)
                
                # Predict
                current_features = data_with_features.loc[current_date, daily_features].values.reshape(1, -1)
                current_scaled = scaler.transform(current_features)
                
                pred_proba = ensemble.predict_proba(current_scaled)[0]
                prediction = ensemble.predict(current_scaled)[0]
                
                # Actual result
                actual_return = (data_with_features.loc[next_date, 'Close'] / 
                               data_with_features.loc[current_date, 'Close'] - 1)
                actual = 1 if actual_return > 0 else 0
                
                daily_results.append({
                    'date': current_date,
                    'next_date': next_date,
                    'predicted': prediction,
                    'prob_up': pred_proba[1],
                    'prob_down': pred_proba[0],
                    'confidence': max(pred_proba),
                    'actual': actual,
                    'actual_return': actual_return,
                    'correct': prediction == actual
                })
                
            except Exception as e:
                continue
        
        # Weekly predictions backtest
        for i in range(len(backtest_period) - 7):  # Need 7 days ahead
            current_date = backtest_period.index[i]
            week_ahead_date = backtest_period.index[i + 7] if i + 7 < len(backtest_period) else None
            
            if week_ahead_date is None:
                continue
                
            train_end_idx = data_with_features.index.get_loc(current_date)
            train_start_idx = max(0, train_end_idx - 400)  # Use more data for weekly
            
            train_data = data_with_features.iloc[train_start_idx:train_end_idx + 1]
            train_clean = train_data[weekly_features + ['Weekly_Target']].dropna()
            
            if len(train_clean) < 150:
                continue
                
            try:
                # Train weekly model
                X_train = train_clean[weekly_features]
                y_train = train_clean['Weekly_Target']
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                ensemble = self._create_ensemble()
                ensemble.fit(X_train_scaled, y_train)
                
                # Predict
                current_features = data_with_features.loc[current_date, weekly_features].values.reshape(1, -1)
                current_scaled = scaler.transform(current_features)
                
                pred_proba = ensemble.predict_proba(current_scaled)[0]
                prediction = ensemble.predict(current_scaled)[0]
                
                # Actual result
                actual_return = (data_with_features.loc[week_ahead_date, 'Close'] / 
                               data_with_features.loc[current_date, 'Close'] - 1)
                actual = 1 if actual_return > 0 else 0
                
                weekly_results.append({
                    'date': current_date,
                    'target_date': week_ahead_date,
                    'predicted': prediction,
                    'prob_up': pred_proba[1],
                    'prob_down': pred_proba[0],
                    'confidence': max(pred_proba),
                    'actual': actual,
                    'actual_return': actual_return,
                    'correct': prediction == actual
                })
                
            except Exception as e:
                continue
        
        return daily_results, weekly_results
    
    def _create_ensemble(self):
        """Create ensemble model"""
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        lr = LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000)
        
        return VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
    
    def analyze_results(self, daily_results, weekly_results):
        """Analyze backtest results for both timeframes"""
        print(f"\n{'='*70}")
        print(f"DUAL TIMEFRAME BACKTEST RESULTS")
        print(f"{'='*70}")
        
        # Daily analysis
        if daily_results:
            daily_df = pd.DataFrame(daily_results)
            daily_accuracy = daily_df['correct'].mean()
            daily_total = len(daily_df)
            daily_correct = daily_df['correct'].sum()
            
            print(f"\n🔸 DAILY (1-DAY) PREDICTIONS:")
            print(f"Total Predictions: {daily_total}")
            print(f"Correct Predictions: {daily_correct}")
            print(f"Accuracy: {daily_accuracy:.2%}")
            
            # Confidence analysis
            high_conf_daily = daily_df[daily_df['confidence'] > 0.7]
            if len(high_conf_daily) > 0:
                print(f"High Confidence (>70%): {len(high_conf_daily)} predictions, {high_conf_daily['correct'].mean():.2%} accuracy")
            
            # Trading simulation
            portfolio_daily = 1000
            for _, row in daily_df.iterrows():
                if row['predicted'] == 1:  # Buy
                    portfolio_daily *= (1 + row['actual_return'])
                else:  # Sell/Short
                    portfolio_daily *= (1 - row['actual_return'])
            
            buy_hold_daily = 1000 * (1 + daily_df['actual_return'].sum())
            daily_return = (portfolio_daily / 1000 - 1)
            daily_bh_return = (buy_hold_daily / 1000 - 1)
            
            print(f"Strategy Return: {daily_return:+.2%}")
            print(f"Buy & Hold Return: {daily_bh_return:+.2%}")
            print(f"Excess Return: {daily_return - daily_bh_return:+.2%}")
        
        # Weekly analysis
        if weekly_results:
            weekly_df = pd.DataFrame(weekly_results)
            weekly_accuracy = weekly_df['correct'].mean()
            weekly_total = len(weekly_df)
            weekly_correct = weekly_df['correct'].sum()
            
            print(f"\n🔹 WEEKLY (7-DAY) PREDICTIONS:")
            print(f"Total Predictions: {weekly_total}")
            print(f"Correct Predictions: {weekly_correct}")
            print(f"Accuracy: {weekly_accuracy:.2%}")
            
            # Confidence analysis
            high_conf_weekly = weekly_df[weekly_df['confidence'] > 0.7]
            if len(high_conf_weekly) > 0:
                print(f"High Confidence (>70%): {len(high_conf_weekly)} predictions, {high_conf_weekly['correct'].mean():.2%} accuracy")
            
            # Trading simulation
            portfolio_weekly = 1000
            for _, row in weekly_df.iterrows():
                if row['predicted'] == 1:  # Buy
                    portfolio_weekly *= (1 + row['actual_return'])
                else:  # Sell/Short
                    portfolio_weekly *= (1 - row['actual_return'])
            
            buy_hold_weekly = 1000 * (1 + weekly_df['actual_return'].sum())
            weekly_return = (portfolio_weekly / 1000 - 1)
            weekly_bh_return = (buy_hold_weekly / 1000 - 1)
            
            print(f"Strategy Return: {weekly_return:+.2%}")
            print(f"Buy & Hold Return: {weekly_bh_return:+.2%}")
            print(f"Excess Return: {weekly_return - weekly_bh_return:+.2%}")
        
        # Recent predictions breakdown
        if daily_results and len(daily_results) >= 10:
            print(f"\n📅 RECENT DAILY PREDICTIONS:")
            print(f"{'Date':<12} {'Pred':<4} {'Actual':<6} {'Correct':<7} {'Conf':<6} {'Return':<8}")
            print(f"{'-'*50}")
            
            for result in daily_results[-10:]:
                pred_str = "UP" if result['predicted'] == 1 else "DOWN"
                actual_str = "UP" if result['actual'] == 1 else "DOWN"
                correct_str = "✓" if result['correct'] else "✗"
                
                print(f"{result['date'].strftime('%Y-%m-%d'):<12} {pred_str:<4} {actual_str:<6} {correct_str:<7} {result['confidence']:<6.1%} {result['actual_return']:+6.2%}")
        
        if weekly_results and len(weekly_results) >= 5:
            print(f"\n📅 RECENT WEEKLY PREDICTIONS:")
            print(f"{'Date':<12} {'Pred':<4} {'Actual':<6} {'Correct':<7} {'Conf':<6} {'Return':<8}")
            print(f"{'-'*50}")
            
            for result in weekly_results[-5:]:
                pred_str = "UP" if result['predicted'] == 1 else "DOWN"
                actual_str = "UP" if result['actual'] == 1 else "DOWN"
                correct_str = "✓" if result['correct'] else "✗"
                
                print(f"{result['date'].strftime('%Y-%m-%d'):<12} {pred_str:<4} {actual_str:<6} {correct_str:<7} {result['confidence']:<6.1%} {result['actual_return']:+6.2%}")
        
        return {
            'daily_accuracy': daily_accuracy if daily_results else 0,
            'weekly_accuracy': weekly_accuracy if weekly_results else 0,
            'daily_results': len(daily_results) if daily_results else 0,
            'weekly_results': len(weekly_results) if weekly_results else 0
        }


def main():
    """Main function for dual timeframe backtesting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Dual Timeframe Stock Analyzer')
    parser.add_argument('symbol', help='Stock symbol (e.g., NVDA, AAPL)')
    parser.add_argument('--days', type=int, default=60, help='Number of days to backtest (default: 60)')
    
    args = parser.parse_args()
    
    backtester = DualBacktestAnalyzer(args.symbol, lookback_days=args.days)
    daily_results, weekly_results = backtester.backtest_dual_timeframe()
    
    if daily_results or weekly_results:
        analysis = backtester.analyze_results(daily_results, weekly_results)
        
        print(f"\n💡 SUMMARY:")
        if daily_results:
            print(f"Daily Model: {analysis['daily_accuracy']:.1%} accuracy over {analysis['daily_results']} predictions")
        if weekly_results:
            print(f"Weekly Model: {analysis['weekly_accuracy']:.1%} accuracy over {analysis['weekly_results']} predictions")
    else:
        print("Backtesting failed")


if __name__ == "__main__":
    main()
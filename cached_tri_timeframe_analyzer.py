#!/usr/bin/env python3
"""
Cached Tri-Timeframe Stock Analyzer
Intelligently caches historical data and only fetches recent updates
Reduces API calls and improves performance dramatically
"""

import yfinance as yf
import talib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import argparse
import os
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CachedTriTimeframeAnalyzer:
    def __init__(self, symbol, period='25y', cache_dir='cache'):
        self.symbol = symbol.upper()
        self.period = period
        self.cache_dir = cache_dir
        self.data = None
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache file paths
        self.cache_file = os.path.join(cache_dir, f"{self.symbol}_{period}_data.pkl")
        self.metadata_file = os.path.join(cache_dir, f"{self.symbol}_{period}_metadata.json")
        
        # Model components
        self.daily_model = None
        self.weekly_model = None
        self.biweekly_model = None
        self.daily_price_model = None
        self.weekly_price_model = None
        self.biweekly_price_model = None
        self.daily_scaler = StandardScaler()
        self.weekly_scaler = StandardScaler()
        self.biweekly_scaler = StandardScaler()
        self.daily_price_scaler = StandardScaler()
        self.weekly_price_scaler = StandardScaler()
        self.biweekly_price_scaler = StandardScaler()
        
    def get_cache_metadata(self):
        """Get metadata about cached data"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_cache_metadata(self, metadata):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f)
    
    def is_cache_valid(self):
        """Check if cache is valid (updated today)"""
        metadata = self.get_cache_metadata()
        if not metadata or not os.path.exists(self.cache_file):
            return False
            
        # Check if cache is from today
        cache_date = datetime.fromisoformat(metadata['last_update']).date()
        today = datetime.now().date()
        
        # Cache is valid if it's from today or if markets are closed
        return cache_date >= today or not self.is_market_day(today)
    
    def is_market_day(self, date):
        """Simple check if it's likely a market day (weekday, basic holidays excluded)"""
        if date.weekday() >= 5:  # Weekend
            return False
        # You could add more sophisticated holiday checking here
        return True
    
    def fetch_smart_data(self):
        """Smart data fetching with caching"""
        print(f"🧠 Smart data fetching for {self.symbol} ({self.period})...")
        
        if self.is_cache_valid():
            print("📁 Loading from cache (no network call needed)...")
            try:
                with open(self.cache_file, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"✅ Loaded {len(self.data)} days from cache")
                return True
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}, fetching fresh data...")
        
        # Determine if we can do incremental update
        metadata = self.get_cache_metadata()
        if metadata and os.path.exists(self.cache_file):
            return self.fetch_incremental_data()
        else:
            return self.fetch_full_data()
    
    def fetch_incremental_data(self):
        """Fetch only new data since last cache update"""
        try:
            # Load existing cached data
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            metadata = self.get_cache_metadata()
            last_date = datetime.fromisoformat(metadata['last_date']).date()
            
            print(f"📈 Incremental update: fetching data since {last_date}...")
            
            # Fetch only recent data (last 30 days to ensure overlap)
            ticker = yf.Ticker(self.symbol)
            start_date = last_date - timedelta(days=30)
            recent_data = ticker.history(start=start_date)
            
            if recent_data.empty:
                print("⚠️  No recent data available, using cached data")
                self.data = cached_data
                return True
            
            # Merge data: use cached data up to overlap point, then new data
            overlap_date = cached_data.index[-1].date()
            
            # Remove any overlapping data from recent_data
            recent_data = recent_data[recent_data.index.date > overlap_date]
            
            if not recent_data.empty:
                # Concatenate old and new data
                self.data = pd.concat([cached_data, recent_data])
                self.data = self.data[~self.data.index.duplicated(keep='last')]  # Remove duplicates
                self.data = self.data.sort_index()
                
                print(f"✅ Added {len(recent_data)} new trading days")
                print(f"📊 Total dataset: {len(self.data)} days")
            else:
                print("📁 No new data to add, using cached data")
                self.data = cached_data
            
            # Update cache
            self.save_data_cache()
            return True
            
        except Exception as e:
            print(f"❌ Incremental update failed: {e}")
            print("🔄 Falling back to full data fetch...")
            return self.fetch_full_data()
    
    def fetch_full_data(self):
        """Fetch full historical data (fallback method)"""
        print(f"🌐 Fetching full {self.period} dataset (this may take a moment)...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            print(f"✅ Downloaded {len(self.data)} days of data")
            
            # Save to cache
            self.save_data_cache()
            return True
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def save_data_cache(self):
        """Save data and metadata to cache"""
        try:
            # Save data
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Save metadata
            metadata = {
                'symbol': self.symbol,
                'period': self.period,
                'last_update': datetime.now().isoformat(),
                'last_date': self.data.index[-1].isoformat(),
                'total_days': len(self.data),
                'date_range': {
                    'start': self.data.index[0].isoformat(),
                    'end': self.data.index[-1].isoformat()
                }
            }
            self.save_cache_metadata(metadata)
            
            print(f"💾 Cached data updated: {len(self.data)} days")
            
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
    
    def calculate_advanced_indicators(self):
        """Calculate comprehensive technical indicators for tri-timeframe analysis"""
        if self.data is None or self.data.empty:
            return False
            
        print("🔧 Calculating technical indicators...")
        
        # Price data (convert to float64 for TA-Lib)
        high = self.data['High'].astype('float64').values
        low = self.data['Low'].astype('float64').values
        close = self.data['Close'].astype('float64').values
        volume = self.data['Volume'].astype('float64').values
        open_price = self.data['Open'].astype('float64').values
        
        # === SHORT-TERM INDICATORS (for daily prediction) ===
        self.data['RSI'] = talib.RSI(close, timeperiod=14)
        self.data['RSI_9'] = talib.RSI(close, timeperiod=9)
        self.data['RSI_21'] = talib.RSI(close, timeperiod=21)
        
        # MACD family
        macd, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Histogram'] = macd_histogram
        self.data['MACD_Slope'] = pd.Series(macd, index=self.data.index).diff()
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        self.data['Stoch_K'] = stoch_k
        self.data['Stoch_D'] = stoch_d
        self.data['Stoch_Diff'] = stoch_k - stoch_d
        
        # Williams %R
        self.data['Williams_R'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Short-term momentum
        self.data['Momentum'] = talib.MOM(close, timeperiod=10)
        self.data['ROC'] = talib.ROC(close, timeperiod=10)
        self.data['ROC_5'] = talib.ROC(close, timeperiod=5)
        
        # === MEDIUM-TERM INDICATORS (for weekly prediction) ===
        # Longer-term RSI
        self.data['RSI_30'] = talib.RSI(close, timeperiod=30)
        self.data['RSI_50'] = talib.RSI(close, timeperiod=50)
        
        # Longer-term momentum
        self.data['Momentum_20'] = talib.MOM(close, timeperiod=20)
        self.data['ROC_20'] = talib.ROC(close, timeperiod=20)
        self.data['ROC_30'] = talib.ROC(close, timeperiod=30)
        
        # Weekly MACD
        macd_w, macd_signal_w, macd_histogram_w = talib.MACD(close, fastperiod=26, slowperiod=52, signalperiod=18)
        self.data['MACD_Weekly'] = macd_w
        self.data['MACD_Signal_Weekly'] = macd_signal_w
        self.data['MACD_Histogram_Weekly'] = macd_histogram_w
        
        # === LONG-TERM INDICATORS (for bi-weekly prediction) ===
        # Very long-term RSI
        self.data['RSI_70'] = talib.RSI(close, timeperiod=70)
        self.data['RSI_100'] = talib.RSI(close, timeperiod=100)
        
        # Long-term momentum
        self.data['Momentum_50'] = talib.MOM(close, timeperiod=50)
        self.data['ROC_50'] = talib.ROC(close, timeperiod=50)
        self.data['ROC_70'] = talib.ROC(close, timeperiod=70)
        
        # Monthly MACD
        macd_m, macd_signal_m, macd_histogram_m = talib.MACD(close, fastperiod=52, slowperiod=104, signalperiod=36)
        self.data['MACD_Monthly'] = macd_m
        self.data['MACD_Signal_Monthly'] = macd_signal_m
        self.data['MACD_Histogram_Monthly'] = macd_histogram_m
        
        # === SHARED INDICATORS ===
        # Bollinger Bands (multiple timeframes)
        bb_upper_20, bb_middle_20, bb_lower_20 = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.data['BB_Upper_20'] = bb_upper_20
        self.data['BB_Middle_20'] = bb_middle_20
        self.data['BB_Lower_20'] = bb_lower_20
        self.data['BB_Position_20'] = (close - bb_lower_20) / (bb_upper_20 - bb_lower_20)
        
        bb_upper_50, bb_middle_50, bb_lower_50 = talib.BBANDS(close, timeperiod=50, nbdevup=2, nbdevdn=2, matype=0)
        self.data['BB_Upper_50'] = bb_upper_50
        self.data['BB_Middle_50'] = bb_middle_50
        self.data['BB_Lower_50'] = bb_lower_50
        self.data['BB_Position_50'] = (close - bb_lower_50) / (bb_upper_50 - bb_lower_50)
        
        # ATR (Average True Range) - multiple timeframes
        self.data['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        self.data['ATR_30'] = talib.ATR(high, low, close, timeperiod=30)
        self.data['ATR_50'] = talib.ATR(high, low, close, timeperiod=50)
        
        # ADX (trend strength)
        self.data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        self.data['ADX_30'] = talib.ADX(high, low, close, timeperiod=30)
        
        # Parabolic SAR
        self.data['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        self.data['Close_vs_SAR'] = (close - self.data['SAR']) / close
        
        # Volume indicators
        self.data['OBV'] = talib.OBV(close, volume)
        self.data['AD'] = talib.AD(high, low, close, volume)
        
        # Volume moving averages
        self.data['Volume_SMA_10'] = talib.SMA(volume, timeperiod=10)
        self.data['Volume_SMA_30'] = talib.SMA(volume, timeperiod=30)
        self.data['Volume_Ratio'] = volume / self.data['Volume_SMA_10']
        
        # Price vs moving averages
        self.data['SMA_5'] = talib.SMA(close, timeperiod=5)
        self.data['SMA_10'] = talib.SMA(close, timeperiod=10)
        self.data['SMA_20'] = talib.SMA(close, timeperiod=20)
        self.data['SMA_50'] = talib.SMA(close, timeperiod=50)
        self.data['SMA_100'] = talib.SMA(close, timeperiod=100)
        self.data['SMA_200'] = talib.SMA(close, timeperiod=200)
        
        self.data['EMA_12'] = talib.EMA(close, timeperiod=12)
        self.data['EMA_26'] = talib.EMA(close, timeperiod=26)
        self.data['EMA_50'] = talib.EMA(close, timeperiod=50)
        self.data['EMA_100'] = talib.EMA(close, timeperiod=100)
        
        # Moving average ratios
        self.data['Close_SMA_5_Ratio'] = close / self.data['SMA_5']
        self.data['Close_SMA_20_Ratio'] = close / self.data['SMA_20']
        self.data['Close_SMA_50_Ratio'] = close / self.data['SMA_50']
        self.data['Close_SMA_200_Ratio'] = close / self.data['SMA_200']
        
        # Volatility measures
        returns = pd.Series(close, index=self.data.index).pct_change()
        self.data['Returns'] = returns
        self.data['Rolling_Volatility_5'] = returns.rolling(5).std()
        self.data['Rolling_Volatility_10'] = returns.rolling(10).std()
        self.data['Rolling_Volatility_20'] = returns.rolling(20).std()
        self.data['Rolling_Volatility_50'] = returns.rolling(50).std()
        
        print("✅ Technical indicators calculated")
        return True
    
    def prepare_datasets(self):
        """Prepare datasets for each timeframe"""
        print("📊 Preparing datasets for tri-timeframe analysis...")
        
        # Define base features (exclude price columns and targets)
        all_features = [col for col in self.data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        # === DAILY DATASET ===
        daily_features = [f for f in all_features if not any(x in f for x in ['Weekly', 'Monthly', '_50', '_70', '_100', '_200'])]
        
        # Create daily targets
        self.data['Daily_Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        self.data['Next_Day_Return'] = self.data['Close'].pct_change().shift(-1)
        
        self.daily_data = self.data[daily_features + ['Daily_Target', 'Next_Day_Return']].dropna()
        self.daily_features = daily_features
        
        # === WEEKLY DATASET ===
        weekly_features = [f for f in all_features if not any(x in f for x in ['Monthly', '_100', '_200'])]
        
        # Create weekly targets
        self.data['Weekly_Target'] = (self.data['Close'].shift(-7) > self.data['Close']).astype(int)
        self.data['Next_Week_Return'] = self.data['Close'].pct_change(7).shift(-7)
        
        self.weekly_data = self.data[weekly_features + ['Weekly_Target', 'Next_Week_Return']].dropna()
        self.weekly_features = weekly_features
        
        # === BI-WEEKLY DATASET ===
        biweekly_features = all_features  # Use all features for longest timeframe
        
        # Create bi-weekly targets
        self.data['BiWeekly_Target'] = (self.data['Close'].shift(-14) > self.data['Close']).astype(int)
        self.data['Next_BiWeek_Return'] = self.data['Close'].pct_change(14).shift(-14)
        
        self.biweekly_data = self.data[biweekly_features + ['BiWeekly_Target', 'Next_BiWeek_Return']].dropna()
        self.biweekly_features = biweekly_features
        
        print(f"✅ Datasets prepared:")
        print(f"   Daily: {len(self.daily_data)} samples, {len(self.daily_features)} features")
        print(f"   Weekly: {len(self.weekly_data)} samples, {len(self.weekly_features)} features")
        print(f"   Bi-weekly: {len(self.biweekly_data)} samples, {len(self.biweekly_features)} features")
        
        return True
    
    # [Rest of the methods from original TriTimeframeAnalyzer would go here...]
    # For brevity, I'm including just the key caching functionality
    
    def analyze_tri_timeframe(self):
        """Main analysis function with smart caching"""
        print(f"🚀 Cached Tri-Timeframe Analysis: {self.symbol}")
        print("=" * 80)
        
        # Smart data fetching (uses cache when possible)
        if not self.fetch_smart_data():
            return None
        
        # Calculate indicators
        if not self.calculate_advanced_indicators():
            return None
        
        # Prepare datasets
        if not self.prepare_datasets():
            return None
        
        # Train models (same as original)
        model_performance = self.train_models()
        if not model_performance:
            return None
        
        # Generate predictions (same as original)
        predictions = self.generate_predictions()
        if not predictions:
            return None
        
        # Format results
        result = {
            'symbol': self.symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_performance': model_performance,
            'predictions': predictions,
            'market_context': self.get_market_context(),
            'cache_info': {
                'data_cached': True,
                'cache_date': self.get_cache_metadata()['last_update'] if self.get_cache_metadata() else None,
                'total_days': len(self.data)
            }
        }
        
        self.print_analysis_results(result)
        return result
    
    def print_cache_info(self):
        """Print cache information"""
        metadata = self.get_cache_metadata()
        if metadata:
            print(f"📁 Cache Info:")
            print(f"   Last Updated: {metadata['last_update']}")
            print(f"   Data Range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
            print(f"   Total Days: {metadata['total_days']}")
        else:
            print("📁 No cache available")


def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description='Cached Tri-Timeframe Stock Analysis')
    parser.add_argument('symbol', help='Stock symbol (e.g., NVDA, AAPL)')
    parser.add_argument('--period', default='25y', help='Data period (default: 25y)')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache and fetch fresh data')
    
    args = parser.parse_args()
    
    analyzer = CachedTriTimeframeAnalyzer(args.symbol, args.period)
    
    if args.clear_cache:
        # Clear cache files
        cache_files = [analyzer.cache_file, analyzer.metadata_file]
        for file in cache_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"🗑️  Cleared cache: {file}")
    
    analyzer.analyze_tri_timeframe()


if __name__ == "__main__":
    main()
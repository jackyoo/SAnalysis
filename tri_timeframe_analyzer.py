#!/usr/bin/env python3
"""
Tri-Timeframe Stock Analyzer
Predicts next-day (1-day), weekly (7-day), and bi-weekly (14-day) price movements
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
import os
import pickle
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TriTimeframeAnalyzer:
    def __init__(self, symbol, period='1y'):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.features = None
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
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance with intelligent caching"""
        
        # Cache setup
        cache_dir = 'cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{self.symbol}_{self.period}_data.pkl")
        metadata_file = os.path.join(cache_dir, f"{self.symbol}_{self.period}_metadata.json")
        
        # Check if cache exists and is recent
        if os.path.exists(cache_file) and os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                cache_date = datetime.fromisoformat(metadata['last_update']).date()
                today = datetime.now().date()
                
                # For weekends, use cached data
                if today.weekday() >= 5:
                    print(f"📁 Weekend: Loading cached data for {self.symbol} (last updated: {cache_date})")
                    with open(cache_file, 'rb') as f:
                        self.data = pickle.load(f)
                    
                    # Verify loaded data is valid
                    if self.data is None or self.data.empty:
                        print(f"⚠️  Cached data is invalid, fetching fresh data...")
                        return self.fetch_fresh_data(cache_file, metadata_file)
                    
                    print(f"✅ Loaded {len(self.data)} days from cache")
                    return True
                
                # For weekdays, always fetch today's live data
                print(f"📈 Fetching live data to update cache (last cache: {cache_date})")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cached data is valid
                if cached_data is None or cached_data.empty:
                    print(f"⚠️  Cached data is invalid, fetching fresh data...")
                    return self.fetch_fresh_data(cache_file, metadata_file)
                
                last_date = datetime.fromisoformat(metadata['last_date']).date()
                
                # Always fetch today's data + small overlap for current prices
                ticker = yf.Ticker(self.symbol)
                start_date = last_date - timedelta(days=2)  # Small overlap
                
                # Fetch recent data with retry logic
                recent_data = None
                for attempt in range(3):  # Try up to 3 times
                    try:
                        print(f"📡 Attempting to fetch recent data (attempt {attempt + 1}/3)...")
                        recent_data = ticker.history(start=start_date)
                        if recent_data is not None and not recent_data.empty:
                            print(f"✅ Successfully fetched recent data")
                            break
                        else:
                            print(f"⚠️  Got empty data on attempt {attempt + 1}")
                    except Exception as fetch_error:
                        print(f"⚠️  Fetch attempt {attempt + 1} failed: {fetch_error}")
                        if attempt < 2:  # Don't sleep on last attempt
                            import time
                            time.sleep(2)  # Wait 2 seconds before retry
                
                # Check if recent_data is valid
                if recent_data is None or recent_data.empty:
                    print(f"⚠️  Could not fetch recent data, using cached data only")
                    self.data = cached_data
                    return True
                
                # Get current day's latest data for live pricing
                try:
                    current_data = ticker.history(period='1d')
                    if current_data is not None and not current_data.empty:
                        latest_price = current_data['Close'].iloc[-1]
                        print(f"🔴 Live price: ${latest_price:.2f}")
                        
                        # Replace today's entire row with current data
                        today_date = datetime.now().date()
                        if not recent_data.empty:
                            today_mask = recent_data.index.date == today_date
                            
                            if today_mask.any():
                                # Update the existing row with current data
                                today_index = recent_data.index[today_mask][0]
                                recent_data.loc[today_index] = current_data.iloc[-1]
                                print(f"✅ Updated today's data with live values")
                            else:
                                # Add today's data if not present
                                recent_data = pd.concat([recent_data, current_data])
                                print(f"✅ Added today's live data")
                        else:
                            # If recent_data is empty, use current_data
                            recent_data = current_data.copy()
                            print(f"✅ Added today's live data")
                            
                except Exception as e:
                    print(f"⚠️  Could not fetch live price: {e}")
                
                if recent_data is not None and not recent_data.empty:
                    # Include overlap to ensure today's updated data is used
                    if not cached_data.empty:
                        overlap_date = cached_data.index[-1].date()
                        new_data = recent_data[recent_data.index.date >= overlap_date]  # Include today's data
                        
                        if not new_data.empty:
                            # Remove the old overlapping data from cache, add new data
                            cached_data_filtered = cached_data[cached_data.index.date < overlap_date]
                            self.data = pd.concat([cached_data_filtered, new_data])
                            self.data = self.data.sort_index()
                            print(f"✅ Updated cache with {len(new_data)} days (including today's live data)")
                        else:
                            self.data = cached_data
                            print(f"📁 No new data, using cached data")
                    else:
                        self.data = recent_data
                        print(f"✅ Using recent data as cache was empty")
                else:
                    self.data = cached_data
                    print(f"📁 No recent data available, using cached data")
                
            except Exception as e:
                import traceback
                print(f"⚠️  Cache error: {e}")
                print(f"⚠️  Stack trace: {traceback.format_exc()}")
                print("⚠️  Fetching fresh data...")
                return self.fetch_fresh_data(cache_file, metadata_file)
        else:
            print(f"🌐 No cache found, fetching {self.period} of data...")
            return self.fetch_fresh_data(cache_file, metadata_file)
        
        # Save updated cache
        try:
            if self.data is not None and not self.data.empty:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.data, f)
                
                metadata = {
                    'symbol': self.symbol,
                    'period': self.period,
                    'last_update': datetime.now().isoformat(),
                    'last_date': self.data.index[-1].isoformat(),
                    'total_days': len(self.data)
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
                    
                print(f"💾 Cache updated: {len(self.data)} total days")
            
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")
        
        return True
    
    def fetch_fresh_data(self, cache_file, metadata_file):
        """Fetch fresh data and save to cache"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Fetch data with retry logic
            self.data = None
            for attempt in range(3):  # Try up to 3 times
                try:
                    print(f"📡 Fetching {self.period} data (attempt {attempt + 1}/3)...")
                    self.data = ticker.history(period=self.period)
                    
                    if self.data is not None and not self.data.empty and len(self.data.index) > 0:
                        print(f"✅ Downloaded {len(self.data)} days of data")
                        break
                    else:
                        print(f"⚠️  Got invalid data on attempt {attempt + 1}")
                        
                except Exception as fetch_error:
                    print(f"⚠️  Fetch attempt {attempt + 1} failed: {fetch_error}")
                    if attempt < 2:  # Don't sleep on last attempt
                        import time
                        time.sleep(2)  # Wait 2 seconds before retry
            
            # Final validation
            if self.data is None:
                raise ValueError(f"Failed to fetch data for symbol {self.symbol} - got None after all retries")
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol} - empty dataset after all retries")
            
            if len(self.data.index) == 0:
                raise ValueError(f"No index data found for symbol {self.symbol} after all retries")
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Create metadata safely
            try:
                last_date_iso = self.data.index[-1].isoformat()
            except (IndexError, AttributeError) as e:
                raise ValueError(f"Cannot access last date from data index: {e}")
            
            metadata = {
                'symbol': self.symbol,
                'period': self.period,
                'last_update': datetime.now().isoformat(),
                'last_date': last_date_iso,
                'total_days': len(self.data)
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
            print(f"💾 Data cached for future use")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_advanced_indicators(self):
        """Calculate comprehensive technical indicators for tri-timeframe analysis"""
        if self.data is None or self.data.empty:
            return False
            
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
        self.data['MACD_Slope'] = self.data['MACD'].diff()
        
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
        # Extended RSI periods
        self.data['RSI_70'] = talib.RSI(close, timeperiod=70)
        self.data['RSI_100'] = talib.RSI(close, timeperiod=100)
        
        # Extended momentum
        self.data['Momentum_30'] = talib.MOM(close, timeperiod=30)
        self.data['Momentum_50'] = talib.MOM(close, timeperiod=50)
        self.data['ROC_50'] = talib.ROC(close, timeperiod=50)
        self.data['ROC_70'] = talib.ROC(close, timeperiod=70)
        
        # Monthly MACD
        macd_m, macd_signal_m, macd_histogram_m = talib.MACD(close, fastperiod=52, slowperiod=104, signalperiod=36)
        self.data['MACD_Monthly'] = macd_m
        self.data['MACD_Signal_Monthly'] = macd_signal_m
        self.data['MACD_Histogram_Monthly'] = macd_histogram_m
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands (multiple timeframes)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.data['BB_Upper'] = bb_upper
        self.data['BB_Middle'] = bb_middle
        self.data['BB_Lower'] = bb_lower
        self.data['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        self.data['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Weekly Bollinger Bands
        bb_upper_w, bb_middle_w, bb_lower_w = talib.BBANDS(close, timeperiod=50, nbdevup=2, nbdevdn=2, matype=0)
        self.data['BB_Upper_Weekly'] = bb_upper_w
        self.data['BB_Middle_Weekly'] = bb_middle_w
        self.data['BB_Lower_Weekly'] = bb_lower_w
        self.data['BB_Width_Weekly'] = (bb_upper_w - bb_lower_w) / bb_middle_w
        self.data['BB_Position_Weekly'] = (close - bb_lower_w) / (bb_upper_w - bb_lower_w)
        
        # Monthly Bollinger Bands
        bb_upper_m, bb_middle_m, bb_lower_m = talib.BBANDS(close, timeperiod=100, nbdevup=2.5, nbdevdn=2.5, matype=0)
        self.data['BB_Upper_Monthly'] = bb_upper_m
        self.data['BB_Middle_Monthly'] = bb_middle_m
        self.data['BB_Lower_Monthly'] = bb_lower_m
        self.data['BB_Width_Monthly'] = (bb_upper_m - bb_lower_m) / bb_middle_m
        self.data['BB_Position_Monthly'] = (close - bb_lower_m) / (bb_upper_m - bb_lower_m)
        
        # ATR (multiple timeframes)
        self.data['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        self.data['ATR_20'] = talib.ATR(high, low, close, timeperiod=20)
        self.data['ATR_50'] = talib.ATR(high, low, close, timeperiod=50)
        self.data['ATR_100'] = talib.ATR(high, low, close, timeperiod=100)
        self.data['ATR_Ratio'] = self.data['ATR'] / close
        self.data['ATR_Ratio_50'] = self.data['ATR_50'] / close
        self.data['TRANGE'] = talib.TRANGE(high, low, close)
        
        # === TREND INDICATORS ===
        # Moving Averages (multiple timeframes)
        self.data['SMA_5'] = talib.SMA(close, timeperiod=5)
        self.data['SMA_10'] = talib.SMA(close, timeperiod=10)
        self.data['SMA_20'] = talib.SMA(close, timeperiod=20)
        self.data['SMA_50'] = talib.SMA(close, timeperiod=50)
        self.data['SMA_100'] = talib.SMA(close, timeperiod=100)
        self.data['SMA_200'] = talib.SMA(close, timeperiod=200)
        
        self.data['EMA_5'] = talib.EMA(close, timeperiod=5)
        self.data['EMA_12'] = talib.EMA(close, timeperiod=12)
        self.data['EMA_26'] = talib.EMA(close, timeperiod=26)
        self.data['EMA_50'] = talib.EMA(close, timeperiod=50)
        self.data['EMA_100'] = talib.EMA(close, timeperiod=100)
        
        # ADX (Directional Movement)
        self.data['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        self.data['ADX_25'] = talib.ADX(high, low, close, timeperiod=25)
        self.data['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        self.data['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Parabolic SAR
        self.data['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        # === VOLUME INDICATORS ===
        self.data['OBV'] = talib.OBV(close, volume)
        self.data['OBV_EMA'] = talib.EMA(self.data['OBV'].astype('float64').values, timeperiod=10)
        self.data['OBV_EMA_20'] = talib.EMA(self.data['OBV'].astype('float64').values, timeperiod=20)
        
        # Volume ratios (multiple timeframes)
        self.data['Volume_SMA_10'] = talib.SMA(volume, timeperiod=10)
        self.data['Volume_SMA_20'] = talib.SMA(volume, timeperiod=20)
        self.data['Volume_SMA_50'] = talib.SMA(volume, timeperiod=50)
        self.data['Volume_Ratio_10'] = volume / self.data['Volume_SMA_10']
        self.data['Volume_Ratio_20'] = volume / self.data['Volume_SMA_20']
        self.data['Volume_Ratio_50'] = volume / self.data['Volume_SMA_50']
        
        # === PRICE ACTION FEATURES ===
        self.data['High_Low_Ratio'] = (high - low) / close
        self.data['Open_Close_Ratio'] = (close - open_price) / open_price
        self.data['Gap'] = (self.data['Open'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1)
        
        # Price changes (multiple timeframes)
        self.data['Price_Change_1d'] = self.data['Close'].pct_change(1)
        self.data['Price_Change_2d'] = self.data['Close'].pct_change(2)
        self.data['Price_Change_3d'] = self.data['Close'].pct_change(3)
        self.data['Price_Change_5d'] = self.data['Close'].pct_change(5)
        self.data['Price_Change_7d'] = self.data['Close'].pct_change(7)
        self.data['Price_Change_10d'] = self.data['Close'].pct_change(10)
        self.data['Price_Change_14d'] = self.data['Close'].pct_change(14)
        self.data['Price_Change_20d'] = self.data['Close'].pct_change(20)
        self.data['Price_Change_30d'] = self.data['Close'].pct_change(30)
        self.data['Price_Change_50d'] = self.data['Close'].pct_change(50)
        
        # Volatility measures (multiple timeframes)
        self.data['Rolling_Volatility_5'] = self.data['Price_Change_1d'].rolling(5).std()
        self.data['Rolling_Volatility_10'] = self.data['Price_Change_1d'].rolling(10).std()
        self.data['Rolling_Volatility_20'] = self.data['Price_Change_1d'].rolling(20).std()
        self.data['Rolling_Volatility_30'] = self.data['Price_Change_1d'].rolling(30).std()
        self.data['Rolling_Volatility_50'] = self.data['Price_Change_1d'].rolling(50).std()
        self.data['Rolling_Volatility_100'] = self.data['Price_Change_1d'].rolling(100).std()
        
        # Support/Resistance levels (multiple timeframes)
        self.data['High_5d'] = self.data['High'].rolling(5).max()
        self.data['Low_5d'] = self.data['Low'].rolling(5).min()
        self.data['High_10d'] = self.data['High'].rolling(10).max()
        self.data['Low_10d'] = self.data['Low'].rolling(10).min()
        self.data['High_20d'] = self.data['High'].rolling(20).max()
        self.data['Low_20d'] = self.data['Low'].rolling(20).min()
        self.data['High_50d'] = self.data['High'].rolling(50).max()
        self.data['Low_50d'] = self.data['Low'].rolling(50).min()
        self.data['High_100d'] = self.data['High'].rolling(100).max()
        self.data['Low_100d'] = self.data['Low'].rolling(100).min()
        
        self.data['Resistance_Distance_5d'] = (self.data['High_5d'] - close) / close
        self.data['Support_Distance_5d'] = (close - self.data['Low_5d']) / close
        self.data['Resistance_Distance_20d'] = (self.data['High_20d'] - close) / close
        self.data['Support_Distance_20d'] = (close - self.data['Low_20d']) / close
        self.data['Resistance_Distance_50d'] = (self.data['High_50d'] - close) / close
        self.data['Support_Distance_50d'] = (close - self.data['Low_50d']) / close
        
        return True
    
    def create_tri_timeframe_features(self):
        """Create features for daily, weekly, and bi-weekly predictions"""
        # Create target variables for all three timeframes
        self.data['Next_Day_Return'] = self.data['Close'].shift(-1) / self.data['Close'] - 1
        self.data['Next_Week_Return'] = self.data['Close'].shift(-7) / self.data['Close'] - 1
        self.data['Next_BiWeek_Return'] = self.data['Close'].shift(-14) / self.data['Close'] - 1
        
        self.data['Daily_Target'] = (self.data['Next_Day_Return'] > 0).astype(int)
        self.data['Weekly_Target'] = (self.data['Next_Week_Return'] > 0).astype(int)
        self.data['BiWeekly_Target'] = (self.data['Next_BiWeek_Return'] > 0).astype(int)
        
        # Time-based features
        self.data['Day_of_Week'] = self.data.index.dayofweek
        self.data['Month'] = self.data.index.month
        self.data['Quarter'] = self.data.index.quarter
        self.data['Is_Month_End'] = (self.data.index.day > 25).astype(int)
        self.data['Is_Quarter_End'] = (self.data.index.month % 3 == 0).astype(int)
        
        # Relative position features (multiple timeframes)
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            if f'SMA_{period}' in self.data.columns:
                self.data[f'Close_vs_SMA{period}'] = (self.data['Close'] / self.data[f'SMA_{period}'] - 1)
        
        ema_periods = [5, 12, 26, 50, 100]
        for period in ema_periods:
            if f'EMA_{period}' in self.data.columns:
                self.data[f'Close_vs_EMA{period}'] = (self.data['Close'] / self.data[f'EMA_{period}'] - 1)
        
        # Bollinger Band positions
        self.data['Close_vs_BB_Upper'] = (self.data['Close'] - self.data['BB_Upper']) / self.data['BB_Upper']
        self.data['Close_vs_BB_Lower'] = (self.data['Close'] - self.data['BB_Lower']) / self.data['BB_Lower']
        self.data['Close_vs_BB_Middle'] = (self.data['Close'] - self.data['BB_Middle']) / self.data['BB_Middle']
        self.data['Close_vs_BB_Upper_Weekly'] = (self.data['Close'] - self.data['BB_Upper_Weekly']) / self.data['BB_Upper_Weekly']
        self.data['Close_vs_BB_Lower_Weekly'] = (self.data['Close'] - self.data['BB_Lower_Weekly']) / self.data['BB_Lower_Weekly']
        
        # SAR position
        self.data['Close_vs_SAR'] = (self.data['Close'] - self.data['SAR']) / self.data['SAR']
        
        # RSI levels
        for period in [9, 14, 21, 30, 50, 70, 100]:
            if f'RSI_{period}' in self.data.columns:
                self.data[f'RSI_{period}_vs_70'] = self.data[f'RSI_{period}'] - 70
                self.data[f'RSI_{period}_vs_30'] = self.data[f'RSI_{period}'] - 30
        
        # Momentum convergence/divergence
        self.data['Price_vs_OBV'] = (self.data['Close'].pct_change() * self.data['OBV'].pct_change()).rolling(5).mean()
        self.data['Price_vs_OBV_20'] = (self.data['Close'].pct_change() * self.data['OBV'].pct_change()).rolling(20).mean()
        self.data['Price_vs_OBV_50'] = (self.data['Close'].pct_change() * self.data['OBV'].pct_change()).rolling(50).mean()
        
        # Advanced patterns
        self.data['MACD_Above_Signal'] = (self.data['MACD'] > self.data['MACD_Signal']).astype(int)
        self.data['MACD_Weekly_Above_Signal'] = (self.data['MACD_Weekly'] > self.data['MACD_Signal_Weekly']).astype(int)
        self.data['MACD_Monthly_Above_Signal'] = (self.data['MACD_Monthly'] > self.data['MACD_Signal_Monthly']).astype(int)
        
        # Multi-timeframe RSI conditions
        self.data['RSI_Oversold'] = (self.data['RSI'] < 30).astype(int)
        self.data['RSI_Overbought'] = (self.data['RSI'] > 70).astype(int)
        self.data['RSI_30_Oversold'] = (self.data['RSI_30'] < 30).astype(int)
        self.data['RSI_30_Overbought'] = (self.data['RSI_30'] > 70).astype(int)
        self.data['RSI_50_Oversold'] = (self.data['RSI_50'] < 30).astype(int)
        self.data['RSI_50_Overbought'] = (self.data['RSI_50'] > 70).astype(int)
        
        # Multi-timeframe Bollinger Band squeezes
        self.data['BB_Squeeze'] = (self.data['BB_Width'] < self.data['BB_Width'].rolling(20).quantile(0.2)).astype(int)
        self.data['BB_Squeeze_Weekly'] = (self.data['BB_Width_Weekly'] < self.data['BB_Width_Weekly'].rolling(50).quantile(0.2)).astype(int)
        self.data['BB_Squeeze_Monthly'] = (self.data['BB_Width_Monthly'] < self.data['BB_Width_Monthly'].rolling(100).quantile(0.2)).astype(int)
        
        # Trend alignments
        self.data['SMA_Alignment_Short'] = ((self.data['SMA_5'] > self.data['SMA_10']) & 
                                           (self.data['SMA_10'] > self.data['SMA_20'])).astype(int)
        self.data['SMA_Alignment_Medium'] = ((self.data['SMA_10'] > self.data['SMA_20']) & 
                                            (self.data['SMA_20'] > self.data['SMA_50'])).astype(int)
        self.data['SMA_Alignment_Long'] = ((self.data['SMA_50'] > self.data['SMA_100']) & 
                                          (self.data['SMA_100'] > self.data['SMA_200'])).astype(int)
        
        # Define feature sets for different timeframes
        self.daily_features = [
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
        
        self.weekly_features = [
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
            'Price_vs_OBV_20', 'MACD_Weekly_Above_Signal', 'BB_Squeeze_Weekly', 'SMA_Alignment_Medium'
        ]
        
        self.biweekly_features = [
            'RSI_30', 'RSI_50', 'RSI_70', 'RSI_100', 'MACD_Monthly', 'MACD_Signal_Monthly', 'MACD_Histogram_Monthly',
            'Momentum_30', 'Momentum_50', 'ROC_30', 'ROC_50', 'ROC_70', 
            'BB_Width_Monthly', 'BB_Position_Monthly', 'ATR_50', 'ATR_100', 'ATR_Ratio_50',
            'ADX_25', 'PLUS_DI', 'MINUS_DI', 'OBV', 'Volume_Ratio_50',
            'Price_Change_14d', 'Price_Change_20d', 'Price_Change_30d', 'Price_Change_50d',
            'Rolling_Volatility_30', 'Rolling_Volatility_50', 'Rolling_Volatility_100',
            'Resistance_Distance_50d', 'Support_Distance_50d', 'Quarter', 'Is_Quarter_End',
            'Close_vs_SMA50', 'Close_vs_SMA100', 'Close_vs_SMA200',
            'Close_vs_EMA50', 'Close_vs_EMA100',
            'Close_vs_BB_Upper_Monthly', 'Close_vs_BB_Lower_Monthly',
            'RSI_50_vs_70', 'RSI_50_vs_30', 'RSI_70_vs_70', 'RSI_70_vs_30', 'RSI_100_vs_70', 'RSI_100_vs_30',
            'Price_vs_OBV_50', 'MACD_Monthly_Above_Signal', 'BB_Squeeze_Monthly', 
            'SMA_Alignment_Long', 'RSI_30_Oversold', 'RSI_30_Overbought', 'RSI_50_Oversold', 'RSI_50_Overbought'
        ]
        
        # Clean features
        all_features = list(set(self.daily_features + self.weekly_features + self.biweekly_features))
        available_features = [f for f in all_features if f in self.data.columns]
        
        self.daily_features = [f for f in self.daily_features if f in available_features]
        self.weekly_features = [f for f in self.weekly_features if f in available_features]
        self.biweekly_features = [f for f in self.biweekly_features if f in available_features]
        
        # Create feature datasets
        self.daily_data = self.data[self.daily_features + ['Daily_Target', 'Next_Day_Return']].dropna()
        self.weekly_data = self.data[self.weekly_features + ['Weekly_Target', 'Next_Week_Return']].dropna()
        self.biweekly_data = self.data[self.biweekly_features + ['BiWeekly_Target', 'Next_BiWeek_Return']].dropna()
        
        return len(self.daily_data) > 100 and len(self.weekly_data) > 100 and len(self.biweekly_data) > 100
    
    def train_tri_models(self):
        """Train separate models for daily, weekly, and bi-weekly predictions including price targets"""
        # Train classification models (direction)
        daily_accuracy = self._train_classification_model(
            self.daily_data, self.daily_features, 'Daily_Target', 
            self.daily_scaler, 'daily_model'
        )
        
        weekly_accuracy = self._train_classification_model(
            self.weekly_data, self.weekly_features, 'Weekly_Target', 
            self.weekly_scaler, 'weekly_model'
        )
        
        biweekly_accuracy = self._train_classification_model(
            self.biweekly_data, self.biweekly_features, 'BiWeekly_Target', 
            self.biweekly_scaler, 'biweekly_model'
        )
        
        # Train regression models (price movement magnitude)
        daily_rmse = self._train_regression_model(
            self.daily_data, self.daily_features, 'Next_Day_Return',
            self.daily_price_scaler, 'daily_price_model'
        )
        
        weekly_rmse = self._train_regression_model(
            self.weekly_data, self.weekly_features, 'Next_Week_Return',
            self.weekly_price_scaler, 'weekly_price_model'
        )
        
        biweekly_rmse = self._train_regression_model(
            self.biweekly_data, self.biweekly_features, 'Next_BiWeek_Return',
            self.biweekly_price_scaler, 'biweekly_price_model'
        )
        
        return {
            'classification': {
                'daily_accuracy': daily_accuracy,
                'weekly_accuracy': weekly_accuracy,
                'biweekly_accuracy': biweekly_accuracy
            },
            'regression': {
                'daily_rmse': daily_rmse,
                'weekly_rmse': weekly_rmse,
                'biweekly_rmse': biweekly_rmse
            }
        }
    
    def _train_classification_model(self, data, features, target, scaler, model_name):
        """Train individual classification model for direction prediction"""
        X = data[features]
        y = data[target]
        
        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
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
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft'
        )
        
        # Train ensemble
        ensemble.fit(X_train_scaled, y_train)
        
        # Set the model
        if model_name == 'daily_model':
            self.daily_model = ensemble
        elif model_name == 'weekly_model':
            self.weekly_model = ensemble
        else:
            self.biweekly_model = ensemble
        
        # Evaluate
        y_pred = ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def _train_regression_model(self, data, features, target, scaler, model_name):
        """Train individual regression model for price movement magnitude prediction"""
        X = data[features]
        y = data[target]
        
        # Remove outliers for better regression performance
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter outliers
        mask = (y >= lower_bound) & (y <= upper_bound)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Time series split
        split_idx = int(len(X_clean) * 0.8)
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Individual regression models
        rf_reg = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=10,
            random_state=42
        )
        
        gb_reg = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        lr_reg = LinearRegression()
        
        # Train models
        rf_reg.fit(X_train_scaled, y_train)
        gb_reg.fit(X_train_scaled, y_train)
        lr_reg.fit(X_train_scaled, y_train)
        
        # Create ensemble
        rf_pred = rf_reg.predict(X_test_scaled)
        gb_pred = gb_reg.predict(X_test_scaled)
        lr_pred = lr_reg.predict(X_test_scaled)
        
        # Simple ensemble average
        ensemble_pred = (rf_pred + gb_pred + lr_pred) / 3
        
        # Store the models
        regression_models = {
            'rf': rf_reg,
            'gb': gb_reg,
            'lr': lr_reg
        }
        
        if model_name == 'daily_price_model':
            self.daily_price_model = regression_models
        elif model_name == 'weekly_price_model':
            self.weekly_price_model = regression_models
        else:
            self.biweekly_price_model = regression_models
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((ensemble_pred - y_test) ** 2))
        
        return rmse
    
    def _predict_price_movement(self, price_models, scaled_features):
        """Predict price movement using ensemble of regression models"""
        rf_pred = price_models['rf'].predict(scaled_features)[0]
        gb_pred = price_models['gb'].predict(scaled_features)[0]
        lr_pred = price_models['lr'].predict(scaled_features)[0]
        
        # Ensemble average
        return (rf_pred + gb_pred + lr_pred) / 3
    
    def _calculate_price_range(self, current_price, predicted_return, timeframe):
        """Calculate price range based on historical volatility"""
        if timeframe == 'daily':
            volatility = self.data['Close'].pct_change().rolling(20).std().iloc[-1]
            confidence_interval = 1.96  # 95% confidence
        elif timeframe == 'weekly':
            volatility = self.data['Close'].pct_change().rolling(50).std().iloc[-1] * np.sqrt(7)
            confidence_interval = 1.96
        else:  # biweekly
            volatility = self.data['Close'].pct_change().rolling(100).std().iloc[-1] * np.sqrt(14)
            confidence_interval = 1.96
        
        # Calculate range
        predicted_price = current_price * (1 + predicted_return)
        price_std = current_price * volatility * confidence_interval
        
        return {
            'target': predicted_price,
            'lower': max(0, predicted_price - price_std),
            'upper': predicted_price + price_std,
            'confidence': 0.95
        }
    
    def get_tri_predictions(self):
        """Get daily, weekly, and bi-weekly predictions with price targets"""
        if (self.daily_model is None or self.weekly_model is None or self.biweekly_model is None or
            self.daily_price_model is None or self.weekly_price_model is None or self.biweekly_price_model is None):
            return None
        
        current_price = self.data['Close'].iloc[-1]
            
        # Daily prediction (direction + price)
        daily_latest = self.daily_data[self.daily_features].iloc[-1:].copy()
        daily_scaled = self.daily_scaler.transform(daily_latest)
        daily_pred = self.daily_model.predict(daily_scaled)[0]
        daily_proba = self.daily_model.predict_proba(daily_scaled)[0]
        
        # Daily price prediction
        daily_price_scaled = self.daily_price_scaler.transform(daily_latest)
        daily_return_pred = self._predict_price_movement(self.daily_price_model, daily_price_scaled)
        daily_price_target = current_price * (1 + daily_return_pred)
        daily_price_range = self._calculate_price_range(current_price, daily_return_pred, 'daily')
        
        # Weekly prediction (direction + price)
        weekly_latest = self.weekly_data[self.weekly_features].iloc[-1:].copy()
        weekly_scaled = self.weekly_scaler.transform(weekly_latest)
        weekly_pred = self.weekly_model.predict(weekly_scaled)[0]
        weekly_proba = self.weekly_model.predict_proba(weekly_scaled)[0]
        
        # Weekly price prediction
        weekly_price_scaled = self.weekly_price_scaler.transform(weekly_latest)
        weekly_return_pred = self._predict_price_movement(self.weekly_price_model, weekly_price_scaled)
        weekly_price_target = current_price * (1 + weekly_return_pred)
        weekly_price_range = self._calculate_price_range(current_price, weekly_return_pred, 'weekly')
        
        # Bi-weekly prediction (direction + price)
        biweekly_latest = self.biweekly_data[self.biweekly_features].iloc[-1:].copy()
        biweekly_scaled = self.biweekly_scaler.transform(biweekly_latest)
        biweekly_pred = self.biweekly_model.predict(biweekly_scaled)[0]
        biweekly_proba = self.biweekly_model.predict_proba(biweekly_scaled)[0]
        
        # Bi-weekly price prediction
        biweekly_price_scaled = self.biweekly_price_scaler.transform(biweekly_latest)
        biweekly_return_pred = self._predict_price_movement(self.biweekly_price_model, biweekly_price_scaled)
        biweekly_price_target = current_price * (1 + biweekly_return_pred)
        biweekly_price_range = self._calculate_price_range(current_price, biweekly_return_pred, 'biweekly')
        
        # Individual model predictions for daily
        daily_individual = {}
        for name, model in self.daily_model.named_estimators_.items():
            pred = model.predict_proba(daily_scaled)[0]
            daily_individual[name] = {
                'prob_down': pred[0],
                'prob_up': pred[1],
                'prediction': 'UP' if pred[1] > 0.5 else 'DOWN'
            }
        
        # Individual model predictions for weekly
        weekly_individual = {}
        for name, model in self.weekly_model.named_estimators_.items():
            pred = model.predict_proba(weekly_scaled)[0]
            weekly_individual[name] = {
                'prob_down': pred[0],
                'prob_up': pred[1],
                'prediction': 'UP' if pred[1] > 0.5 else 'DOWN'
            }
        
        # Individual model predictions for bi-weekly
        biweekly_individual = {}
        for name, model in self.biweekly_model.named_estimators_.items():
            pred = model.predict_proba(biweekly_scaled)[0]
            biweekly_individual[name] = {
                'prob_down': pred[0],
                'prob_up': pred[1],
                'prediction': 'UP' if pred[1] > 0.5 else 'DOWN'
            }
        
        return {
            'daily': {
                'prediction': 'UP' if daily_pred == 1 else 'DOWN',
                'probability_up': daily_proba[1],
                'probability_down': daily_proba[0],
                'confidence': max(daily_proba),
                'price_target': daily_price_target,
                'price_range': daily_price_range,
                'expected_return': daily_return_pred,
                'individual_models': daily_individual
            },
            'weekly': {
                'prediction': 'UP' if weekly_pred == 1 else 'DOWN',
                'probability_up': weekly_proba[1],
                'probability_down': weekly_proba[0],
                'confidence': max(weekly_proba),
                'price_target': weekly_price_target,
                'price_range': weekly_price_range,
                'expected_return': weekly_return_pred,
                'individual_models': weekly_individual
            },
            'biweekly': {
                'prediction': 'UP' if biweekly_pred == 1 else 'DOWN',
                'probability_up': biweekly_proba[1],
                'probability_down': biweekly_proba[0],
                'confidence': max(biweekly_proba),
                'price_target': biweekly_price_target,
                'price_range': biweekly_price_range,
                'expected_return': biweekly_return_pred,
                'individual_models': biweekly_individual
            }
        }
    
    def get_market_context(self):
        """Get current market context and technical levels"""
        if self.data is None:
            return None
            
        latest = self.data.iloc[-1]
        latest_week = self.data.iloc[-5:]
        latest_month = self.data.iloc[-20:]
        
        return {
            'current_price': latest['Close'],
            'volume_vs_avg_10d': latest['Volume_Ratio_10'] if 'Volume_Ratio_10' in latest else None,
            'volume_vs_avg_50d': latest['Volume_Ratio_50'] if 'Volume_Ratio_50' in latest else None,
            'volatility_5d': latest_week['Close'].pct_change().std() * np.sqrt(252),
            'volatility_20d': latest_month['Close'].pct_change().std() * np.sqrt(252),
            'support_5d': latest['Low_5d'] if 'Low_5d' in latest else None,
            'resistance_5d': latest['High_5d'] if 'High_5d' in latest else None,
            'support_20d': latest['Low_20d'] if 'Low_20d' in latest else None,
            'resistance_20d': latest['High_20d'] if 'High_20d' in latest else None,
            'trend_strength': latest['ADX'] if 'ADX' in latest else None,
            'weekly_trend_strength': latest['ADX_25'] if 'ADX_25' in latest else None
        }
    
    def analyze_tri_timeframe(self):
        """Complete tri-timeframe analysis"""
        print(f"Analyzing {self.symbol} for tri-timeframe prediction...")
        
        if not self.fetch_data():
            return None
            
        if not self.calculate_advanced_indicators():
            print("Error calculating technical indicators")
            return None
            
        if not self.create_tri_timeframe_features():
            print("Insufficient data for analysis")
            return None
            
        model_results = self.train_tri_models()
        if not model_results:
            print("Error training models")
            return None
            
        predictions = self.get_tri_predictions()
        market_context = self.get_market_context()
        
        return {
            'symbol': self.symbol,
            'model_performance': model_results,
            'predictions': predictions,
            'market_context': market_context
        }


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tri-Timeframe Stock Analysis (1-day + 7-day + 14-day)')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--period', default='2y', help='Data period (default: 2y)')
    
    args = parser.parse_args()
    
    analyzer = TriTimeframeAnalyzer(args.symbol, args.period)
    result = analyzer.analyze_tri_timeframe()
    
    if result is None:
        print("Analysis failed")
        return
    
    # Display results
    print(f"\n{'='*80}")
    print(f"TRI-TIMEFRAME ANALYSIS: {result['symbol']}")
    print(f"{'='*80}")
    
    print(f"\nMODEL PERFORMANCE:")
    classification = result['model_performance']['classification']
    regression = result['model_performance']['regression']
    print(f"Direction Accuracy - Daily: {classification['daily_accuracy']:.2%} | Weekly: {classification['weekly_accuracy']:.2%} | Bi-Weekly: {classification['biweekly_accuracy']:.2%}")
    print(f"Price Prediction RMSE - Daily: {regression['daily_rmse']:.4f} | Weekly: {regression['weekly_rmse']:.4f} | Bi-Weekly: {regression['biweekly_rmse']:.4f}")
    
    daily_pred = result['predictions']['daily']
    weekly_pred = result['predictions']['weekly']
    biweekly_pred = result['predictions']['biweekly']
    
    context = result['market_context']
    current_price = context['current_price']
    
    print(f"\n🔸 NEXT TRADING DAY PREDICTION:")
    print(f"Direction: {daily_pred['prediction']}")
    print(f"Confidence: {daily_pred['confidence']:.2%}")
    print(f"Expected Return: {daily_pred['expected_return']:+.2%}")
    print(f"Price Target: ${daily_pred['price_target']:.2f}")
    print(f"Price Range: ${daily_pred['price_range']['lower']:.2f} - ${daily_pred['price_range']['upper']:.2f} (95% confidence)")
    
    print(f"\n🔹 NEXT WEEK (7-DAY) PREDICTION:")
    print(f"Direction: {weekly_pred['prediction']}")
    print(f"Confidence: {weekly_pred['confidence']:.2%}")
    print(f"Expected Return: {weekly_pred['expected_return']:+.2%}")
    print(f"Price Target: ${weekly_pred['price_target']:.2f}")
    print(f"Price Range: ${weekly_pred['price_range']['lower']:.2f} - ${weekly_pred['price_range']['upper']:.2f} (95% confidence)")
    
    print(f"\n🔶 NEXT 2-WEEKS (14-DAY) PREDICTION:")
    print(f"Direction: {biweekly_pred['prediction']}")
    print(f"Confidence: {biweekly_pred['confidence']:.2%}")
    print(f"Expected Return: {biweekly_pred['expected_return']:+.2%}")
    print(f"Price Target: ${biweekly_pred['price_target']:.2f}")
    print(f"Price Range: ${biweekly_pred['price_range']['lower']:.2f} - ${biweekly_pred['price_range']['upper']:.2f} (95% confidence)")
    
    print(f"\n📊 MODEL CONSENSUS:")
    print(f"Daily Models Agreement:")
    for model_name, model_pred in daily_pred['individual_models'].items():
        model_display = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'lr': 'Logistic Regression'}
        print(f"  {model_display[model_name]}: {model_pred['prediction']} ({max(model_pred['prob_up'], model_pred['prob_down']):.1%})")
    
    print(f"Weekly Models Agreement:")
    for model_name, model_pred in weekly_pred['individual_models'].items():
        model_display = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'lr': 'Logistic Regression'}
        print(f"  {model_display[model_name]}: {model_pred['prediction']} ({max(model_pred['prob_up'], model_pred['prob_down']):.1%})")
    
    print(f"Bi-Weekly Models Agreement:")
    for model_name, model_pred in biweekly_pred['individual_models'].items():
        model_display = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'lr': 'Logistic Regression'}
        print(f"  {model_display[model_name]}: {model_pred['prediction']} ({max(model_pred['prob_up'], model_pred['prob_down']):.1%})")
    
    print(f"\n📈 MARKET CONTEXT:")
    print(f"Current Price: ${current_price:.2f}")
    if context['volume_vs_avg_10d'] and context['volume_vs_avg_50d']:
        print(f"Volume vs Avg (10d/50d): {context['volume_vs_avg_10d']:.1f}x / {context['volume_vs_avg_50d']:.1f}x")
    print(f"Volatility (5d/20d): {context['volatility_5d']:.1%} / {context['volatility_20d']:.1%}")
    if context['support_5d'] and context['resistance_5d']:
        print(f"Short-term S/R: ${context['support_5d']:.2f} / ${context['resistance_5d']:.2f}")
    if context['support_20d'] and context['resistance_20d']:
        print(f"Medium-term S/R: ${context['support_20d']:.2f} / ${context['resistance_20d']:.2f}")
    
    # Strategy alignment analysis
    predictions = [daily_pred['prediction'], weekly_pred['prediction'], biweekly_pred['prediction']]
    unique_predictions = set(predictions)
    
    if len(unique_predictions) == 1:
        alignment = "✅ FULLY ALIGNED"
        alignment_desc = f"All timeframes predict {predictions[0]} movement - Very strong signal!"
    elif len(unique_predictions) == 2:
        alignment = "⚠️ PARTIALLY DIVERGENT"
        alignment_desc = f"Mixed signals across timeframes - Moderate confidence"
    else:
        alignment = "🔴 FULLY DIVERGENT"
        alignment_desc = f"All timeframes disagree - High uncertainty"
    
    print(f"\n🎯 TIMEFRAME ALIGNMENT: {alignment}")
    print(f"{alignment_desc}")
    
    print(f"\n🔮 PRICE TARGET SUMMARY:")
    print(f"Daily (1d): {daily_pred['prediction']} → ${daily_pred['price_target']:.2f} ({daily_pred['expected_return']:+.1%})")
    print(f"Weekly (7d): {weekly_pred['prediction']} → ${weekly_pred['price_target']:.2f} ({weekly_pred['expected_return']:+.1%})")
    print(f"Bi-Weekly (14d): {biweekly_pred['prediction']} → ${biweekly_pred['price_target']:.2f} ({biweekly_pred['expected_return']:+.1%})")
    
    print(f"\n💡 TRADING INSIGHTS:")
    if daily_pred['prediction'] == weekly_pred['prediction'] == biweekly_pred['prediction']:
        print(f"🚀 Strong consensus: All models agree on {daily_pred['prediction']} trend")
        print(f"   → Consider {daily_pred['prediction'].lower()} positions across all timeframes")
    elif daily_pred['prediction'] == weekly_pred['prediction']:
        print(f"📊 Short-medium consensus: {daily_pred['prediction']} for next week")
        print(f"   → Longer-term outlook: {biweekly_pred['prediction']} reversal expected")
    elif weekly_pred['prediction'] == biweekly_pred['prediction']:
        print(f"📈 Medium-long consensus: {weekly_pred['prediction']} trend developing")
        print(f"   → Short-term volatility: {daily_pred['prediction']} expected tomorrow")
    else:
        print(f"⚡ High volatility period: Different signals across timeframes")
        print(f"   → Consider waiting for clearer directional consensus")


if __name__ == "__main__":
    main()
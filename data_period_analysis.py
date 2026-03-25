#!/usr/bin/env python3
"""
Analyze the composition of different training periods to understand prediction differences
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_period_composition(symbol, periods=['10y', '15y', '20y', '25y']):
    """Analyze data composition for different periods"""
    print(f"📊 DATA PERIOD COMPOSITION ANALYSIS: {symbol}")
    print("=" * 70)
    
    ticker = yf.Ticker(symbol)
    
    for period in periods:
        print(f"\n🔍 {period.upper()} PERIOD ANALYSIS:")
        print("-" * 40)
        
        try:
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"No data available for {period}")
                continue
                
            # Basic stats
            start_date = data.index[0].date()
            end_date = data.index[-1].date()
            total_days = len(data)
            
            print(f"Date Range: {start_date} to {end_date}")
            print(f"Trading Days: {total_days}")
            
            # Price statistics
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            min_price = data['Close'].min()
            max_price = data['Close'].max()
            
            total_return = (end_price - start_price) / start_price
            volatility = data['Close'].pct_change().std() * np.sqrt(252)
            
            print(f"Price Range: ${start_price:.2f} → ${end_price:.2f}")
            print(f"Min/Max: ${min_price:.2f} / ${max_price:.2f}")
            print(f"Total Return: {total_return:.1%}")
            print(f"Annualized Volatility: {volatility:.1%}")
            
            # Market conditions analysis
            returns = data['Close'].pct_change()
            
            # Bull/bear periods
            rolling_return = returns.rolling(252).sum()  # 1-year rolling
            bull_days = (rolling_return > 0.20).sum()  # >20% annual return
            bear_days = (rolling_return < -0.20).sum()  # <-20% annual return
            
            print(f"Bull Market Days (>20% annual): {bull_days} ({bull_days/total_days:.1%})")
            print(f"Bear Market Days (<-20% annual): {bear_days} ({bear_days/total_days:.1%})")
            
            # Volatility periods
            rolling_vol = returns.rolling(60).std() * np.sqrt(252)
            high_vol_days = (rolling_vol > 0.35).sum()  # >35% annual volatility
            low_vol_days = (rolling_vol < 0.15).sum()   # <15% annual volatility
            
            print(f"High Volatility Days (>35%): {high_vol_days} ({high_vol_days/total_days:.1%})")
            print(f"Low Volatility Days (<15%): {low_vol_days} ({low_vol_days/total_days:.1%})")
            
            # Major events/periods identification
            major_events = []
            
            # COVID crash (March 2020)
            covid_period = data[(data.index >= '2020-02-01') & (data.index <= '2020-05-01')]
            if not covid_period.empty:
                covid_drop = (covid_period['Close'].min() - covid_period['Close'].iloc[0]) / covid_period['Close'].iloc[0]
                major_events.append(f"COVID-19 crash: {covid_drop:.1%}")
            
            # 2018 volatility
            vol_2018 = data[(data.index >= '2018-10-01') & (data.index <= '2018-12-31')]
            if not vol_2018.empty:
                drop_2018 = (vol_2018['Close'].min() - vol_2018['Close'].iloc[0]) / vol_2018['Close'].iloc[0]
                major_events.append(f"2018 Q4 drop: {drop_2018:.1%}")
            
            # 2015-2016 correction
            correction_2016 = data[(data.index >= '2015-12-01') & (data.index <= '2016-02-29')]
            if not correction_2016.empty:
                drop_2016 = (correction_2016['Close'].min() - correction_2016['Close'].iloc[0]) / correction_2016['Close'].iloc[0]
                major_events.append(f"2015-16 correction: {drop_2016:.1%}")
            
            if major_events:
                print(f"Major Events Included:")
                for event in major_events:
                    print(f"  • {event}")
            
        except Exception as e:
            print(f"Error analyzing {period}: {e}")

def main():
    analyze_period_composition('NVDA')

if __name__ == "__main__":
    main()
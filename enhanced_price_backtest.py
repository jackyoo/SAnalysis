#!/usr/bin/env python3
"""
Enhanced Backtest Analyzer for Price Prediction Models
Tests both directional accuracy and price target accuracy across multiple timeframes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from tri_timeframe_analyzer import TriTimeframeAnalyzer
import warnings
warnings.filterwarnings('ignore')

class EnhancedPriceBacktest:
    def __init__(self, symbol, test_days=60):
        self.symbol = symbol
        self.test_days = test_days
        self.results = []
        
    def run_backtest(self):
        """Run comprehensive backtest for price predictions"""
        print(f"Running Enhanced Price Backtest for {self.symbol}")
        print(f"Testing period: {self.test_days} days")
        print("=" * 80)
        
        # Get extended data for backtesting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.test_days + 400)  # Extra data for model training
        
        # Download historical data
        stock = yf.Ticker(self.symbol)
        full_data = stock.history(start=start_date, end=end_date)
        
        if len(full_data) < 100:
            print(f"Insufficient data for {self.symbol}")
            return None
            
        print(f"Downloaded {len(full_data)} days of data")
        print(f"Date range: {full_data.index[0].date()} to {full_data.index[-1].date()}")
        
        # Run rolling backtests
        test_start_idx = len(full_data) - self.test_days
        
        for i in range(test_start_idx, len(full_data) - 14):  # Need 14 days ahead for 2-week prediction
            try:
                # Get training data up to current point
                train_data = full_data.iloc[:i+1].copy()
                current_date = train_data.index[-1]
                current_price = train_data['Close'].iloc[-1]
                
                # Create analyzer with training data
                analyzer = TriTimeframeAnalyzer(self.symbol, period='2y')
                analyzer.data = train_data
                
                # Get prediction
                result = analyzer.analyze_tri_timeframe()
                
                if result and 'predictions' in result:
                    predictions = result['predictions']
                    
                    # Get actual future prices
                    actual_1d = full_data['Close'].iloc[i+1] if i+1 < len(full_data) else None
                    actual_7d = full_data['Close'].iloc[i+7] if i+7 < len(full_data) else None
                    actual_14d = full_data['Close'].iloc[i+14] if i+14 < len(full_data) else None
                    
                    # Store results
                    backtest_result = {
                        'date': current_date,
                        'current_price': current_price,
                        'predictions': predictions,
                        'actual_1d': actual_1d,
                        'actual_7d': actual_7d,
                        'actual_14d': actual_14d
                    }
                    
                    self.results.append(backtest_result)
                    
                    if len(self.results) % 10 == 0:
                        print(f"Processed {len(self.results)} predictions...")
                        
            except Exception as e:
                print(f"Error on {current_date}: {e}")
                continue
        
        print(f"\nBacktest completed: {len(self.results)} predictions made")
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze backtest results"""
        if not self.results:
            print("No results to analyze")
            return None
            
        analysis = {
            'total_predictions': len(self.results),
            'timeframes': {}
        }
        
        # Analyze each timeframe
        for timeframe in ['daily', 'weekly', 'biweekly']:
            timeframe_analysis = self.analyze_timeframe(timeframe)
            analysis['timeframes'][timeframe] = timeframe_analysis
            
        # Print results
        self.print_analysis(analysis)
        return analysis
    
    def analyze_timeframe(self, timeframe):
        """Analyze specific timeframe results"""
        if timeframe == 'daily':
            actual_key = 'actual_1d'
            days_ahead = 1
        elif timeframe == 'weekly':
            actual_key = 'actual_7d'
            days_ahead = 7
        else:  # biweekly
            actual_key = 'actual_14d'
            days_ahead = 14
            
        valid_results = [r for r in self.results if r[actual_key] is not None]
        
        if not valid_results:
            return {'valid_predictions': 0}
            
        # Direction accuracy
        correct_directions = 0
        price_errors = []
        return_errors = []
        range_accuracies = []
        
        for result in valid_results:
            pred = result['predictions'][timeframe]
            current_price = result['current_price']
            actual_price = result[actual_key]
            
            # Direction accuracy
            actual_direction = 'UP' if actual_price > current_price else 'DOWN'
            if pred['prediction'] == actual_direction:
                correct_directions += 1
                
            # Price target accuracy
            predicted_price = pred['price_target']
            price_error = abs(actual_price - predicted_price)
            price_errors.append(price_error)
            
            # Return accuracy
            actual_return = (actual_price - current_price) / current_price
            predicted_return = pred['expected_return']
            return_error = abs(actual_return - predicted_return)
            return_errors.append(return_error)
            
            # Range accuracy (if actual price falls within confidence range)
            price_range = pred['price_range']
            in_range = price_range['lower'] <= actual_price <= price_range['upper']
            range_accuracies.append(in_range)
        
        direction_accuracy = correct_directions / len(valid_results)
        avg_price_error = np.mean(price_errors)
        avg_return_error = np.mean(return_errors)
        range_accuracy = np.mean(range_accuracies)
        
        return {
            'valid_predictions': len(valid_results),
            'direction_accuracy': direction_accuracy,
            'avg_price_error': avg_price_error,
            'avg_return_error': avg_return_error,
            'range_accuracy': range_accuracy,
            'price_errors': price_errors,
            'return_errors': return_errors
        }
    
    def print_analysis(self, analysis):
        """Print detailed analysis results"""
        print("\n" + "=" * 80)
        print("ENHANCED PRICE BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"Total Predictions Made: {analysis['total_predictions']}")
        
        for timeframe_name, data in analysis['timeframes'].items():
            if data['valid_predictions'] == 0:
                continue
                
            print(f"\n📊 {timeframe_name.upper()} PREDICTIONS:")
            print(f"   Valid Predictions: {data['valid_predictions']}")
            print(f"   🎯 Direction Accuracy: {data['direction_accuracy']:.1%}")
            print(f"   💰 Avg Price Error: ${data['avg_price_error']:.2f}")
            print(f"   📈 Avg Return Error: {data['avg_return_error']:.1%}")
            print(f"   🎚️  Range Accuracy: {data['range_accuracy']:.1%}")
            
            # Additional statistics
            price_errors = data['price_errors']
            if price_errors:
                print(f"   Price Error Std: ${np.std(price_errors):.2f}")
                print(f"   Max Price Error: ${np.max(price_errors):.2f}")
                print(f"   Min Price Error: ${np.min(price_errors):.2f}")
        
        # Trading simulation
        self.simulate_trading(analysis)
    
    def simulate_trading(self, analysis):
        """Simulate trading strategy using predictions"""
        print(f"\n💹 TRADING SIMULATION:")
        print("-" * 40)
        
        initial_capital = 10000
        capital = initial_capital
        trades = 0
        winning_trades = 0
        
        for result in self.results[:30]:  # Use first 30 for daily trading simulation
            if result['actual_1d'] is None:
                continue
                
            pred = result['predictions']['daily']
            current_price = result['current_price']
            actual_price = result['actual_1d']
            confidence = pred['confidence']
            
            # Trade only if confidence > 55%
            if confidence > 0.55:
                trades += 1
                
                if pred['prediction'] == 'UP':
                    # Buy signal
                    shares = capital / current_price
                    new_capital = shares * actual_price
                else:
                    # Short signal (simplified as inverse return)
                    return_pct = (actual_price - current_price) / current_price
                    new_capital = capital * (1 - return_pct)
                
                if new_capital > capital:
                    winning_trades += 1
                    
                capital = new_capital
        
        if trades > 0:
            total_return = (capital - initial_capital) / initial_capital
            win_rate = winning_trades / trades
            
            print(f"   Initial Capital: ${initial_capital:,.2f}")
            print(f"   Final Capital: ${capital:,.2f}")
            print(f"   Total Return: {total_return:.1%}")
            print(f"   Trades Made: {trades}")
            print(f"   Win Rate: {win_rate:.1%}")
        else:
            print("   No trades made (insufficient confidence)")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Price Backtest Analyzer')
    parser.add_argument('symbol', help='Stock symbol (e.g., NVDA, AAPL)')
    parser.add_argument('--days', type=int, default=60, help='Number of days to backtest')
    
    args = parser.parse_args()
    
    backtest = EnhancedPriceBacktest(args.symbol, args.days)
    backtest.run_backtest()

if __name__ == "__main__":
    main()
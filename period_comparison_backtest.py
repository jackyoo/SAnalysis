#!/usr/bin/env python3
"""
Compare different training periods for tri-timeframe analyzer accuracy
Tests 10y, 15y, 20y, 25y, 30y periods against historical data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from tri_timeframe_analyzer import TriTimeframeAnalyzer
import warnings
warnings.filterwarnings('ignore')

class PeriodComparisonBacktest:
    def __init__(self, symbol, test_days=30):
        self.symbol = symbol
        self.test_days = test_days
        self.periods = ['25y', '30y']  # Test remaining periods
        self.results = {}
        
    def run_comparison(self):
        """Run backtest comparison across different training periods"""
        print(f"Comparing Training Periods for {self.symbol}")
        print(f"Test period: {self.test_days} days")
        print("=" * 80)
        
        for period in self.periods:
            print(f"\n🔄 Testing with {period} training period...")
            result = self.test_period(period)
            self.results[period] = result
            
        self.analyze_comparison()
        return self.results
    
    def test_period(self, period):
        """Test a specific training period"""
        try:
            # Get extended data for backtesting
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.test_days + 500)
            
            # Download historical data
            stock = yf.Ticker(self.symbol)
            full_data = stock.history(start=start_date, end=end_date)
            
            if len(full_data) < 100:
                return None
                
            # Run rolling backtests
            test_start_idx = len(full_data) - self.test_days
            predictions = []
            
            for i in range(test_start_idx, len(full_data) - 14, 5):  # Every 5 days to speed up
                try:
                    # Get training data up to current point
                    train_data = full_data.iloc[:i+1].copy()
                    current_date = train_data.index[-1]
                    current_price = train_data['Close'].iloc[-1]
                    
                    # Create analyzer with specific period
                    analyzer = TriTimeframeAnalyzer(self.symbol, period=period)
                    analyzer.data = train_data
                    
                    # Get prediction
                    result = analyzer.analyze_tri_timeframe()
                    
                    if result and 'predictions' in result:
                        pred = result['predictions']
                        
                        # Get actual future prices
                        actual_1d = full_data['Close'].iloc[i+1] if i+1 < len(full_data) else None
                        actual_7d = full_data['Close'].iloc[i+7] if i+7 < len(full_data) else None
                        actual_14d = full_data['Close'].iloc[i+14] if i+14 < len(full_data) else None
                        
                        prediction_result = {
                            'date': current_date,
                            'current_price': current_price,
                            'predictions': pred,
                            'actual_1d': actual_1d,
                            'actual_7d': actual_7d,
                            'actual_14d': actual_14d
                        }
                        
                        predictions.append(prediction_result)
                        
                except Exception as e:
                    continue
            
            return self.analyze_period_results(predictions)
            
        except Exception as e:
            print(f"Error testing {period}: {e}")
            return None
    
    def analyze_period_results(self, predictions):
        """Analyze results for a specific period"""
        if not predictions:
            return None
            
        # Calculate accuracies for each timeframe
        timeframe_results = {}
        
        for timeframe, actual_key in [('daily', 'actual_1d'), ('weekly', 'actual_7d'), ('biweekly', 'actual_14d')]:
            valid_preds = [p for p in predictions if p[actual_key] is not None]
            
            if not valid_preds:
                continue
                
            correct_directions = 0
            price_errors = []
            return_errors = []
            range_hits = 0
            
            for pred in valid_preds:
                current_price = pred['current_price']
                actual_price = pred[actual_key]
                model_pred = pred['predictions'][timeframe]
                
                # Direction accuracy
                actual_direction = 'UP' if actual_price > current_price else 'DOWN'
                if model_pred['prediction'] == actual_direction:
                    correct_directions += 1
                
                # Price accuracy
                predicted_price = model_pred['price_target']
                price_error = abs(actual_price - predicted_price)
                price_errors.append(price_error)
                
                # Return accuracy
                actual_return = (actual_price - current_price) / current_price
                predicted_return = model_pred['expected_return']
                return_error = abs(actual_return - predicted_return)
                return_errors.append(return_error)
                
                # Range accuracy
                price_range = model_pred['price_range']
                if price_range['lower'] <= actual_price <= price_range['upper']:
                    range_hits += 1
            
            timeframe_results[timeframe] = {
                'valid_predictions': len(valid_preds),
                'direction_accuracy': correct_directions / len(valid_preds) if valid_preds else 0,
                'avg_price_error': np.mean(price_errors) if price_errors else float('inf'),
                'avg_return_error': np.mean(return_errors) if return_errors else float('inf'),
                'range_accuracy': range_hits / len(valid_preds) if valid_preds else 0
            }
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(timeframe_results)
        
        return {
            'timeframes': timeframe_results,
            'overall_score': overall_score,
            'total_predictions': len(predictions)
        }
    
    def calculate_overall_score(self, timeframe_results):
        """Calculate overall performance score"""
        if not timeframe_results:
            return 0
            
        total_score = 0
        count = 0
        
        for timeframe, results in timeframe_results.items():
            if results['valid_predictions'] > 0:
                # Weight: direction accuracy (40%) + range accuracy (40%) + price accuracy (20%)
                direction_score = results['direction_accuracy']
                range_score = results['range_accuracy']
                price_score = max(0, 1 - (results['avg_price_error'] / 50))  # Normalize price error
                
                timeframe_score = (direction_score * 0.4 + range_score * 0.4 + price_score * 0.2)
                total_score += timeframe_score
                count += 1
        
        return total_score / count if count > 0 else 0
    
    def analyze_comparison(self):
        """Analyze and display comparison results"""
        print("\n" + "=" * 80)
        print("TRAINING PERIOD COMPARISON RESULTS")
        print("=" * 80)
        
        # Sort by overall score
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['overall_score'] if x[1] else 0, 
                              reverse=True)
        
        print(f"\n📊 RANKING BY OVERALL PERFORMANCE:")
        print("-" * 50)
        
        for rank, (period, result) in enumerate(sorted_results, 1):
            if result is None:
                print(f"{rank}. {period}: ❌ Failed")
                continue
                
            print(f"{rank}. {period}: 🏆 Score {result['overall_score']:.3f}")
            print(f"   Total Predictions: {result['total_predictions']}")
            
            for timeframe, data in result['timeframes'].items():
                if data['valid_predictions'] > 0:
                    print(f"   {timeframe.capitalize()}: "
                          f"Dir {data['direction_accuracy']:.1%} | "
                          f"Range {data['range_accuracy']:.1%} | "
                          f"Price Err ${data['avg_price_error']:.2f}")
            print()
        
        # Best period
        if sorted_results and sorted_results[0][1]:
            best_period = sorted_results[0][0]
            best_score = sorted_results[0][1]['overall_score']
            
            print(f"🥇 BEST TRAINING PERIOD: {best_period}")
            print(f"   Overall Score: {best_score:.3f}")
            print(f"   Recommendation: Use {best_period} period for most accurate predictions")
            
            return best_period
        
        return None

def main():
    parser = argparse.ArgumentParser(description='Period Comparison Backtest')
    parser.add_argument('symbol', help='Stock symbol (e.g., NVDA, AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    
    args = parser.parse_args()
    
    backtest = PeriodComparisonBacktest(args.symbol, args.days)
    backtest.run_comparison()

if __name__ == "__main__":
    main()
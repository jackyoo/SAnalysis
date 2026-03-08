#!/usr/bin/env python3
"""
Example usage of the Stock Analyzer
"""

from stock_analyzer import StockAnalyzer

def example_analysis():
    """Example analysis for popular stocks"""
    symbols = ['AAPL', 'TSLA', 'MSFT']
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Analyzing {symbol}")
        print(f"{'='*60}")
        
        try:
            analyzer = StockAnalyzer(symbol)
            result = analyzer.analyze()
            
            if result:
                pred = result['prediction']
                indicators = result['indicators']
                
                print(f"Model Accuracy: {result['model_accuracy']:.2%}")
                print(f"Prediction: {pred['prediction']} ({pred['confidence']:.2%} confidence)")
                print(f"Probability UP: {pred['probability_up']:.2%}")
                print(f"RSI: {indicators['RSI']:.2f}")
                print(f"Price vs SMA(20): {indicators['Close_vs_SMA20']:+.2%}")
            else:
                print(f"Failed to analyze {symbol}")
                
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

if __name__ == "__main__":
    example_analysis()
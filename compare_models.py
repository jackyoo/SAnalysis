#!/usr/bin/env python3
"""
Compare original vs enhanced stock analyzer models
"""

from stock_analyzer import StockAnalyzer
from enhanced_stock_analyzer import EnhancedStockAnalyzer

def compare_models(symbol, period='1y'):
    """Compare original and enhanced models for a stock"""
    print(f"Comparing models for {symbol}...")
    
    # Original model
    print("\n" + "="*50)
    print("ORIGINAL MODEL")
    print("="*50)
    
    try:
        analyzer = StockAnalyzer(symbol, period)
        original_result = analyzer.analyze()
        
        if original_result:
            pred = original_result['prediction']
            print(f"Prediction: {pred['prediction']}")
            print(f"Confidence: {pred['confidence']:.2%}")
            print(f"Model Accuracy: {original_result['model_accuracy']:.2%}")
        else:
            print("Original analysis failed")
            
    except Exception as e:
        print(f"Original model error: {e}")
    
    # Enhanced model
    print("\n" + "="*50)
    print("ENHANCED MODEL (Next-Day Focus)")
    print("="*50)
    
    try:
        enhanced_analyzer = EnhancedStockAnalyzer(symbol, period)
        enhanced_result = enhanced_analyzer.analyze_for_next_day()
        
        if enhanced_result:
            pred = enhanced_result['prediction']
            print(f"Next-Day Prediction: {pred['prediction']}")
            print(f"Confidence: {pred['confidence']:.2%}")
            print(f"Overall Accuracy: {enhanced_result['model_accuracy']:.2%}")
            print(f"Recent Accuracy: {enhanced_result['recent_accuracy']:.2%}")
            print(f"Model Consensus:")
            for model_name, model_pred in pred['individual_models'].items():
                print(f"  - {model_name}: {model_pred['prediction']} ({max(model_pred['prob_up'], model_pred['prob_down']):.1%})")
        else:
            print("Enhanced analysis failed")
            
    except Exception as e:
        print(f"Enhanced model error: {e}")

def main():
    """Compare models for multiple stocks"""
    stocks = ['AAPL', 'TSLA', 'MSFT']
    
    for symbol in stocks:
        compare_models(symbol)
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    main()
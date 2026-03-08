#!/usr/bin/env python3
"""
Final validation of best performing periods with more comprehensive testing
"""

import subprocess
import sys

def run_backtest(period, days=30):
    """Run enhanced backtest for a specific period"""
    print(f"\n🔍 Testing {period} period with {days} days backtest...")
    
    # Temporarily modify tri_timeframe_analyzer to use specific period
    cmd = f"source venv/bin/activate && python enhanced_price_backtest.py NVDA --days {days}"
    
    # We'll need to create a modified version that accepts period parameter
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    return result.stdout, result.stderr

def main():
    """Compare top performing periods with more data"""
    top_periods = ['10y', '25y', '30y']
    
    print("🚀 FINAL PERIOD VALIDATION FOR NVDA")
    print("=" * 60)
    print("Comparing top 3 periods from initial testing:")
    print("- 10y: Score 0.555")
    print("- 25y: Score 0.626") 
    print("- 30y: Score 0.588")
    
    results = {}
    
    for period in top_periods:
        print(f"\n{'='*60}")
        print(f"TESTING {period} PERIOD")
        print(f"{'='*60}")
        
        # Create temporary analyzer with specific period
        analyzer_content = f'''
from tri_timeframe_analyzer import TriTimeframeAnalyzer

# Test with {period}
analyzer = TriTimeframeAnalyzer('NVDA', period='{period}')
result = analyzer.analyze_tri_timeframe()

if result:
    predictions = result['predictions']
    daily = predictions['daily']
    weekly = predictions['weekly']
    biweekly = predictions['biweekly']
    
    print(f"\\n📊 {period} PERIOD RESULTS:")
    print(f"Daily: {{daily['prediction']}} ({{daily['confidence']:.1%}}) - ${{daily['price_target']:.2f}}")
    print(f"Weekly: {{weekly['prediction']}} ({{weekly['confidence']:.1%}}) - ${{weekly['price_target']:.2f}}")
    print(f"Biweekly: {{biweekly['prediction']}} ({{biweekly['confidence']:.1%}}) - ${{biweekly['price_target']:.2f}}")
else:
    print("Analysis failed")
'''
        
        with open(f'/tmp/test_{period}.py', 'w') as f:
            f.write(analyzer_content)
            
        cmd = f"cd /Users/sejungyoo/workspace/StockAnalysis && source venv/bin/activate && python /tmp/test_{period}.py"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")

if __name__ == "__main__":
    main()
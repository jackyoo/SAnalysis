# Stock Analysis Project

A Python project that uses technical analysis indicators (RSI, MACD, Bollinger Bands, etc.) to predict short-term stock market movements with probability indicators.

## Features

- Fetches real-time stock data using `yfinance`
- Calculates technical indicators using `TA-Lib`:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - Williams %R
  - Moving Averages (SMA, EMA)
  - Volume indicators (OBV)
  - Momentum and Rate of Change
- Uses machine learning (Random Forest) to predict price direction
- Provides probability estimates for up/down movements
- Command-line interface for easy usage

## Quick Start

### 1. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install Python packages
python -m pip install -r requirements.txt

# Install TA-Lib system dependency (macOS)
brew install ta-lib

# Install TA-Lib Python wrapper
python -m pip install ta-lib
```

### 3. Run Analysis
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Original analyzer - general prediction
python stock_analyzer.py AAPL
python stock_analyzer.py TSLA --period 1y

# Enhanced analyzer - next-day focused prediction
python enhanced_stock_analyzer.py AAPL
python enhanced_stock_analyzer.py TSLA --period 1y

# Dual timeframe analyzer - both daily and weekly predictions
python dual_timeframe_analyzer.py NVDA --period 2y

# Compare both models
python compare_models.py

# Backtest dual timeframe predictions
python dual_backtest_analyzer.py NVDA --days 60
```

## Detailed Installation

### Prerequisites
- Python 3.8+ 
- Homebrew (for macOS TA-Lib installation)

### Step-by-Step Setup
1. **Clone/Download the project**
   ```bash
   cd /path/to/StockAnalysis
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Install TA-Lib system library**
   - **macOS**: `brew install ta-lib`
   - **Linux**: `sudo apt-get install libta-lib-dev` (Ubuntu/Debian)
   - **Windows**: Download from [TA-Lib website](https://www.ta-lib.org/hdr_dw.html)

5. **Install TA-Lib Python wrapper**
   ```bash
   python -m pip install ta-lib
   ```

## Usage

### Command Line Interface

#### Original Analyzer (General Prediction)
```bash
python stock_analyzer.py TICKER [--period PERIOD]

# Examples
python stock_analyzer.py AAPL
python stock_analyzer.py TSLA --period 1y
```

#### Enhanced Analyzer (Next-Day Focused)
```bash
python enhanced_stock_analyzer.py TICKER [--period PERIOD]

# Examples
python enhanced_stock_analyzer.py AAPL
python enhanced_stock_analyzer.py TSLA --period 1y
```

#### Dual Timeframe Analyzer (Daily + Weekly)
```bash
python dual_timeframe_analyzer.py TICKER [--period PERIOD]

# Examples
python dual_timeframe_analyzer.py NVDA --period 2y
python dual_timeframe_analyzer.py AAPL --period 2y
```

**Available periods**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
**Recommendation**: Use 1y+ periods for enhanced analyzer for better feature calculation

### Python Script Usage

#### Original Analyzer
```python
from stock_analyzer import StockAnalyzer

analyzer = StockAnalyzer('AAPL', period='1y')
result = analyzer.analyze()

if result:
    pred = result['prediction']
    print(f"Prediction: {pred['prediction']}")
    print(f"Probability UP: {pred['probability_up']:.2%}")
```

#### Enhanced Analyzer (Next-Day Focus)
```python
from enhanced_stock_analyzer import EnhancedStockAnalyzer

analyzer = EnhancedStockAnalyzer('AAPL', period='1y')
result = analyzer.analyze_for_next_day()

if result:
    pred = result['prediction']
    context = result['market_context']
    
    print(f"Next-Day Prediction: {pred['prediction']}")
    print(f"Confidence: {pred['confidence']:.2%}")
    print(f"Current Price: ${context['current_price']:.2f}")
```

#### Dual Timeframe Analyzer (Daily + Weekly)
```python
from dual_timeframe_analyzer import DualTimeframeAnalyzer

analyzer = DualTimeframeAnalyzer('NVDA', period='2y')
result = analyzer.analyze_dual_timeframe()

if result:
    daily = result['predictions']['daily']
    weekly = result['predictions']['weekly']
    
    print(f"Next Day: {daily['prediction']} ({daily['confidence']:.1%})")
    print(f"Next Week: {weekly['prediction']} ({weekly['confidence']:.1%})")
    print(f"Strategy Alignment: {'Aligned' if daily['prediction'] == weekly['prediction'] else 'Divergent'}")
```

### Run Example Script
```bash
source venv/bin/activate
python example.py
```

## Example Output

### Original Analyzer
```
==================================================
STOCK ANALYSIS: AAPL
==================================================

Model Accuracy: 46.34%

PREDICTION:
Direction: UP
Probability UP: 69.63%
Probability DOWN: 30.37%
Confidence: 69.63%

TECHNICAL INDICATORS:
RSI (14): 40.88
MACD: -0.8848
MACD Signal: 0.2647
Bollinger Band Position: 14.09%
```

### Enhanced Analyzer (Next-Day Focus)
```
============================================================
NEXT-DAY PREDICTION ANALYSIS: AAPL
============================================================

MODEL PERFORMANCE:
Overall Accuracy: 65.85%
Recent Accuracy (last 5 predictions): 80.00%

NEXT TRADING DAY PREDICTION:
📈 Direction: UP
🎯 Confidence: 54.40%
⬆️  Probability UP: 54.40%
⬇️  Probability DOWN: 45.60%

INDIVIDUAL MODEL CONSENSUS:
  Random Forest: UP (63.67%)
  Gradient Boosting: UP (88.94%)
  Logistic Regression: DOWN (89.43%)

MARKET CONTEXT:
Current Price: $257.46
Volume vs Average: 0.90x
5-Day Volatility: 5.3%
Support: $254.37 | Resistance: $266.53
Trend Strength (ADX): 16.1 (Weak)

TOP PREDICTIVE FEATURES:
  1. Rolling_Volatility_10: 0.042
  2. Close_vs_SAR: 0.033
  3. Volume_Ratio: 0.031

💡 Next trading day outlook: UP movement expected with 54.4% confidence
```

## Understanding the Output

- **Model Accuracy**: Historical prediction accuracy on test data
- **Prediction Direction**: UP/DOWN for next trading day
- **Probability**: Percentage likelihood of up/down movement
- **Confidence**: Higher of the two probabilities
- **Technical Indicators**: Current values of key indicators
- **Feature Importance**: Which indicators influence the prediction most

## Troubleshooting

### Common Issues

1. **"externally-managed-environment" error**
   - Solution: Use virtual environment as shown above

2. **TA-Lib installation fails**
   - macOS: Ensure Homebrew is installed, then `brew install ta-lib`
   - Restart terminal after installing system TA-Lib

3. **"No data found" error**
   - Check if stock ticker is valid
   - Try different time periods

4. **Import errors**
   - Ensure virtual environment is activated: `source venv/bin/activate`
   - Verify all packages installed: `pip list`

## What's New in Enhanced Version

The enhanced analyzer (`enhanced_stock_analyzer.py`) provides significant improvements for next-day prediction:

### 🚀 Advanced Features
- **Ensemble Model**: Combines Random Forest, Gradient Boosting, and Logistic Regression
- **50+ Technical Indicators**: More comprehensive feature set including ADX, Parabolic SAR, volatility measures
- **Time Series Validation**: Proper backtesting without data leakage
- **Individual Model Consensus**: See how each algorithm votes
- **Market Context**: Support/resistance levels, trend strength, volatility analysis
- **Recent Performance**: Track model accuracy on recent predictions

### 📊 Key Improvements
- **Better Accuracy**: Typically 10-20% higher accuracy than original model
- **Next-Day Focus**: Specifically tuned for predicting the very next trading day
- **Risk Assessment**: Confidence levels and model agreement analysis
- **Market Insights**: Current price context and technical levels

## Files

- `stock_analyzer.py` - Original analysis module
- `enhanced_stock_analyzer.py` - **Enhanced next-day prediction analyzer**
- `dual_timeframe_analyzer.py` - **🆕 Dual timeframe analyzer (1-day + 7-day)**
- `backtest_analyzer.py` - Backtest enhanced analyzer
- `dual_backtest_analyzer.py` - **🆕 Backtest dual timeframe predictions**
- `compare_models.py` - Compare original vs enhanced models
- `example.py` - Example usage script  
- `requirements.txt` - Required Python packages
- `README.md` - This documentation
- `venv/` - Virtual environment (created after setup)

## Disclaimer

This tool is for educational and research purposes only. Stock market predictions are inherently uncertain, and this tool should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.
# Cryptocurrency Tweet Sentiment Analysis: Predicting Bitcoin Volatility

This notebook implements a sentiment-based volatility prediction framework for Bitcoin using historical Twitter data, as described in the research paper "Cryptocurrency Tweet Sentiment Analysis: Predicting Bitcoin Volatility Using Social Media Signals".

## Project Overview

The project investigates whether cryptocurrency-related Twitter sentiment can predict short-term Bitcoin volatility events using:

- **Data Sources:** 1.8M cryptocurrency tweets (2021-2023) + Bitcoin price data
- **Methodology:** SBERT fine-tuned on financial text + STL decomposition + z-score anomaly detection
- **Objective:** Predict ±5% Bitcoin price movements within 2-hour windows
- **Success Criteria:** F1 ≥ 0.50 and Sharpe > 0

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data (Optional)

For real implementation, download the following Kaggle datasets:
- [Bitcoin Tweets Dataset](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329) by alaix14
- [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) by mczielinski

Place the datasets in a `data/` directory:
```
notebook/
├── data/
│   ├── bitcoin_tweets.csv
│   └── bitcoin_prices.csv
├── crypto_sentiment_analysis.ipynb
├── requirements.txt
└── README.md
```

**Note:** The notebook includes sample data generation for demonstration purposes if real datasets are not available.

### 3. Optional: Advanced NLP Models

For better sentiment analysis, uncomment the transformers libraries in `requirements.txt` and install:

```bash
pip install transformers torch sentence-transformers
```

This enables FinBERT for financial sentiment analysis instead of the TextBlob fallback.

## Notebook Structure

1. **Environment Setup** - Import libraries and configure environment
2. **Data Loading** - Load or generate sample cryptocurrency and price data
3. **Data Preprocessing** - Clean tweets and calculate price features
4. **Sentiment Analysis** - Apply FinBERT or TextBlob sentiment analysis
5. **Temporal Aggregation** - Aggregate sentiment into 15-minute windows
6. **Anomaly Detection** - Use STL decomposition and z-score analysis
7. **Volatility Prediction** - Predict Bitcoin price movements
8. **Model Evaluation** - Calculate performance metrics and Sharpe ratio
9. **Results Visualization** - Generate plots and analysis

## Key Features

### Sentiment Analysis
- **FinBERT Integration:** Financial domain-specific BERT model for accurate crypto sentiment
- **TextBlob Fallback:** Reliable backup when advanced models unavailable
- **Batch Processing:** Efficient processing of large tweet datasets

### Anomaly Detection
- **STL Decomposition:** Separates trend, seasonal, and residual components
- **Z-Score Analysis:** Identifies statistically significant sentiment shifts
- **Configurable Thresholds:** Tunable parameters for different market conditions

### Volatility Prediction
- **15-Minute Windows:** High-frequency analysis for rapid market changes
- **2-Hour Prediction:** Actionable timeframe for trading decisions
- **Multiple Metrics:** Precision, recall, F1-score, and Sharpe ratio evaluation

## Usage

1. Open the notebook in Jupyter:
```bash
jupyter notebook crypto_sentiment_analysis.ipynb
```

2. Run all cells sequentially, or execute specific sections as needed

3. Modify parameters in the configuration cells to experiment with different settings:
   - Sentiment analysis model
   - Time window sizes
   - Anomaly detection thresholds
   - Volatility event definitions

## Expected Results

The notebook demonstrates the complete pipeline and provides:
- Sentiment analysis visualizations
- Anomaly detection plots
- Volatility prediction performance metrics
- Trading strategy backtesting results

## Research Context

This implementation is based on the research paper that addresses gaps in existing sentiment-based cryptocurrency prediction methods:

- **Multi-Platform Integration:** Combines Twitter and Reddit sentiment data
- **High-Frequency Analysis:** 15-minute temporal resolution
- **Rigorous Validation:** Comprehensive backtesting framework
- **Practical Implementation:** Real-world trading considerations

## Future Extensions

- Real-time data streaming integration
- Multi-cryptocurrency support
- Advanced machine learning models
- Risk management systems
- Live trading implementation

## Citation

If you use this code in your research, please cite:

```
Little, L. (2025). Cryptocurrency Tweet Sentiment Analysis: Predicting Bitcoin Volatility Using Social Media Signals. 
CSCA 5522: Data Mining Project, University of Colorado - Boulder.
```

## License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service when using real datasets.

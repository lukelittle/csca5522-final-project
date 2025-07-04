# Cryptocurrency Tweet Sentiment Analysis: Predicting Bitcoin Volatility Using Social Media Signals

**Author:** Lucas Little  
**Course:** CSCA 5522: Data Mining Project  
**University:** University of Colorado - Boulder  
**Contact:** lucas.little@colorado.edu

## Abstract

This project conducts a comprehensive analysis to determine whether short-term Bitcoin volatility events can be predicted using cryptocurrency-related sentiment analysis from Twitter. The research utilizes a comprehensive dataset containing cryptocurrency-related tweets alongside high-frequency Bitcoin price data to develop and validate a volatility prediction framework. The methodology employs advanced NLP techniques including FinBERT for sentiment analysis, comprehensive feature engineering, and machine learning models to predict market volatility events.

## Project Overview

Cryptocurrency markets are known for their extreme volatility and unpredictability, with retail investors having significant influence through social media sentiment. This research investigates whether collective sentiment expressed on Twitter possesses predictive power for Bitcoin volatility events, focusing on short-term (2-hour) prediction windows.

### Research Questions

1. **Can sentiment anomalies in cryptocurrency Twitter discussions reliably precede Bitcoin volatility events within a 2-hour window?**
2. **What temporal resolution optimally captures sentiment signals for volatility prediction?**
3. **Can sentiment-based approaches outperform traditional technical indicators?**
4. **How effective is the combination of anomaly detection and feature engineering for distinguishing meaningful sentiment patterns?**

## Dataset Description

### Bitcoin Tweets Dataset
- **Source**: Kaggle "Bitcoin Tweets" dataset
- **Size**: 16 million cryptocurrency-related tweets
- **Timeframe**: January 1, 2016 to March 29, 2019
- **Features**: Tweet text, timestamps, user metadata, engagement metrics

### Bitcoin Historical Price Data
- **Source**: Kaggle "Bitcoin Historical Data" dataset
- **Resolution**: High-frequency OHLCV data with 1-minute resolution
- **Features**: Open, High, Low, Close prices, Volume, market indicators
- **Coverage**: Comprehensive market data aligned with tweet timeframe

## Methodology

### 1. Data Processing and Preprocessing
- **Text Normalization**: Cleaning and standardizing tweet text
- **Timestamp Synchronization**: Aligning social media and price data
- **Data Quality Assurance**: Filtering spam, bots, and low-quality content
- **Missing Data Handling**: Interpolation strategies for data continuity

### 2. Sentiment Analysis
- **Primary Method**: FinBERT (Financial domain-specific BERT model)
- **Fallback Method**: TextBlob for baseline comparison
- **Temporal Aggregation**: 15-minute and hourly sentiment windows
- **Metrics**: Mean sentiment, sentiment volatility, tweet volume, engagement

### 3. Technical Analysis and Feature Engineering
- **Price Features**: Returns, volatility, price ranges, OHLCV metrics
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages, ATR
- **Volume Indicators**: On-Balance Volume, volume ratios, price-volume interactions
- **Lag Features**: Historical sentiment and price patterns
- **Rolling Features**: Moving averages and volatility measures

### 4. Advanced Feature Engineering
- **Anomaly Detection**: Isolation Forest and Z-score based anomaly identification
- **Market Regime Detection**: Bull, bear, and sideways market classification
- **Interaction Features**: Price-sentiment, volume-sentiment combinations
- **Time Features**: Cyclical encoding of hour, day, month patterns
- **Momentum Features**: Price and sentiment momentum indicators

### 5. Machine Learning Models
- **Fast Models**: Random Forest, Gradient Boosting, Logistic Regression
- **SVM Variants**: Linear SVM, SGD Classifier for scalability
- **Evaluation**: Time series cross-validation, AUC scoring
- **Target Variables**: High volatility events, directional movements

### 6. Backtesting and Validation
- **Strategy Implementation**: Volatility-based trading signals
- **Performance Metrics**: Sharpe ratio, total returns, win rates
- **Risk Management**: Drawdown analysis, volatility assessment
- **Comparison**: Strategy vs. market performance

## Project Structure

```
├── notebooks/
│   ├── 01_data_download_and_preprocessing.ipynb    # Data acquisition and cleaning
│   ├── 02_sentiment_analysis.ipynb                 # FinBERT sentiment analysis
│   ├── 03_price_data_processing_and_alignment.ipynb # Technical indicators and alignment
│   ├── 04_anomaly_detection_and_feature_engineering.ipynb # Advanced feature creation
│   ├── 05_modeling_and_backtesting.ipynb          # ML models and backtesting
│   ├── 06_visualization_and_results.ipynb         # Results analysis and visualization
│   ├── data/
│   │   ├── raw/                                    # Original datasets
│   │   └── processed/                              # Processed and engineered features
│   └── requirements.txt                            # Notebook-specific dependencies
├── requirements.txt                                # Project dependencies
├── .gitignore                                      # Git ignore rules
├── proposal.tex                                    # Academic proposal document
└── README.md                                       # This file
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- CUDA-compatible GPU (optional, for faster transformer models)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/lukelittle/csca5522-final-project.git
cd csca5522-final-project
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter:**
```bash
jupyter notebook
```

### Key Dependencies
- **Data Science**: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
- **NLP**: transformers, torch, sentence-transformers, textblob, nltk
- **Technical Analysis**: ta (Technical Analysis library)
- **Time Series**: statsmodels
- **Machine Learning**: joblib
- **Utilities**: tqdm

## Usage

### Running the Analysis

Execute the notebooks in order:

1. **Data Preprocessing** (`01_data_download_and_preprocessing.ipynb`)
   - Download and clean Bitcoin tweets and price data
   - Perform initial data exploration and quality assessment

2. **Sentiment Analysis** (`02_sentiment_analysis.ipynb`)
   - Apply FinBERT sentiment analysis to tweets
   - Create temporal sentiment aggregations
   - Compare with baseline TextBlob sentiment

3. **Price Data Processing** (`03_price_data_processing_and_alignment.ipynb`)
   - Calculate comprehensive technical indicators
   - Create lag and rolling window features
   - Align sentiment and price data

4. **Feature Engineering** (`04_anomaly_detection_and_feature_engineering.ipynb`)
   - Implement anomaly detection algorithms
   - Create advanced engineered features
   - Detect market regime changes

5. **Modeling and Backtesting** (`05_modeling_and_backtesting.ipynb`)
   - Train and evaluate machine learning models
   - Implement backtesting strategy
   - Analyze model performance

6. **Results and Visualization** (`06_visualization_and_results.ipynb`)
   - Create comprehensive visualizations
   - Analyze and summarize findings
   - Present conclusions and future work

### Sample Data Generation

The notebooks include functionality to generate sample data for demonstration purposes when the full Kaggle datasets are not available. This allows you to test the pipeline and methodology with synthetic but realistic data.

## Key Features

### Sentiment Analysis
- **FinBERT Integration**: Financial domain-specific sentiment analysis
- **Multi-timeframe Aggregation**: 15-minute and hourly sentiment windows
- **Comprehensive Metrics**: Sentiment polarity, confidence, volume, and volatility

### Technical Analysis
- **100+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R
- **Volume Analysis**: OBV, volume ratios, price-volume relationships
- **Market Regime Detection**: Automated bull/bear/sideways market classification

### Machine Learning
- **Multiple Model Types**: Tree-based, linear, and SVM models
- **Fast Training**: Optimized for quick iteration and experimentation
- **Time Series Validation**: Proper temporal cross-validation methodology

### Backtesting Framework
- **Realistic Trading Simulation**: Transaction costs, slippage considerations
- **Risk Metrics**: Sharpe ratio, maximum drawdown, volatility analysis
- **Performance Visualization**: Cumulative returns, drawdown charts

## Results and Performance

The project demonstrates:
- **Model Performance**: Comparison of different ML approaches for volatility prediction
- **Feature Importance**: Identification of most predictive sentiment and technical features
- **Backtesting Results**: Risk-adjusted returns and trading strategy performance
- **Sentiment-Price Relationships**: Quantified correlations between social sentiment and market movements

## Limitations and Future Work

### Current Limitations
- **Historical Data Only**: Analysis limited to 2016-2019 timeframe
- **Bitcoin Focus**: Single cryptocurrency analysis
- **Simulation Only**: No real-time trading implementation

### Future Enhancements
- **Real-time Implementation**: Live data streaming and prediction system
- **Multi-asset Analysis**: Extension to other cryptocurrencies
- **Deep Learning Models**: LSTM, Transformer architectures for sequence modeling
- **Alternative Data Sources**: News sentiment, on-chain metrics, Reddit discussions

## Academic Context

This project serves as a validation study for the research proposal "Cryptocurrency Tweet Sentiment Analysis: Predicting Bitcoin Volatility Using Social Media Signals." The implementation validates the proposed methodology through comprehensive historical backtesting and provides a foundation for future real-time applications.

### Success Criteria
- **F1 Score ≥ 0.50**: Balanced precision and recall for volatility prediction
- **Sharpe Ratio > 0**: Positive risk-adjusted returns
- **Statistical Significance**: Outperformance of baseline models

## Contributing

This is an academic research project. For questions or collaboration opportunities, please contact lucas.little@colorado.edu.

## License

This project is for academic and educational purposes. Please cite appropriately if using this work in your research.

## Acknowledgments

- **Data Sources**: Kaggle community for providing comprehensive Bitcoin datasets
- **Libraries**: HuggingFace Transformers, scikit-learn, pandas, and the broader Python data science ecosystem
- **Academic Support**: University of Colorado Boulder CSCA 5522 course framework

---

**Note**: This project represents a comprehensive implementation of sentiment-based cryptocurrency volatility prediction, combining advanced NLP techniques with traditional financial analysis methods. The modular design allows for easy experimentation and extension to other cryptocurrencies or time periods.

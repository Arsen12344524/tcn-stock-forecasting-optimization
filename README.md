# TCN Architecture Optimization for Time Series Forecasting

This repository contains a laboratory project focused on the implementation and optimization of Temporal Convolutional Networks (TCN) for stock price prediction. The study addresses the limitations of recurrent architectures and applies systematic hyperparameter tuning for financial data.

## Theoretical Background

The project provides a comparative analysis of TCN against traditional models like RNN, LSTM, and GRU. Key technical aspects include:

* Overcoming RNN Limitations: TCN resolves issues related to sequential processing bottlenecks, slowing down, and difficulties with long dependencies
* Causal Convolutions: Implementation of filters that ensure the output at time t depends only on inputs from moments s <= t, preventing data leakage
* Dilated Convolutions: Utilization of dilation rates to exponentially expand the receptive field without increasing the number of layers significantly

## Methodology

### Data Acquisition
The model is trained on historical stock data for Apple (AAPL) retrieved via the yfinance library for the period 2018–2023. The dataset includes Open, High, Low, Close, and Volume indicators.

### Validation Strategy
To maintain chronological integrity and avoid "looking into the future," a TimeSeriesSplit approach was implemented instead of standard K-Fold cross-validation. This ensures that the training set always precedes the test set in time.

The validation folds were structured as follows:
1. Fold 1: Train (2018–2020) > Test (2021)
2. Fold 2: Train (2018–2021) > Test (2022)
3. Fold 3: Train (2018–2022) > Test (2023)

### Hyperparameter Optimization
A Grid Search was performed to identify the optimal structural configuration for the TCN. The search space focused on parameters affecting the receptive field and capacity:
* kernel_size: [2, 3, 5]
* nb_filters: [32, 64]
* nb_stacks: [1, 2]

## Results and Analysis
The project compares a baseline TCN model (configured with nb_filters: 32, kernel_size: 3, and dilations: [1, 2, 4, 8]) against the optimized version. The analysis demonstrates how variations in architectural parameters impact the error metrics (e.g., RMSE) and the model's overall effectiveness.

## Technical Stack
* Deep Learning: TensorFlow / Keras (via the keras-tcn library)
* Machine Learning: Scikit-learn (GridSearchCV, TimeSeriesSplit)
* Data Processing: Pandas, NumPy, Yfinance

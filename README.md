We are creating this project for live tracking of heavy machinary model for CBM.

A dynamic and interactive predictive maintenance application built using Python and Streamlit. This tool forecasts the Remaining Useful Life (RUL) of critical machinery components using ARIMA-based time series modeling and provides actionable maintenance insights to enhance operational efficiency.

Features-
Core Functionalities:
Remaining Useful Life (RUL) Estimation:
Calculates RUL for multiple machine parameters based on normalized thresholds.
Flags potential failures and provides actionable maintenance recommendations.

Dynamic Forecasting:
Forecast machinery behavior for up to 720 hours (30 days).
Adjustable forecasting period via an interactive slider.

Threshold Monitoring:
Real-time detection of threshold breaches for critical parameters.
Alerts displayed with specific maintenance recommendations.

ARIMA Model Integration:
Auto-tuning of ARIMA parameters using pmdarima.
ACF and PACF plots for detailed time series analysis.

Visualizations:
Historical data and forecast trends visualized through graphs.
Threshold lines and anomaly markers for clarity.

Technologies Used:-
Programming Language:
Python

Libraries:
Streamlit: For creating the web-based User Interface.
pandas: Data handling and preprocessing.
NumPy: Numerical computations.
matplotlib: For plotting historical and forecasted data.
statsmodels: ARIMA model implementation and ACF/PACF visualization.
pmdarima: Automated ARIMA model selection.
scikit-learn: Data normalization using MinMaxScaler.

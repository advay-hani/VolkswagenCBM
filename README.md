# 🔧 Predictive Maintenance Dashboard

A **Streamlit** application designed for predictive maintenance, enabling users to forecast potential failures based on time series data and provide actionable maintenance insights. This app utilizes ARIMA forecasting and Remaining Useful Life (RUL) estimation to help maintain machinery health.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Data Format](#data-format)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Features

- **RUL Estimation**: Calculates Remaining Useful Life (RUL) for multiple parameters using threshold detection.
- **ARIMA Forecasting**: Provides time series forecasting using the ARIMA model for selected parameters.
- **Interactive Visualization**: Displays historical and forecasted data with threshold lines for easy identification of potential issues.
- **Maintenance Recommendations**: Suggests specific maintenance actions based on parameter threshold crossings.

## 🛠️ Installation

1. **Clone the repository**:
```bash
   git clone https://github.com/your-username/predictive-maintenance.git
   cd predictive-maintenance

## Install the required packages:

2. **bash**
```bash
Copy code
pip install -r requirements.txt


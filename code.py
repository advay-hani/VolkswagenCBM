import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # For automated ARIMA tuning

# File path
csv_path = r"D:\Github\HeavyVehicleCBM\dataset.csv"  # Replace with the correct path to your CSV file

# Define thresholds for each parameter
thresholds = {
    'vibr_1_X': 27.9,
    'vibr_1_Y': 27.9,
    'vibr_1_Z': 27.9,
    'vibr_1_noise': 99.2,
    'vibr_2_X': 27.9,
    'vibr_2_Y': 27.9,
    'vibr_2_Z': 27.9,
    'vibr_2_noise': 107.3,
    'oil_temp': 100,
    'cool_temp': 104,
    'oil_pressure': 300,
    'total_rms_vibr': 50,
    'total_rms_noise': 50,
}

# Define maintenance actions
maintenance_actions = {
    'vibr_1_X': "Check and replace vibration dampeners.",
    'vibr_1_Y': "Inspect the foundation and tighten bolts.",
    'vibr_1_Z': "Lubricate motor and inspect bearings.",
    'vibr_1_noise': "Inspect for excessive noise or damage.",
    'vibr_2_X': "Check and replace vibration dampeners.",
    'vibr_2_Y': "Inspect the foundation and tighten bolts.",
    'vibr_2_Z': "Lubricate motor and inspect bearings.",
    'vibr_2_noise': "Inspect for excessive noise or damage.",
    'oil_temp': "Inspect oil cooling system.",
    'cool_temp': "Inspect cooling system and replace coolant.",
    'oil_pressure': "Inspect oil pressure system.",
    'total_rms_vibr': "Inspect vibration dampeners and the system foundation.",
    'total_rms_noise': "Check for loose components or wear in mechanical parts."
}

# Load the CSV
df = pd.read_csv(csv_path)

# Convert timestamp column to datetime if it exists
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

# Ensure numeric values
df = df.apply(pd.to_numeric, errors='coerce')

# Maintenance actions tracking
maintenance_needed = {}

# Process each parameter
for column in thresholds.keys():
    if column in df.columns:
        print(f"Processing: {column}")

        # ACF/PACF Analysis
        print(f"Generating ACF/PACF plots for {column}...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_acf(df[column].dropna(), ax=plt.gca(), lags=40)
        plt.title(f"ACF for {column}")

        plt.subplot(1, 2, 2)
        plot_pacf(df[column].dropna(), ax=plt.gca(), lags=40)
        plt.title(f"PACF for {column}")
        plt.tight_layout()
        plt.show()

        # Automated ARIMA selection
        print(f"Tuning ARIMA parameters for {column}...")
        try:
            model_auto = auto_arima(df[column].dropna(), seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
            print(model_auto.summary())

            # Forecast
            model = ARIMA(df[column].dropna(), order=model_auto.order).fit()
            forecast = model.predict(start=len(df), end=len(df) + 30 * 24, dynamic=True)
            future_dates = pd.date_range(start=df.index[-1], periods=len(forecast), freq='H')

            # Check threshold violations
            if np.any(forecast > thresholds[column]):
                maintenance_needed[column] = maintenance_actions[column]

            # Plot historical data and forecast
            plt.figure(figsize=(10, 6))
            plt.plot(df[column], label='Historical Data', color='blue')
            plt.plot(future_dates, forecast, label='Forecast', color='green', linestyle='dashed')
            plt.axhline(y=thresholds[column], color='red', linestyle='dotted', label='Threshold')
            plt.title(f"{column} Analysis")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid()
            plt.show()

        except Exception as e:
            print(f"Error processing column {column}: {e}")

# Display maintenance actions
if maintenance_needed:
    print("Maintenance required for the following parameters:")
    for param, action in maintenance_needed.items():
        print(f"{param}: {action}")
else:
    print("No maintenance required.")
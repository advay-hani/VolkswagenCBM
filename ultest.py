import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler

# Thresholds for each parameter
thresholds = {
    'vibr_1_X': 27.9, 'vibr_1_Y': 27.9, 'vibr_1_Z': 27.9, 'vibr_1_noise': 99.2,
    'vibr_2_X': 27.9, 'vibr_2_Y': 27.9, 'vibr_2_Z': 27.9, 'vibr_2_noise': 107.3,
    'oil_temp': 100, 'cool_temp': 104, 'oil_pressure': 300,
    'total_rms_vibr': 50, 'total_rms_noise': 50,
}

# Maintenance actions for each parameter
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

# Function to normalize data
def normalize_data(data, thresholds):
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    normalized_thresholds = {col: scaler.transform([[threshold] * len(data.columns)])[0][data.columns.get_loc(col)] for col, threshold in thresholds.items()}
    return normalized_data, normalized_thresholds

# Detect threshold crossings
def detect_threshold_crossings(data, column, threshold):
    crossed = data[column] >= threshold
    crossing_index = crossed.idxmax() if crossed.any() else None
    if crossing_index is not None:
        time_to_failure = data.index[-1] - crossing_index
        return time_to_failure.total_seconds() / 3600  # Convert to hours
    else:
        return np.inf

# Calculate RUL for all parameters
def calculate_rul(data, thresholds):
    rul_results = {}
    for column in thresholds:
        rul = detect_threshold_crossings(data, column, thresholds[column])
        rul_results[column] = rul
    return rul_results

# ARIMA forecasting
def perform_arima_forecasting(column_data, column, forecast_hours):
    try:
        # Plot ACF and PACF
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plot_acf(column_data.dropna(), lags=50, ax=ax1)
        plot_pacf(column_data.dropna(), lags=50, ax=ax2)
        ax1.set_title(f'ACF of {column}', color='darkblue')
        ax2.set_title(f'PACF of {column}', color='darkblue')
        ax1.set_facecolor('#f7f7f7')
        ax2.set_facecolor('#f7f7f7')
        st.pyplot(fig)
        
        # Fit ARIMA model
        model_auto = auto_arima(column_data.dropna(), seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
        model = ARIMA(column_data.dropna(), order=model_auto.order).fit()
        forecast = model.predict(start=len(column_data), end=len(column_data) + forecast_hours, dynamic=True)
        future_dates = pd.date_range(start=column_data.index[-1], periods=len(forecast), freq='H')
        return forecast, future_dates
    except Exception as e:
        st.write(f"Error processing ARIMA for {column}: {e}")
        return None, None

# Streamlit app
def main():
    st.title("🔧 Predictive Maintenance Dashboard")
    st.sidebar.title("Configuration")
    csv_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    # Slider for forecast hours
    forecast_hours = st.sidebar.slider("Select Forecast Horizon (in hours)", min_value=24, max_value=720, value=720, step=4)
    help="Choose the number of hours for the forecast."

    if csv_file is not None:
        df = pd.read_csv(csv_file)

        # Convert timestamp if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # Display preview
        st.write("Uploaded Data Preview:")
        st.write(df.head())

        # Normalize data
        df = df.apply(pd.to_numeric, errors='coerce')
        normalized_data, normalized_thresholds = normalize_data(df, thresholds)

        # Calculate RUL
        rul_results = calculate_rul(normalized_data, normalized_thresholds)

        # Display RUL
        st.subheader("Remaining Useful Life (RUL) Estimates:")
        for param, rul in rul_results.items():
            if rul == np.inf:
                st.markdown(f"<p style='color:green;'>**{param}**: No degradation detected yet</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:orange;'>**{param}**: {rul:.2f} hours</p>", unsafe_allow_html=True)

        # Maintenance detection and ARIMA forecasting
        st.subheader("Maintenance Actions and Forecasting")
        maintenance_needed = {}

        # Interactive Parameter Selector
        selected_columns = st.multiselect(
            "Select Parameters for Forecasting and Maintenance", options=df.columns, default=list(thresholds.keys())
        )

        # Display progress bar during ARIMA processing
        progress_bar = st.progress(0)
        total_steps = len(selected_columns)
        step = 0

        for column in selected_columns:
            forecast, future_dates = perform_arima_forecasting(df[column], column, forecast_hours)
            step += 1
            progress_bar.progress(step / total_steps)

            if forecast is not None:
                if np.any(forecast > thresholds[column]):
                    maintenance_needed[column] = maintenance_actions[column]

                # Plot results
                plt.figure(figsize=(10, 6))
                plt.plot(df[column], label="Historical Data", color='blue')
                plt.plot(future_dates, forecast, label="Forecast", color='green', linestyle='dashed')
                plt.axhline(y=thresholds[column], color='red', linestyle='--', label="Threshold")
                plt.title(f"{column} Analysis", color='darkblue')
                plt.xlabel("Time", color='black')
                plt.ylabel(f"{column} Value", color='black')
                plt.legend()
                plt.grid(True)
                st.pyplot(plt)

        # Display maintenance needed
        if maintenance_needed:
            st.warning("⚠️ Maintenance Required ⚠️", icon="⚙️")
            for param, action in maintenance_needed.items():
                st.markdown(f"<p style='color:red;'>**{param}**: {action}</p>", unsafe_allow_html=True)
        else:
            st.success("✅ No maintenance required. All systems normal!", icon="✅")

if __name__ == "__main__":
    main()

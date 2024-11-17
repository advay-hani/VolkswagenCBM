import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

# Function to process the data
def process_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# Thresholds for each parameter
thresholds = {
    'vibr_1_X': 27.9, 'vibr_1_Y': 27.9, 'vibr_1_Z': 27.9,
    'vibr_1_noise': 99.2, 'vibr_2_X': 27.9, 'vibr_2_Y': 27.9,
    'vibr_2_Z': 27.9, 'vibr_2_noise': 107.3, 'oil_temp': 100,
    'cool_temp': 104, 'oil_pressure': 100,
    'total_rms_vibr': 50, 'total_rms_noise': 50
}

# Maintenance actions
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

# Streamlit app
def main():
    st.title("Maintenance Prediction and Forecasting with Animations")

    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if csv_file is not None:
        df = process_data(csv_file)
        st.write("Data preview", df.head())

        maintenance_needed = {}
        animated_columns = []

        for column in thresholds.keys():
            if column in df.columns:
                st.write(f"Processing: {column}")

                # Automated ARIMA selection
                try:
                    model_auto = auto_arima(df[column].dropna(), seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                    forecast_period = st.slider(f"Select forecast period for {column} (hours)", min_value=1, max_value=720, value=720, step=24)
                    
                    # ARIMA model and forecast
                    model = ARIMA(df[column].dropna(), order=model_auto.order).fit()
                    forecast = model.predict(start=len(df), end=len(df) + forecast_period, dynamic=True)
                    future_dates = pd.date_range(start=df.index[-1], periods=len(forecast), freq='H')

                    # Check threshold violations
                    if np.any(forecast > thresholds[column]):
                        maintenance_needed[column] = maintenance_actions[column]
                        animated_columns.append(column)

                    # Create an animated Plotly graph
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name='Historical Data'))
                    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Forecast', line=dict(dash='dot')))
                    
                    # Animation frames
                    frames = [go.Frame(data=[go.Scatter(x=future_dates[:i], y=forecast[:i], mode='lines', line=dict(color='green'))],
                                       name=str(i)) for i in range(1, len(future_dates))]
                    
                    # Layout and animation settings
                    fig.update_layout(title=f"{column} Analysis with Animated Forecast",
                                      xaxis_title="Time", yaxis_title="Value",
                                      updatemenus=[dict(type="buttons",
                                                        buttons=[dict(label="Play",
                                                                      method="animate",
                                                                      args=[None, {"frame": {"duration": 50, "redraw": True},
                                                                                   "fromcurrent": True}]),
                                                                 dict(label="Pause",
                                                                      method="animate",
                                                                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                                     "mode": "immediate"}])])])
                    
                    fig.update(frames=frames)
                    fig.add_hline(y=thresholds[column], line_dash="dot", line_color="red", annotation_text="Threshold", annotation_position="bottom right")

                    # Display the animated graph
                    st.plotly_chart(fig)

                except Exception as e:
                    st.write(f"Error processing column {column}: {str(e)}")

        # Display animated alerts if maintenance is needed
        if maintenance_needed:
            st.write("⚠️ Maintenance Alerts ⚠️")
            for param, action in maintenance_needed.items():
                st.metric(label=param, value=f"⚠️ {action}")
        else:
            st.success("No maintenance required. All systems normal. ✅")

if __name__ == "__main__":
    main()

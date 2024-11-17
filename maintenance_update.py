import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import time
import warnings
warnings.filterwarnings("ignore")

# Page configuration must be set at the very start of the script
st.set_page_config(page_title="Maintenance Predictor", layout="wide", page_icon="🛠")

# Inject custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00ccff;
        }
        .stButton>button {
            background-color: #00cc99;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #009977;
        }
        .scroll-to-top button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #333333;
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 20px;
            cursor: pointer;
        }
        .scroll-to-top button:hover {
            background-color: #555555;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Function to process the data
def process_data(csv_path):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Convert timestamp column to datetime if it exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Ensure numeric values
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


# Thresholds and actions
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
    st.title("🛠 Maintenance Predictor with Interactive Visualizations")
    st.markdown(
        """
        <div style="background-color: #333333; color: #00ccff; padding: 10px; border-radius: 10px; text-align: center;">
        <h3>Analyze sensor data to predict trends and identify maintenance needs with interactive charts!</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File upload section
    csv_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    
    if csv_file is not None:
        with st.spinner("Processing your data... ⏳"):
            time.sleep(2)
            df = process_data(csv_file)
        st.success("Data loaded successfully!")
        st.write("### Data Preview")
        st.dataframe(df.head())

        maintenance_needed = {}

        for column in thresholds.keys():
            if column in df.columns:
                st.markdown(
                    f"<div style='background-color: #222222; color: #00ccff; padding: 10px; margin-bottom: 10px;'>"
                    f"<h4>Analyzing Sensor Data: {column}</h4></div>", 
                    unsafe_allow_html=True
                )

                # Animated visualization for historical data
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, 
                    y=df[column],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='blue'),
                ))
                fig.update_layout(
                    title=f"Historical Data for {column}",
                    xaxis_title="Time",
                    yaxis_title=column,
                    template="plotly_dark",
                )
                st.plotly_chart(fig)

                # Forecasting and visualization
                st.write(f"### Forecasting Trends for *{column}*")
                try:
                    model_auto = auto_arima(df[column].dropna(), seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
                    model = ARIMA(df[column].dropna(), order=model_auto.order).fit()

                    forecast = model.predict(start=len(df), end=len(df) + 30 * 24, dynamic=True)
                    future_dates = pd.date_range(start=df.index[-1], periods=len(forecast), freq='H')

                    # Animated forecast visualization
                    forecast_fig = go.Figure()
                    forecast_fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=df[column], 
                        mode='lines', 
                        name='Historical Data',
                        line=dict(color='blue'),
                    ))
                    forecast_fig.add_trace(go.Scatter(
                        x=future_dates, 
                        y=forecast, 
                        mode='lines+markers', 
                        name='Forecast',
                        line=dict(color='green', dash='dash'),
                    ))
                    forecast_fig.add_trace(go.Scatter(
                        x=future_dates, 
                        y=[thresholds[column]] * len(future_dates), 
                        mode='lines',
                        name='Threshold',
                        line=dict(color='red', dash='dot'),
                    ))
                    forecast_fig.update_layout(
                        title=f"Forecast with Thresholds for {column}",
                        xaxis_title="Time",
                        yaxis_title=column,
                        template="plotly_dark",
                        legend=dict(x=0, y=1),
                    )
                    st.plotly_chart(forecast_fig)

                    # Check threshold violations
                    if np.any(forecast > thresholds[column]):
                        maintenance_needed[column] = maintenance_actions[column]

                except Exception as e:
                    st.error(f"Error processing {column}: {e}")

        # Display maintenance recommendations
        st.write("## Maintenance Recommendations")
        if maintenance_needed:
            for param, action in maintenance_needed.items():
                st.warning(f"{param}: {action}")
        else:
            st.success("No maintenance required. All parameters are within thresholds.")

        # Add a "Scroll to Top" button
        st.markdown(
            """
            <div class="scroll-to-top">
                <button onclick="window.scrollTo({top: 0, behavior: 'smooth'});">⬆</button>
            </div>
            """, 
            unsafe_allow_html=True,
        )

    else:
        st.info("Please upload a CSV file to start.")
#update

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# APP CONFIGURATION
# ------------------------------
st.set_page_config(page_title="Energy Forecast Pro", page_icon="⚡", layout="wide")

st.title("⚡ Advanced Home Energy Consumption Forecasting App")
st.markdown("""
This enhanced app allows you to **upload your own data** or generate synthetic data, with **multiple forecasting models**, 
**performance metrics**, **confidence intervals**, and **advanced diagnostics**.
""")

# ------------------------------
# CACHED FUNCTIONS
# ------------------------------
@st.cache_data
def generate_energy_data(start, end):
    """Generate realistic synthetic energy data with caching"""
    np.random.seed(42)
    date_rng = pd.date_range(start=start, end=end, freq='H')
    n = len(date_rng)
    
    # Enhanced seasonal patterns
    base = 2 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 24)  # Daily
    yearly = 0.3 * np.sin(2 * np.pi * np.arange(n) / (24*365))  # Yearly
    weekly = 0.2 * np.sin(2 * np.pi * np.arange(n) / (24*7))   # Weekly
    noise = np.random.normal(0, 0.2, n)
    values = 3 + base + yearly + weekly + noise  # kWh per hour
    
    df = pd.DataFrame({'timestamp': date_rng, 'value': values})
    return df

@st.cache_data
def train_arima_model(data, order):
    """Train ARIMA model with caching"""
    try:
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        st.error(f"ARIMA model failed: {e}")
        return None

@st.cache_data
def train_sarima_model(data, order, seasonal_order):
    """Train SARIMA model with caching"""
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        return model_fit
    except Exception as e:
        st.error(f"SARIMA model failed: {e}")
        return None

# ------------------------------
# SIDEBAR SETTINGS
# ------------------------------
st.sidebar.header("📊 Data Configuration")

data_source = st.sidebar.radio("Select Data Source", 
                              ["Generate Synthetic Data", "Upload Your Own CSV"])

if data_source == "Generate Synthetic Data":
    start_date = st.sidebar.date_input("Select Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.sidebar.date_input("Select End Date", value=pd.to_datetime("2024-12-31"))
    
    if start_date >= end_date:
        st.error("⚠️ End date must be after the start date.")
        st.stop()
    
    with st.spinner("Generating synthetic energy data..."):
        df = generate_energy_data(start_date, end_date)

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Auto-detect timestamp and value columns
            timestamp_col = st.sidebar.selectbox("Select timestamp column", df.columns)
            value_col = st.sidebar.selectbox("Select value column", df.columns)
            
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.rename(columns={timestamp_col: 'timestamp', value_col: 'value'})
            df = df[['timestamp', 'value']].sort_values('timestamp')
            
            st.sidebar.success(f"✅ Data loaded: {len(df)} records")
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    else:
        st.info("📁 Please upload a CSV file with timestamp and energy consumption data")
        st.stop()

# ------------------------------
# DATA PREVIEW & DOWNLOAD
# ------------------------------
st.subheader("🔹 Data Overview")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", len(df))
    st.metric("Date Range", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
with col2:
    st.metric("Total Consumption", f"{df['value'].sum():.2f} kWh")
    st.metric("Average Hourly", f"{df['value'].mean():.2f} kWh")

st.dataframe(df.head(10))

# Download option for synthetic data
if data_source == "Generate Synthetic Data":
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Generated CSV Data",
        data=csv_buffer.getvalue(),
        file_name="custom_energy_data.csv",
        mime="text/csv"
    )

# ------------------------------
# DATA PREPARATION
# ------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df_daily = df['value'].resample('D').sum()

st.subheader("📊 Daily Energy Consumption")
st.line_chart(df_daily)

# ------------------------------
# DATA VERIFICATION
# ------------------------------
st.subheader("🔎 Data Verification & Summary")

col1, col2 = st.columns(2)
with col1:
    w1 = st.number_input("Window 1 length (days)", value=7, min_value=1, max_value=365)
    w1_series = df_daily[-w1:]
    w1_total = w1_series.sum()
    w1_avg = w1_series.mean()
    st.metric(label=f"Last {w1} days — Total", value=f"{w1_total:.2f} kWh")
    st.metric(label=f"Last {w1} days — Average", value=f"{w1_avg:.2f} kWh/day")

with col2:
    w2 = st.number_input("Window 2 length (days)", value=30, min_value=1, max_value=365)
    w2_series = df_daily[-w2:]
    w2_total = w2_series.sum()
    w2_avg = w2_series.mean()
    st.metric(label=f"Last {w2} days — Total", value=f"{w2_total:.2f} kWh")
    st.metric(label=f"Last {w2} days — Average", value=f"{w2_avg:.2f} kWh/day")

# ------------------------------
# MODEL CONFIGURATION
# ------------------------------
st.sidebar.header("⚙️ Forecasting Settings")

model_type = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["ARIMA", "SARIMA", "Simple Exponential Smoothing"]
)

# Model parameters
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    p = st.number_input("AR order (p)", value=2, min_value=0, max_value=10)
with col2:
    d = st.number_input("Difference order (d)", value=1, min_value=0, max_value=2)
with col3:
    q = st.number_input("MA order (q)", value=1, min_value=0, max_value=10)

if model_type == "SARIMA":
    col1, col2, col3, col4 = st.sidebar.columns(4)
    with col1:
        sp = st.number_input("Seasonal P", value=1, min_value=0, max_value=5)
    with col2:
        sd = st.number_input("Seasonal D", value=1, min_value=0, max_value=2)
    with col3:
        sq = st.number_input("Seasonal Q", value=1, min_value=0, max_value=5)
    with col4:
        s = st.number_input("Seasonality", value=7, min_value=2, max_value=365)

forecast_option = st.sidebar.selectbox(
    "Select Forecast Period",
    ["Next 7 days", "Next 30 days", "Next 90 days", "Next 365 days", "Custom"]
)

if forecast_option == "Custom":
    steps = st.sidebar.number_input("Custom forecast days", value=30, min_value=1, max_value=730)
else:
    steps = {
        "Next 7 days": 7,
        "Next 30 days": 30,
        "Next 90 days": 90,
        "Next 365 days": 365
    }[forecast_option]

# ------------------------------
# MODEL TRAINING & FORECASTING
# ------------------------------
if st.sidebar.button("🚀 Train Model & Forecast", type="primary"):
    
    # Split data for validation
    train_size = int(len(df_daily) * 0.8)
    train_data = df_daily[:train_size]
    test_data = df_daily[train_size:]
    
    with st.spinner("Training model and generating forecast..."):
        
        # Model training
        if model_type == "ARIMA":
            model_fit = train_arima_model(df_daily, (p, d, q))
        elif model_type == "SARIMA":
            model_fit = train_sarima_model(df_daily, (p, d, q), (sp, sd, sq, s))
        else:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            model_fit = SimpleExpSmoothing(df_daily).fit()
        
        if model_fit is None:
            st.error("Model training failed. Please adjust parameters.")
            st.stop()
        
        # Generate forecast with confidence intervals
        if model_type in ["ARIMA", "SARIMA"]:
            forecast_result = model_fit.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)
        else:
            forecast = model_fit.forecast(steps=steps)
            conf_int = None
        
        future_dates = pd.date_range(df_daily.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted': forecast,
            'lower_ci': conf_int.iloc[:, 0] if conf_int is not None else None,
            'upper_ci': conf_int.iloc[:, 1] if conf_int is not None else None
        }).set_index('date')
        
        # ------------------------------
        # MODEL PERFORMANCE METRICS
        # ------------------------------
        st.subheader("📈 Model Performance Metrics")
        
        # Validation forecast
        val_forecast_steps = len(test_data)
        if model_type in ["ARIMA", "SARIMA"]:
            val_forecast = model_fit.get_prediction(start=len(train_data), end=len(df_daily)-1)
            val_predicted = val_forecast.predicted_mean
        else:
            val_predicted = model_fit.forecast(steps=val_forecast_steps)
        
        # Calculate metrics
        mae = mean_absolute_error(test_data, val_predicted)
        rmse = np.sqrt(mean_squared_error(test_data, val_predicted))
        mape = np.mean(np.abs((test_data - val_predicted) / test_data)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{mae:.2f} kWh")
        with col2:
            st.metric("RMSE", f"{rmse:.2f} kWh")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        with col4:
            if hasattr(model_fit, 'aic'):
                st.metric("AIC", f"{model_fit.aic:.2f}")
        
        # ------------------------------
        # VISUALIZATION
        # ------------------------------
        st.subheader("🔮 Forecast Visualization")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Full timeline with forecast
        ax1.plot(df_daily.index, df_daily.values, label='Historical', color='blue', alpha=0.7)
        ax1.plot(forecast_df.index, forecast_df['predicted'], label='Forecast', color='red', linewidth=2)
        
        if conf_int is not None:
            ax1.fill_between(forecast_df.index, 
                           forecast_df['lower_ci'], 
                           forecast_df['upper_ci'], 
                           color='red', alpha=0.2, label='95% Confidence Interval')
        
        ax1.set_title(f"Energy Consumption Forecast - {model_type} Model")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Energy Consumption (kWh)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation plot
        ax2.plot(train_data.index, train_data.values, label='Training', color='blue')
        ax2.plot(test_data.index, test_data.values, label='Actual Test', color='green')
        ax2.plot(test_data.index, val_predicted, label='Predicted', color='red', linestyle='--')
        ax2.set_title("Model Validation (Train-Test Split)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Energy Consumption (kWh)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ------------------------------
        # FORECAST RESULTS
        # ------------------------------
        st.subheader("📅 Forecast Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_forecast = forecast_df['predicted'].sum()
            avg_daily = forecast_df['predicted'].mean()
            peak_day = forecast_df['predicted'].idxmax()
            peak_value = forecast_df['predicted'].max()
            
            st.markdown(f"""
            ### 📊 Forecast Summary
            - **Total Predicted Consumption:** {total_forecast:.2f} kWh  
            - **Average Daily Consumption:** {avg_daily:.2f} kWh/day  
            - **Peak Consumption Day:** {peak_day.strftime('%Y-%m-%d')} ({peak_value:.2f} kWh)
            - **Forecast Period:** {steps} days
            - **Model Used:** {model_type}
            """)
        
        with col2:
            st.dataframe(forecast_df.round(2))
        
        # ------------------------------
        # MODEL DIAGNOSTICS
        # ------------------------------
        if model_type in ["ARIMA", "SARIMA"]:
            st.subheader("🔧 Model Diagnostics")
            
            try:
                # Residuals analysis
                residuals = model_fit.resid
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Residuals plot
                ax1.plot(residuals)
                ax1.set_title("Residuals Over Time")
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Residuals")
                ax1.grid(True, alpha=0.3)
                
                # Residuals histogram
                ax2.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
                ax2.set_title("Residuals Distribution")
                ax2.set_xlabel("Residuals")
                ax2.set_ylabel("Frequency")
                
                # Q-Q plot
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=ax3)
                ax3.set_title("Q-Q Plot")
                
                # ACF of residuals
                from statsmodels.graphics.tsaplots import plot_acf
                plot_acf(residuals, ax=ax4, lags=20)
                ax4.set_title("ACF of Residuals")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Ljung-Box test for autocorrelation
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
                st.write("Ljung-Box Test for Residual Autocorrelation:")
                st.dataframe(lb_test)
                
            except Exception as e:
                st.warning(f"Could not generate full diagnostics: {e}")
        
        st.success("✅ Forecasting complete!")

# ------------------------------
# SIDEBAR FOOTER
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.info("""
**Enhanced Features:**
- Multiple model types
- Performance metrics
- Confidence intervals
- Model diagnostics
- Real data upload
- Caching for speed
""")
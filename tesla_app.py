import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Tesla Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown('''
<style>
    .main-header {
        font-size: 3rem;
        color: #E82127;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
''', unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🚗 Tesla Stock Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Using Deep Learning (LSTM & SimpleRNN)</p>', unsafe_allow_html=True)
st.markdown(" Соответствии с заявленной темой")

# Sidebar
st.sidebar.header(" Configuration")
st.sidebar.markdown("Configure your prediction parameters:")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["LSTM", "SimpleRNN"],
    help="Choose between LSTM and SimpleRNN models"
)

prediction_days = st.sidebar.slider(
    "Prediction Horizon (days)",
    min_value=1,
    max_value=10,
    value=1,
    help="Number of days to forecast into the future"
)

st.sidebar.markdown("### 📊 About This App")
st.sidebar.info(
    "This application uses deep learning models (LSTM and SimpleRNN) "
    "trained on historical Tesla stock data to predict future prices."
)

st.sidebar.markdown("###  Model Information")
st.sidebar.markdown(f"""
- **Training Data**: 5+ years of Tesla stock prices
- **Features**: Open, High, Low, Close, Volume
- **Architecture**: Multi-layer RNN with dropout
- **Optimization**: GridSearchCV hyperparameter tuning
""")

# Load data function
@st.cache_data
def load_stock_data():
    '''Load Tesla stock data'''
    try:
        # Try to load from CSV
        df = pd.read_csv('TSLA.csv', parse_dates=['Date'])
        df = df.sort_values('Date')
        st.sidebar.success("-> Data loaded from TSLA.csv")
        return df
    except FileNotFoundError:
        # Fallback to yfinance
        st.sidebar.info(" Loading data from Yahoo Finance...")
        import yfinance as yf
        ticker = yf.Ticker("TSLA")
        df = ticker.history(period="5y")
        df.reset_index(inplace=True)
        st.sidebar.success("-> Data loaded from Yahoo Finance")
        return df
    except Exception as e:
        st.sidebar.error(f"XXX Error loading data: {e}")
        return None

# Load models function
@st.cache_resource
def load_models():
    '''Load trained models'''
    lstm_model = None
    rnn_model = None

    try:
        lstm_model = load_model('lstm_final_model.h5', compile=False)
        st.sidebar.success("-> LSTM model loaded")
    except Exception as e:
        st.sidebar.warning(f"XXX LSTM model not found: {e}")

    try:
        rnn_model = load_model('simple_rnn_final_model.h5', compile=False)
        st.sidebar.success("-> SimpleRNN model loaded")
    except Exception as e:
        st.sidebar.warning(f"XXX SimpleRNN model not found: {e}")

    return lstm_model, rnn_model

# Prepare data for prediction
def prepare_data(df, time_steps=60):
    '''Prepare data for model prediction'''
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Use Close price
    if 'Close' in df.columns:
        data = df['Close'].values.reshape(-1, 1)
    elif 'Adj Close' in df.columns:
        data = df['Adj Close'].values.reshape(-1, 1)
    else:
        st.error("No price column found in data!")
        return None, None, None

    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, scaler, data

# Main app
def main():
    # Load data
    with st.spinner("Loading data..."):
        df = load_stock_data()

    if df is None:
        st.error("XXX Failed to load data. Please ensure TSLA.csv is in the same directory or check internet connection.")
        st.stop()

    # Load models
    with st.spinner("Loading models..."):
        lstm_model, rnn_model = load_models()

    if lstm_model is None and rnn_model is None:
        st.error("XXX No models found! Please ensure model files are in the same directory.")
        st.info("Expected files: lstm_final_model.h5 and/or simple_rnn_final_model.h5")
        st.stop()

    # Check if selected model is available
    if model_choice == "LSTM" and lstm_model is None:
        st.warning("XXX LSTM model not available. Please select SimpleRNN.")
        st.stop()

    if model_choice == "SimpleRNN" and rnn_model is None:
        st.warning("XXX SimpleRNN model not available. Please select LSTM.")
        st.stop()

    # Display basic statistics
    st.header(" Current Stock Information")

    col1, col2, col3, col4 = st.columns(4)

    current_price = df['Close'].iloc[-1] if 'Close' in df.columns else df['Adj Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if 'Close' in df.columns else df['Adj Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    with col1:
        st.metric("Current Price", f"${current_price:.2f}",
                 f"${price_change:.2f} ({price_change_pct:.2f}%) Kishan")

    with col2:
        high_52w = df['High'].tail(252).max()
        st.metric("52-Week High", f"${high_52w:.2f}")

    with col3:
        low_52w = df['Low'].tail(252).min()
        st.metric("52-Week Low", f"${low_52w:.2f}")

    with col4:
        avg_volume = df['Volume'].tail(30).mean()
        st.metric("Avg Volume (30d)", f"{avg_volume/1e6:.2f}M")

  

    # Show stock price chart
    st.header("📈 Historical Stock Prices")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox(
            "Select Time Range",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "All Time"]
        )

    # Filter data based on selection
    if time_range == "1 Month":
        plot_df = df.tail(30)
    elif time_range == "3 Months":
        plot_df = df.tail(90)
    elif time_range == "6 Months":
        plot_df = df.tail(180)
    elif time_range == "1 Year":
        plot_df = df.tail(365)
    elif time_range == "2 Years":
        plot_df = df.tail(730)
    else:
        plot_df = df

    fig, ax = plt.subplots(figsize=(12, 6))
    price_col = 'Close' if 'Close' in plot_df.columns else 'Adj Close'
    ax.plot(plot_df['Date'], plot_df[price_col], label='Close Price', linewidth=2, color='#E82127')
    ax.fill_between(plot_df['Date'], plot_df[price_col], alpha=0.3, color='#E82127')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'Tesla Stock Price - {time_range}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


    # Make predictions
    st.header(" Model Predictions")

    with st.spinner(f"Making predictions using {model_choice} model..."):
        # Prepare data
        X, scaler, original_data = prepare_data(df)

        if X is None:
            st.error("Failed to prepare data for prediction")
            st.stop()

        # Select model
        selected_model = lstm_model if model_choice == "LSTM" else rnn_model

        # Make predictions
        predictions = selected_model.predict(X, verbose=0)
        predictions = scaler.inverse_transform(predictions)

        # Get actual values
        actual_values = original_data[60:]

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 {model_choice} Performance Metrics")

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("RMSE", f"${rmse:.2f}")
                st.metric("MAE", f"${mae:.2f}")
            with metric_col2:
                st.metric("R² Score", f"{r2:.4f}")
                accuracy_pct = r2 * 100
                st.metric("Accuracy", f"{accuracy_pct:.2f}%")

            st.info(f"""
            **Interpretation:**
            - RMSE of ${rmse:.2f} means predictions are typically within ±${rmse:.2f}
            - R² of {r2:.4f} means the model explains {accuracy_pct:.2f}% of price variance
            """)

        with col2:
            st.subheader(f" {prediction_days}-Day Forecast")

            # Future predictions
            last_sequence = X[-1]
            future_predictions = []

            for _ in range(prediction_days):
                next_pred = selected_model.predict(last_sequence.reshape(1, 60, 1), verbose=0)
                future_predictions.append(next_pred[0, 0])

                # Update sequence
                last_sequence = np.append(last_sequence[1:], next_pred)

            future_predictions = scaler.inverse_transform(
                np.array(future_predictions).reshape(-1, 1)
            )

            # Display predictions
            for i, pred in enumerate(future_predictions, 1):
                change = pred[0] - current_price
                change_pct = (change / current_price) * 100

                delta_color = "normal" if change >= 0 else "inverse"
                st.metric(
                    f"Day {i} Prediction",
                    f"${pred[0]:.2f}",
                    f"${change:.2f} ({change_pct:.2f}%) Kishan"
                )

            avg_prediction = np.mean(future_predictions)
            st.success(f"📈 Average {prediction_days}-day prediction: ${avg_prediction:.2f}")


    # Plot predictions vs actual
    st.header("📈 Predictions vs Actual Prices")

    # Show last N points for clarity
    plot_points = st.slider("Number of points to display", 50, 500, 200, 50)
    plot_points = min(plot_points, len(actual_values))

    fig, ax = plt.subplots(figsize=(14, 7))

    x_axis = range(plot_points)
    ax.plot(x_axis, actual_values[-plot_points:], label='Actual Price',
            linewidth=2.5, color='blue', alpha=0.7)
    ax.plot(x_axis, predictions[-plot_points:], label=f'{model_choice} Prediction',
            linewidth=2.5, color='red', alpha=0.7, linestyle='--')

    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_choice} Model: Predictions vs Actual (Last {plot_points} points)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Prediction error plot
    errors = actual_values[-plot_points:] - predictions[-plot_points:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x_axis, errors, color='purple', alpha=0.6, linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.fill_between(x_axis, errors.flatten(), alpha=0.3, color='purple')
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Error ($)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


    # Model comparison
    st.header("⚖️ Model Comparison")

    try:
        comparison_df = pd.read_csv('model_comparison.csv')

        st.subheader("Performance Metrics Comparison")
        st.dataframe(comparison_df, use_container_width=True)

        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE comparison
        models = ['SimpleRNN', 'LSTM']
        train_rmse = comparison_df['SimpleRNN (Train)'].iloc[1] # RMSE row, SimpleRNN (Train) column
        test_rmse_rnn = comparison_df['SimpleRNN (Test)'].iloc[1] # RMSE row, SimpleRNN (Test) column
        train_rmse_lstm = comparison_df['LSTM (Train)'].iloc[1] # RMSE row, LSTM (Train) column
        test_rmse_lstm = comparison_df['LSTM (Test)'].iloc[1] # RMSE row, LSTM (Test) column

        # Convert string values to float for plotting
        train_rmse = float(train_rmse)
        test_rmse_rnn = float(test_rmse_rnn)
        train_rmse_lstm = float(train_rmse_lstm)
        test_rmse_lstm = float(test_rmse_lstm)


        x = np.arange(len(models))
        width = 0.35

        # Plot 1: RMSE
        ax1 = axes[0]
        ax1.bar(x[0] - width/2, train_rmse, width, label='SimpleRNN Train', color='skyblue')
        ax1.bar(x[0] + width/2, test_rmse_rnn, width, label='SimpleRNN Test', color='lightcoral')
        ax1.bar(x[1] - width/2, train_rmse_lstm, width, label='LSTM Train', color='lightblue')
        ax1.bar(x[1] + width/2, test_rmse_lstm, width, label='LSTM Test', color='salmon')
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('RMSE', fontweight='bold')
        ax1.set_title('RMSE Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: R² Score
        train_r2_rnn = comparison_df['SimpleRNN (Train)'].iloc[3] # R2 Score row, SimpleRNN (Train) column
        test_r2_rnn = comparison_df['SimpleRNN (Test)'].iloc[3] # R2 Score row, SimpleRNN (Test) column
        train_r2_lstm = comparison_df['LSTM (Train)'].iloc[3] # R2 Score row, LSTM (Train) column
        test_r2_lstm = comparison_df['LSTM (Test)'].iloc[3] # R2 Score row, LSTM (Test) column

        # Convert string values to float for plotting
        train_r2_rnn = float(train_r2_rnn)
        test_r2_rnn = float(test_r2_rnn)
        train_r2_lstm = float(train_r2_lstm)
        test_r2_lstm = float(test_r2_lstm)

        ax2 = axes[1]
        ax2.bar(x[0] - width/2, train_r2_rnn, width, label='SimpleRNN Train', color='skyblue')
        ax2.bar(x[0] + width/2, test_r2_rnn, width, label='SimpleRNN Test', color='lightcoral')
        ax2.bar(x[1] - width/2, train_r2_lstm, width, label='LSTM Train', color='lightblue')
        ax2.bar(x[1] + width/2, test_r2_lstm, width, label='LSTM Test', color='salmon')
        ax2.set_xlabel('Model', fontweight='bold')
        ax2.set_ylabel('R² Score', fontweight='bold')
        ax2.set_title('R² Score Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)

    except FileNotFoundError:
        st.info("📊 Model comparison data not available. Run the notebook to generate model_comparison.csv")


    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong> Tesla Stock Price Prediction using Deep Learning</strong></p>
        <p>Built with ❤️ using Streamlit, TensorFlow & Scikit-learn</p>
        <p> <em>Disclaimer: This is for educational purposes only. Not financial advice!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

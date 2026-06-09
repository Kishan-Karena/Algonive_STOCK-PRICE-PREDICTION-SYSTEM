import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction System")
st.markdown("""
This system uses a **Deep Learning LSTM model** to forecast future stock prices based on historical data.
Upload your dataset, visualize trends, train the AI, and predict multi-step future prices.
""")


st.sidebar.header("📋 Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Stock CSV File", type=["csv"])

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.success("File uploaded successfully!")
    
    columns = df.columns.tolist()
    date_col = st.sidebar.selectbox("Select Date Column", columns, index=0 if 'Date' in columns else 0)
    target_col = st.sidebar.selectbox("Select Target Price Column (e.g., Close)", columns, index=1 if 'Close' in columns else 0)
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    st.subheader("📊 Historical Data & Visualizations")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### Data Preview", df.head())
        st.write("### Summary Statistics", df.describe())
        
    with col2:
        st.write(f"### Historical {target_col} Price Trend")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[date_col], df[target_col], label=target_col, color='#1f77b4')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)


    st.subheader("🤖 Prediction Model (LSTM)")
    
    st.markdown("#### Model Training Parameters")
    c1, c2, c3 = st.columns(3)
    seq_len = c1.slider("Sequence Length (Lookback days)", 10, 90, 30)
    epochs = c2.slider("Training Epochs", 5, 100, 10)
    future_days = c3.slider("Multi-Step Forecast (Days to predict)", 1, 1000, 7)
    
    if st.button("🚀 Train Model & Forecast"):
        with st.spinner("Preparing data and training LSTM model... Please wait."):
            
            data = df[[target_col]].values
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            X, y = create_sequences(scaled_data, seq_len)
            
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
            
            test_predictions = model.predict(X_test)
            test_predictions = scaler.inverse_transform(test_predictions)
            y_test_actual = scaler.inverse_transform(y_test)
            
            
            last_sequence = scaled_data[-seq_len:]
            future_predictions = []
            
            current_seq = last_sequence.copy()
            for _ in range(future_days):
                current_seq_reshaped = current_seq.reshape(1, seq_len, 1)
                pred = model.predict(current_seq_reshaped, verbose=0)
                future_predictions.append(pred[0, 0])
                current_seq = np.append(current_seq[1:], pred).reshape(seq_len, 1)
                
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            
            last_date = df[date_col].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                f'Predicted {target_col}': future_predictions.flatten()
            })
            
        st.success("Model trained and forecasting complete!")
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.write("### 📈 Model Evaluation & Future Forecast")
            
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            ax2.plot(df[date_col], df[target_col], label='Actual Historical Price', color='#1f77b4')
            
            test_dates = df[date_col].iloc[train_size + seq_len:]
            ax2.plot(test_dates, test_predictions, label='Model Validation Fit', color='orange', linestyle='--')
            
            ax2.plot(
                forecast_df['Date'], 
                forecast_df[f'Predicted {target_col}'], 
                label=f'Future {future_days}-Day Forecast', 
                color='red', 
                linestyle='-', 
                linewidth=1.2, 
                marker='o', 
                markersize=3
            )

            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price")
            ax2.legend()
            st.pyplot(fig2)
            
        with col_res2:
            st.write(f"### 🔮 Next {future_days} Days Forecast")
            st.dataframe(forecast_df.style.format({f'Predicted {target_col}': "{:.2f}"}))
            
            csv_data = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", data=csv_data, file_name="stock_forecast.csv", mime="text/csv")

else:
    st.info("💡 Please upload a historical stock data CSV file from the sidebar to begin.")
    st.markdown("""
    **Expected CSV format example:**
    | Date       | Open   | High   | Low    | Close  | Volume |
    |------------|--------|--------|--------|--------|--------|
    | 2026-01-01 | 150.25 | 152.00 | 149.50 | 151.10 | 500000 |
    | 2026-01-02 | 151.15 | 153.45 | 150.10 | 152.95 | 620000 |
    """)
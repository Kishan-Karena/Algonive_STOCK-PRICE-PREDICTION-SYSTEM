# 📈 Stock Price Prediction System

A powerful stock price forecasting application built with **Streamlit** and **Deep Learning**. This system utilizes a **Long Short-Term Memory (LSTM)** neural network to analyze historical stock price trends and predict future values.

## 🚀 Features

- **Data Visualization**: Upload historical stock data in CSV format and visualize price trends using interactive plots.
- **Customizable Parameters**:
    - **Sequence Length**: Adjust the lookback window (number of previous days the model uses to predict the next day).
    - **Training Epochs**: Control the number of iterations the model trains on the dataset.
    - **Multi-Step Forecast**: Predict stock prices for a custom number of future days.
- **Deep Learning Core**: Implements a multi-layer LSTM architecture with Dropout layers to prevent overfitting.
- **Forecast Export**: Download the predicted future prices as a CSV file for further analysis.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/)
- **Machine Learning**: [TensorFlow/Keras](https://tensorflow.org/) (LSTM)
- **Preprocessing**: [Scikit-Learn](https://scikit-learn.org/) (MinMaxScaler)

## 📋 Getting Started

### Prerequisites

Ensure you have Python installed. You will need the following libraries:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn tensorflow
```

### Running the Application

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Stoke Price Prediction System"
   ```

2. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## 📊 How to Use

1. **Upload Data**: Use the sidebar to upload a CSV file containing historical stock data.
2. **Configure Columns**: Select the `Date` column and the `Target Price` column (e.g., 'Close') from the dropdowns.
3. **Set Model Parameters**: Adjust the lookback days, epochs, and forecast duration using the sliders.
4. **Train & Predict**: Click the **"🚀 Train Model & Forecast"** button.
5. **Analyze Results**: View the historical trends, the model's validation fit, and the red-colored future forecast line.

### Expected CSV Format

Your CSV should look similar to this:

| Date       | Open   | High   | Low    | Close  | Volume |
|------------|--------|--------|--------|--------|--------|
| 2026-01-01 | 150.25 | 152.00 | 149.50 | 151.10 | 500000 |
| 2026-01-02 | 151.15 | 153.45 | 150.10 | 152.95 | 620000 |

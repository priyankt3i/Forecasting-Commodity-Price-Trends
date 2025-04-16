import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data('./Dataset/all_commodities_data.csv')

# Sidebar for user input
st.sidebar.title("Commodity Price Trends Analysis")
commodity = st.sidebar.selectbox("Select Commodity", df['commodity'].unique())
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Year-wise Trend", "Correlation", "Volatility", "Model Comparison"])

# Filter data for selected commodity
commodity_df = df[df['commodity'] == commodity].copy()

# Main content
st.title("Commodity Price Trends")
st.write(f"Analyzing price trends for {commodity}")
st.write(commodity_df.head())

if analysis_type == "Year-wise Trend":
    # Year-wise trend analysis
    st.subheader("Year-wise Trend Analysis")
    commodity_df['date'] = pd.to_datetime(commodity_df['date'])
    commodity_df['year'] = commodity_df['date'].dt.year

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=commodity_df, x='year', y='close', marker='o', ax=ax)
    ax.set_title(f'{commodity} Prices Across Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Close Price')
    st.pyplot(fig)

elif analysis_type == "Model Comparison":
    st.subheader("Model Comparison")
    model = st.selectbox("Select Model", ["ARIMA", "Prophet", "LSTM"])
    if model == "ARIMA":
        st.write("ARIMA model evaluation")
        commodity_df['date'] = pd.to_datetime(commodity_df['date'])
        commodity_df.set_index('date', inplace=True)
        ts_data = commodity_df['close'].dropna()
        train_size = int(len(ts_data) * 0.8)
        train, test = ts_data[:train_size], ts_data[train_size:]
        arima_model = ARIMA(train, order=(1, 1, 1))
        results = arima_model.fit()
        predictions = results.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        st.write(f"RMSE: {rmse}, MAE: {mae}")
    elif model == "Prophet":
        st.write("Prophet model evaluation")
        prophet_data = commodity_df.reset_index()[['date', 'close']]
        prophet_data.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
        model = Prophet()
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        fig = model.plot(forecast)
        st.pyplot(fig)
    elif model == "LSTM":
        st.write("LSTM model evaluation")
        merged_df = commodity_df.reset_index()[['date', 'close']]
        prices = merged_df['close'].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
        def create_dataset(data, look_back=60):
            X, y = [], []
            for i in range(look_back, len(data)):
                X.append(data[i-look_back:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)
        look_back = 60
        X, y = create_dataset(prices_scaled, look_back)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=20)
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        actual_prices_y = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(actual_prices_y, predictions))
        mae = mean_absolute_error(actual_prices_y, predictions)
        st.write(f"RMSE: {rmse}, MAE: {mae}")
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.lineplot(data=pd.DataFrame({'date': merged_df.reset_index()['date'].iloc[-len(predictions_scaled):], 'actual': actual_prices_y.flatten()}), x='date', y='actual', ax=ax[0], color='blue', label='Actual Price')
        ax[0].set_title("Actual Prices")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Price")
        ax[0].tick_params(axis='x', rotation=45)
        sns.lineplot(data=pd.DataFrame({'date': merged_df.reset_index()['date'].iloc[-len(predictions_scaled):], 'predicted': predictions.flatten()}), x='date', y='predicted', ax=ax[1], color='red', label='Predicted Price')
        ax[1].set_title("Predicted Prices")
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Price")
        ax[1].tick_params(axis='x', rotation=45)
        st.pyplot(fig)

elif analysis_type == "Correlation":
    # Correlation between commodity prices
    st.subheader("Correlation Between Commodity Prices")
    df['date'] = pd.to_datetime(df['date'])
    pivot_data = df.pivot(index='date', columns='commodity', values='close')
    correlation_matrix = pivot_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, square=True, ax=ax)
    ax.set_title('Correlation Between Commodity Prices')
    st.pyplot(fig)

elif analysis_type == "Volatility":
    # Volatility analysis
    st.subheader("Price Volatility of Commodities")
    df['daily_pct_change'] = df.groupby('commodity')['close'].pct_change()
    vol_data = df.groupby('commodity')['daily_pct_change'].std().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=vol_data, x='commodity', y='daily_pct_change', ax=ax)
    ax.set_title('Price Volatility of Commodities')
    ax.set_xlabel('Commodity')
    ax.set_ylabel('Volatility (Std. Dev. of Daily % Change)')
    st.pyplot(fig)

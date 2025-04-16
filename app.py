import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data('./Dataset/all_commodities_data.csv')

# Sidebar for user input
st.sidebar.title("Commodity Price Trends Analysis")
commodity = st.sidebar.selectbox("Select Commodity", df['commodity'].unique())
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Year-wise Trend", "Correlation", "Volatility"])

# Filter data for selected commodity
commodity_df = df[df['commodity'] == commodity]

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

elif analysis_type == "Correlation":
    # Correlation between commodity prices
    st.subheader("Correlation Between Commodity Prices")
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

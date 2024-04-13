# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:34:42 2024

@author: mskam
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def get_historical_price(coin_names, days):
    historical_prices = {}
    for coin_name in coin_names:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_name.lower()}/market_chart?vs_currency=usd&days={days}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            prices = [entry[1] for entry in data['prices']]
            dates = [datetime.fromtimestamp(entry[0] / 1000) for entry in data['prices']]
            historical_prices[coin_name] = pd.DataFrame({'Date': dates, 'Price': prices})
        else:
            st.error(f"Error fetching historical price data for {coin_name}")
            return None
            st.error(f"Error fetching historical price data for {coin_name}")
            return None
    return historical_prices

def plot_price_history(coin_names, days):
    historical_prices = get_historical_price(coin_names, days)
    if historical_prices is not None:
        fig, ax = plt.subplots()
        max_prices = {}
        min_prices = {}
        for coin_name, data in historical_prices.items():
            ax.plot(data['Date'], data['Price'], label=coin_name)
            max_price = data['Price'].max()
            min_price = data['Price'].min()
            max_date = data.loc[data['Price'].idxmax(), 'Date']
            min_date = data.loc[data['Price'].idxmin(), 'Date']
            max_prices[coin_name] = (max_price, max_date)
            min_prices[coin_name] = (min_price, min_date)
        ax.set_title("Price History Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        for coin_name in coin_names:
            max_price, max_date = max_prices[coin_name]
            min_price, min_date = min_prices[coin_name]
            st.write(f"{coin_name}:")
            st.write(f"Max Price: ${max_price} (on {max_date})")
            st.write(f"Min Price: ${min_price} (on {min_date})")
    else:
        st.error("Error fetching historical price data")

def main():
    st.title("Stock Comparison App")
    coin_name1 = st.text_input("Enter first cryptocurrency name (e.g., Bitcoin, Ethereum):")
    coin_name2 = st.text_input("Enter second cryptocurrency name (e.g., Bitcoin, Ethereum):")
    time_frame = st.selectbox("Select time frame:", ["1 week", "1 month", "1 year", "5 years"])
    if st.button("Compare"):
        if time_frame == "1 week":
            days = 7
        elif time_frame == "1 month":
            days = 30
        elif time_frame == "1 year":
            days = 365
        elif time_frame == "5 years":
            days = 365 * 5
        coin_names = [coin_name1, coin_name2]
        plot_price_history(coin_names, days)

if __name__ == "__main__":
    main()

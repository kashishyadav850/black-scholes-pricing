import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
import pandas as pd

def black_scholes (S,K,T,r,sigma,option ="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: #put option
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
#streamlit UI

st.sidebar.header("BLACK SCHOLES MODEL")
S = st.sidebar.number_input("Current Asset Price",value = 100.0)
K = st.sidebar.number_input("Strike Price",value=100.0)
T = st.sidebar.number_input("Time to Maturity", value=1.0)
sigma = st.sidebar.number_input("Volatility(σ)",value=0.2)
r= st.sidebar.number_input("Risk Free Interest Rate",value=0.05)

call_price = black_scholes(S,K,T,r,sigma,"call")
put_price = black_scholes(S,K,T,r,sigma,"put")

st.success(f"CALL Value:{call_price:.2f}")    # will show the output in green
st.error(f"PUT Value:{put_price:.2f}")         #will show the output as red

#heatmap
spot_prices = np.linspace(S * 0.8 , S * 1.2 , 10)     # creates 10 evenly spaced spotprices between 80 to 90 %
volatilities = np.linspace(0.1 , 0.3 , 10)             #creates evenly spaced volatilities from 10 to 30%
call_matrix = np.zeros((len(volatilities),len(spot_prices)))

spot_prices_grid , volatilities_grid = np.meshgrid(spot_prices,volatilities)    #Create two 2D grids from your spot prices and volatilities arrays #    Each grid pairs every spot price with every volatility
vectorized_bs = np.vectorize(lambda S, v:black_scholes (S,K,T,r,sigma,"call"))   #Convert your black_scholes function into a "vectorized" function - meaning it can accept and operate elementwise on arrays
call_matrix = vectorized_bs(spot_prices_grid,volatilities_grid)

st.write("Call Option Heatmap")
fig , ax = plt.subplots()
c = ax.imshow(call_matrix, aspect ="auto", origin = "lower" , cmap = "viridis")
plt.colorbar(c, ax=ax)
ax.set_xticks(np.arange(len(spot_prices)))
ax.set_yticks(np.arange(len(volatilities)))
ax.set_xticklabels([f"{sp:.2f}" for sp in spot_prices])
ax.set_yticklabels([f"{v:.2f}" for v in volatilities])
ax.set_xlabel("Spot Price (S)")
ax.set_ylabel("Volatility (σ)")
st.pyplot(fig)



import numpy as np
import streamlit as st
from math import ceil
from scipy import linalg
import matplotlib.pyplot as plt

def lowess(x, y, f, iterations):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)], [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest

# Streamlit app
st.title("LOWESS Smoothing with Streamlit")
st.write("""
    This app applies LOWESS (Locally Weighted Scatterplot Smoothing) to noisy data.
""")

# Sidebar parameters
n = st.sidebar.slider("Number of data points", 50, 200, 100)
f = st.sidebar.slider("Smoothing factor (f)", 0.01, 1.0, 0.25)
iterations = st.sidebar.slider("Number of iterations", 1, 10, 3)

# Generate test data
x = np.linspace(0, 2 * np.pi, n)
np.random.seed(0)  # For reproducibility
y = np.sin(x) + 0.3 * np.random.randn(n)

# Apply LOWESS smoothing
yest = lowess(x, y, f, iterations)

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, "r.", label='Noisy data')
ax.plot(x, yest, "b-", label='LOWESS smoothing')
ax.legend()

# Display the plot in the Streamlit app
st.pyplot(fig)

# Display data
st.write("Generated data:")
st.write(pd.DataFrame({'x': x, 'y': y, 'yest': yest}))


   

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import altair as alt

# --- App Title and Description ---
st.set_page_config(
    page_title="Interactive Linear Regression Visualizer",
    page_icon="📈",
    layout="centered",
)
st.title("📈 Interactive Linear Regression Visualizer")
st.write(
    "This app visualizes linear regression. "
    "Use the sliders in the sidebar to generate data and see how the regression line changes."
)

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("⚙️ Data Controls")

    # Sliders for user to control data generation
    n_points = st.slider("Number of data points", min_value=10, max_value=500, value=100, step=10)
    noise = st.slider("Noise level", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    true_slope = st.slider("True slope", min_value=-10.0, max_value=10.0, value=2.5, step=0.1)
    true_intercept = st.slider("True intercept", min_value=-20.0, max_value=20.0, value=5.0, step=0.5)
    
    st.header("ℹ️ About")
    st.markdown(
        '''
        This application is built by Gemini to demonstrate linear regression 
        in an interactive way using [Streamlit](https://streamlit.io/).
        '''
    )


# --- Data Generation ---
# Generate data based on a linear relationship with some noise
np.random.seed(42) # for reproducibility
X = np.random.rand(n_points) * 10  # Feature
y = true_slope * X + true_intercept + np.random.randn(n_points) * noise

# Create a pandas DataFrame
df = pd.DataFrame({'X': X, 'y': y})


# --- Linear Regression ---
# Reshape X for scikit-learn
X_reshaped = X.reshape(-1, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X_reshaped, y)

# Get the regression line parameters
slope = model.coef_[0]
intercept = model.intercept_

# Generate points for the regression line
x_line = np.array([df['X'].min(), df['X'].max()])
y_line = slope * x_line + intercept
line_df = pd.DataFrame({'X': x_line, 'y': y_line})


# --- Visualization ---
st.header("Data and Regression Line")

# Create the scatter plot
scatter_plot = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X('X', title='Feature (X)'),
    y=alt.Y('y', title='Target (y)'),
    tooltip=['X', 'y']
).interactive()

# Create the regression line plot
regression_line = alt.Chart(line_df).mark_line(color='red', strokeWidth=3).encode(
    x='X',
    y='y'
)

# Combine the plots
chart = (scatter_plot + regression_line).properties(
    title=f"Scatter Plot with Regression Line (n={n_points}, noise={noise})"
)

st.altair_chart(chart, use_container_width=True)


# --- Display Model Information ---
st.header("Regression Model Details")
st.markdown(f"The calculated regression equation is:")
st.latex(f"y = {slope:.2f}X + {intercept:.2f}")

st.write("---")

# --- Show Raw Data ---
with st.expander("View Raw Data"):
    st.dataframe(df)

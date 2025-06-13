import streamlit as st
import joblib as jl
import pandas as pd
import numpy as np

# Load the pre-trained model
# model = jl.load('best_xgb_model.pkl')
st.set_page_config(page_title="Heart Disease Prediction", page_icon=":heart:", layout="wide")
st.title("Heart Disease Prediction App")
st.markdown("This app predicts the presence of heart disease based on user input.")

# Create multi-page navigation
st.sidebar.title("Navigation")

from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stoggle import stoggle





from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header  
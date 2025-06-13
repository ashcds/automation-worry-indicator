import streamlit as st
import joblib as jl
import pandas as pd
import numpy as np

# Side-bar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Resume", "Projects", "Automation Worry Predictor"])

# -------------------------------------
# Page 1: Biographical Hompage 
# -------------------------------------

if page == "Home":
    st.title("Welcome to Personal Website")
    st.image("my_pic.jpg", width=200)





















# Load the pre-trained model
# model = jl.load('best_xgb_model.pkl')

# Create multi-page navigation


# from streamlit_extras.switch_page_button import switch_page
# from streamlit_extras.stoggle import stoggle





# from streamlit_extras.add_vertical_space import add_vertical_space
# from streamlit_extras.colored_header import colored_header  
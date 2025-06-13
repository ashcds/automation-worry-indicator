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
    st.title("Welcome to my Portfolio")
    st.image("my_pic.jpg", width=200)
    st.header("About Me")
    st.markdown("""
                Hello! I'm Aishwarya (Ash), a Senior Data Analyst in the home appliances industry. I have experience in various domains, including finance, consulting, and manufacturing. I enjoy working with data to solve complex business problems and drive strategic decision-making.

                **Academic Background:**
                I have a bachelors in Mechanical Engineering and Finance from the University of Sydney and a M.S. in Mechanical Engineering from the University of Michigan, Ann Arbor.

                **Career Aspirations:**
                I aspire to transition into a Data Scientist role and work at the intersection of ML/AI and strategy, where I can leverage my technical and business skills to drive strategic initiatives.

                **Interests and Hobbies:**
                In my free time I enjoy playing badminton, latin dancing, reading and travelling to new places. I also love to explore new technologies and stay updated with the latest trends in data science and machine learning.

                """)














# Load the pre-trained model
# model = jl.load('best_xgb_model.pkl')

# Create multi-page navigation


# from streamlit_extras.switch_page_button import switch_page
# from streamlit_extras.stoggle import stoggle





# from streamlit_extras.add_vertical_space import add_vertical_space
# from streamlit_extras.colored_header import colored_header  
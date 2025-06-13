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
    st.title("Welcome to My Portfolio")
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
    

# -------------------------------------
# Page 2: Resume 
# -------------------------------------

if page == "Resume":
    st.title("Aishwarya Cherian")
    st.header("Education")
    st.markdown("""
                **M.S. Data Science**  
                Eastern University, 2024 - 2025

                **M.S. Mechanical Engineering, 4.0**   
                University of Michigan - Ann Arbor, 2021 - 2022

                **B.S. Mechanical Engineering (Honors) / Commerce (Finance)**   
                University of Sydney, 2014 - 2018 
    """)

    st.header("Work Experience")
    st.markdown("""
                **Whirlpool Corporation**   
                *Senior Data Analyst, 2022 - Present*   
                - Led a cross-functional team of developers to successfully deliver project data requirements and prepare data for ingestion into GCP BigQuery for the data strategy implementation   
                - Analyzed data with Postgres SQL and developed dashboards in Looker and Tableau on product development project progression, resources, electronics asset volumes and cost to allow easy access to actionable insights for engineers and EES Leadership reducing their analytics effort by 30%   
                - Supported analytics for electronics Zero-base Plan Complexity Reduction Workstream and overall compliance to reduce legacy complexity by ~10% per year   

                **Commonwealth Bank of Australia**   
                *Data Analyst, Data and Decision Science, 2019 - 2020*   
                - Analyzed large datasets of over 10M records with SQL to understand key drivers of trends in retail banking portfolio   
                - Built interactive Tableau dashboards for management to monitor essential portfolio performance measures   
                - Delivered key insights on the ~$500B home loan portfolio to set tolerances for pertinent risk metrics as part of the Risk Appetite Statement annual refresh to reduce risk by 15 basis points.
                """)

    st.header("Technical Skills")
    st.markdown("""
                **Programming Languages:** Python, SQL, R

                **Data Analysis Tools:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, TensorFlow, Keras

                **Data Visualization:** Tableau, Looker Studio 

                **Databases:** MySQL, PostgreSQL

                **Cloud Platforms:** AWS (S3, EC2), GCP

                **Certifications:** AWS Certified Cloud Practioner, Lean Six Sigma Green Belt (Black Belt Trained)
    """)


# -------------------------------------
# Page 3: Other Projects 
# -------------------------------------

if page == "Projects":
    st.title("My Projects")
    st.markdown("""
                - [Diabetes Risk Predictor] (https://github.com/ashcds/US-diabetes-health-indicators)   

                ** More to come!**
                """)






# Load the pre-trained model
# model = jl.load('best_xgb_model.pkl')

# Create multi-page navigation


# from streamlit_extras.switch_page_button import switch_page
# from streamlit_extras.stoggle import stoggle





# from streamlit_extras.add_vertical_space import add_vertical_space
# from streamlit_extras.colored_header import colored_header  
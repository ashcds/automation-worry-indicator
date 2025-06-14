import streamlit as st
import joblib as jl
import pandas as pd
import numpy as np

# Side-bar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Resume", "Projects", "Capstone Project Overview", "Automation Worry Predictor"])

# -------------------------------------
# Page 1: Biographical Hompage 
# -------------------------------------

if page == "Home":
    st.title("Welcome to My Portfolio")
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
    st.image("my_pic.jpg", width=200)
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

                **Databases and Cloud:** MySQL, PostgreSQL, AWS Redshift, GCP Bigquery   

                **Certifications:** AWS Certified Cloud Practioner, Lean Six Sigma Green Belt (Black Belt Trained)
    """)


# -------------------------------------
# Page 3: Other Projects 
# -------------------------------------

if page == "Projects":
    st.title("My Projects")
    st.markdown("""
                - [Diabetes Risk Predictor](https://github.com/ashcds/US-diabetes-health-indicators)   

                **More to come!**
                """)


# -------------------------------------
# Page 3: Automation Worry Predictor 
# -------------------------------------

if page == "Capstone Project Overview":
    st.title("Capstone Project Overview: Automation Worry Predictor")
    st.markdown("""
                **Problem Statement:**   
                As AI and automation technologies continue to advance at a rapid pace, concerns about their impact on jobs and livelihoods is growing. 
                This project aims to predict the likelihood of individuals feeling worried about job automation so that organizations can take proactive measures to address these concerns, provide opportunities for upskilling and support their employees.
                
                The model is trained on data obtained from PEW Research Center's survey on technology and automation (American Trends Panel Wave 27). 
                
                **Approach:**  
                Problem framed as a supervised binary classification task.   
                - Target variable: `ROBJOB3b_W27` (1 = worried, 0 = not worried)

                Feature selection was done using chi-square test and SHAP to understand which features with the most relevant for model performance. Based on this analysis the following features were selected:
                - `CARS3B_W27`
                - `ROBJOB4B_W27`
                - `CARS7B_W27`
                - `ROBJOB4A_W27`
                - `CARS7A_W27`

                RandomOverSampler was used to balance the dataset as ~70% of the data was of positive class (worried about automation). The model was trained using Logistic Regression, XGBoost and Random Forest classifiers. Cross validation was used to evaluate model performance and hyperparameter tuning was done using GridSearchCV. The best performing model was selected based on F1 score.


                **Results:**
                - Best performing model: XGBoost Classifier with F1 score of 0.72 on test set. The F1 score for the positive class (worried about automation) was 0.80. 
                - Model was saved as a pickle file for deployment.
                """)


# -------------------------------------
# Page 5: Automation Worry Predictor 
# -------------------------------------
if page == "Automation Worry Predictor":
    st.title("Automation Worry Predictor")
    st.markdown("""
                This application predicts the likelihood of individuals feeling worried about job automation based on their responses to a survey. 
                The model is trained on data obtained from PEW Research Center's survey on technology and automation (American Trends Panel Wave 27).
                """)

    # Load the pre-trained model
    model = jl.load('best_xgb_model.pkl')

    # Input fields for user to enter data
    st.header("Please answer the following questions:")
    cars3b = st.selectbox(
        "How ENTHUSIASTIC are you, if at all, about the development of driverless vehicles?", 
        ["--Select an Option--", "Very Enthusiastic", "Somewhat Enthusiastic", "Not Too Enthusiastic", "Not at All Enthusiastic"])
    
    cars7a = st.selectbox(
        "How safe would you feel sharing the road with a driverless passenger vehicle?",
        ["--Select an Option--", "Very Safe", "Somewhat Safe", "Not Too Safe", "Not at All Safe"])

    cars7b = st.selectbox(
        "How safe would you feel sharing the road with a driverless freight truck?", 
        ["--Select an Option--", "Very Safe", "Somewhat Safe", "Not Too Safe", "Not at All Safe"])

    robjob4b = st.selectbox("If robots and computers were able to perform most of the jobs currently being done by humans, would it be likely or unlikely that inequality between rich and poor would be much worse than it is today?", 
                            ["--Select an Option--", "Yes Likely", "No, Not Likely"])

    robjob4a = st.selectbox("If robots and computers were able to perform most of the jobs currently being done by humans, would it be likely or unlikely that people would have a hard time finding things to do with their lives?", 
                            ["--Select an Option--", "Yes Likely", "No, Not Likely"])

    # Convert input data to appropriate format for prediction
    input_data = pd.DataFrame({
        'CARS3B_W27': [1 if cars3b == "Yes" else 0],
        'ROBJOB4B_W27': [1 if robjob4b == "Yes" else 0],
        'CARS7B_W27': [1 if cars7b == "Yes" else 0],
        'ROBJOB4A_W27': [1 if robjob4a == "Yes" else 0],
        'CARS7A_W27': [1 if cars7a == "Yes" else 0]
    })

    # Predict using the loaded model
    if st.button("Predict"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0]
        
        if prediction[0] == 1:
            st.success(f"You are predicted to be worried about job automation with a probability of {probability:.2f}.")
        else:
            st.success(f"You are predicted to not be worried about job automation with a probability of {probability:.2f}.")


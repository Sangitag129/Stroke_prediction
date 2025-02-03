import streamlit as st
import pickle
import pandas as pd

# Function to load custom CSS
def load_css():
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #FF6347;
            color: black;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stTextInput>div>input {
            background-color:rgb(255, 225, 246);
            border: 2px solid #FF6347;
            color: #2F4F4F;
            padding: 10px;
            border-radius: 8px;
        }
        .stSlider>div>div {
            background-color: #FF6347;
        }
        h1 {
            color: #FF6347;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True
    )

# Load custom CSS
load_css()

# Streamlit app UI
st.title('Stroke Prediction')

age = st.number_input('Age')
hypertension = st.number_input('Hypertension')
heart_disease = st.number_input('Heart Disease')
avg_glucose_level = st.number_input('Average Glucose Level')
bmi = st.number_input('BMI')

data = {'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi}

features = pd.DataFrame(data, index=[0])

# Load the model and preprocessor
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocess = pickle.load(f)

# Process the input data
input_data_std = preprocess.transform(features)

# Make prediction
prediction = best_model.predict(input_data_std)

# Display prediction button and result
if st.button('Predict'):
    if prediction[0] == 0:
        st.write('No Stroke', color='green')  # Green text color for no stroke
    else:
        st.write('Stroke', color='red')  # Red text color for stroke

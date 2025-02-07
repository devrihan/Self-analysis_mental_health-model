
import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai  

model = joblib.load("mental_health_model.pkl")
expected_features = joblib.load("feature_names.pkl")  
label_encoders = joblib.load("encoder.pkl")  

# Gemini API Setup
from api_key import API_KEY
import google.generativeai as genai

genai.configure(api_key=API_KEY)

def chat_with_ai(user_message):
    model = genai.GenerativeModel("gemini-pro")  
    response = model.generate_content(user_message)  
    return response.text if response else "Sorry, I couldn't generate a response."

# Streamlit UI
st.title("Mental Health Prediction")

st.write("### Please enter your basic information:")

name = st.text_input("Name")
age = st.number_input("Age", min_value=0)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"])
family_history = st.selectbox("Family History of Mental Health Issues", ["Yes", "No"])
tech_company = st.selectbox("Do you work in a tech company?", ["Yes", "No"])
seek_help = st.selectbox("Have you sought help for mental health issues?", ["Yes", "No"])
mental_health_consequence = st.selectbox("Do you fear negative consequences for mental health disclosure?", ["Yes", "No"])
phys_health_consequence = st.selectbox("Do you fear negative consequences for physical health disclosure?", ["Yes", "No"])
remote_work = st.selectbox("Do you work remotely?", ["Yes", "No"])

if st.button("Predict"):
    user_inputs = {
        "name": name,
        "age": age,
        "gender": gender,
        "employment_status": employment_status,
        "family_history": 1 if family_history == "Yes" else 0,
        "tech_company": 1 if tech_company == "Yes" else 0,
        "seek_help": 1 if seek_help == "Yes" else 0,
        "mental_health_consequence": 1 if mental_health_consequence == "Yes" else 0,
        "phys_health_consequence": 1 if phys_health_consequence == "Yes" else 0,
        "remote_work": 1 if remote_work == "Yes" else 0
    }

    input_data = pd.DataFrame([user_inputs])

    for col in ["gender", "employment_status"]:
        if col in label_encoders:
            encoder = label_encoders[col]
            input_data[col] = encoder.transform(input_data[col])

    st.write("### Processed Input Data for Prediction:")
    st.write(input_data)

    input_data_encoded = input_data.copy()

    for col in expected_features:
        if col not in input_data_encoded:
            input_data_encoded[col] = 0

    input_data_encoded = input_data_encoded[expected_features]

    # Predict
    predictions = model.predict(input_data_encoded)

    if predictions == 1:
        st.write("### Treatment is recommended.")
    else:
        st.write("### No treatment is recommended.")

    st.markdown("[Go to Image Analysis](https://aimedvision.streamlit.app/)", unsafe_allow_html=True)

    st.markdown("[Talk with AI Mentalhelp Bot](https://reflectify-qw367d507-devrihans-projects.vercel.app/)", unsafe_allow_html=True)

# Chatbot for Mental Health Support
st.write("---")
st.subheader("💬 Chat with AI - Mental Health Support")
st.write("Describe your mental health symptoms, and AI will provide guidance.")

user_message = st.text_input("Type your message here:")

if st.button("Ask AI"):
    if user_message:
        response = chat_with_ai(user_message)
        st.write("**AI Response:**")
        st.write(response)
    else:
        st.warning("Please enter a message to chat.")

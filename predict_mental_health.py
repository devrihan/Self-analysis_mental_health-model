import joblib
import numpy as np
import pandas as pd

model = joblib.load("mental_health_model.pkl")
encoders = joblib.load("encoder.pkl")  
feature_names = joblib.load("feature_names.pkl")  

new_data = pd.read_csv("survey.csv")

print("Expected Features (47):", feature_names)
print("Input Features (52):", list(new_data.columns))


def predict_mental_health(symptoms):
    """
    Predicts mental health condition based on user symptoms.
    """

    new_data_encoded = pd.get_dummies(new_data)

    for col in feature_names:
        if col not in new_data_encoded:
            new_data_encoded[col] = 0


    new_data_encoded = new_data_encoded[feature_names]

    predictions = model.predict(new_data_encoded)

    return predictions

# **Example user input (Replace with actual feature values)**
user_symptoms = {
    "work_interference": "Often",
    "benefits": "Yes",
    "anonymity": "No",
    "care_options": "Yes",
    "leave": "Somewhat easy"
}

result = predict_mental_health(user_symptoms)
print("Predicted Mental Health Condition:", result)

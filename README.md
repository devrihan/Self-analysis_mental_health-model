# Self Analysis Mental health model

## Overview
This project is a **mental health prediction system** that leverages **machine learning** to assess whether an individual should seek treatment for mental health conditions. The system features:
- A **Streamlit-based UI** for easy user interaction.
- A **machine learning model** trained on mental health datasets.
- **AI-powered conversational chatbots** for additional insights. Users can converse with AI models to receive guidance and support.
- **Image analysis** for physical health-related assessments.


## Features
### 1. Mental Health Prediction Model
- Predicts whether treatment is recommended based on user input.
- Uses a **pre-trained model** for inference.

### 2. Additional AI Features
- **Gemini AI Chatbot**: Provides mental health insights based on user queries.
- **Mental Health Chatbot**: A conversational AI tool that allows users to discuss mental well-being, ask questions, and receive AI-generated responses in real-time.
- **Image Analysis**: Allows users to upload images for AI-driven assessments related to physical health conditions.


## Dataset Preprocessing Steps
1. **Data Cleaning**: Removed null values, handled missing data.
2. **Feature Encoding**: Converted categorical features to numerical values using label encoding.
3. **Feature Selection**: Selected relevant features impacting mental health.
4. **Normalization**: Scaled data for optimal model performance.


## Model Selection Rationale
- **Algorithm**: Chose a suitable classifier based on performance metrics.
- **Evaluation Metrics**: Measured accuracy, precision, and recall.
- **Optimization**: Fine-tuned hyperparameters for improved results.


## How to Run the Inference Script
### Prerequisites
Ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt
```

### Steps
1. Run the script:
   ```bash
   python predict_mental_health.py
   ```
2. Input the required details when prompted.
3. View the prediction output, which suggests whether treatment is recommended.


## UI Usage Instructions
### **Streamlit Interface**
#### **How to Use:**
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Also, set up the **Gemini API key** inside `api_key.py`.

2. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
3. **Fill in the following details:**
   - Name
   - Age
   - Gender
   - Employment Status
   - Other relevant inputs like `family_history`, `remote_work`, and `mental_health_consequence`.
4. **Click "Predict"** to get the results.
5. **The system will process the inputs and display:**
   - Whether treatment is recommended or not.
   - Additional insights generated via **Gemini AI chatbot**.

🔹 **Note:** Before running the application, ensure you have set up your **Gemini API Key** inside `api_key.py` to enable AI-based insights.


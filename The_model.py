import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')
df = df.drop_duplicates()
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define preprocessing steps for numeric and categorical features
numeric_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']
numeric_transformer = Pipeline(
    steps=[
        ('scaler', StandardScaler())  # Standardize numeric features
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
    ]
)

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a list of classifiers
classifiers = [
    ('K-NN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Logistic Regression', LogisticRegression()),
    ('SVM', SVC())
]

# Streamlit app
st.title("Diabetes Prediction Model")
st.write("## Predict Diabetes")

# User input for prediction
st.write("### Input Features")
gender = st.radio("Gender", ('Male', 'Female'))
age = st.number_input("Age", value=30, min_value=1, max_value=120)
hypertension = st.checkbox("Hypertension")
heart_disease = st.checkbox("Heart Disease")
smoking_history = st.radio("Smoking History", ('Never', 'Former', 'Current'))
bmi = st.number_input("BMI", value=25.0, min_value=10.0, max_value=60.0, step=0.1)
HbA1c_level = st.number_input("HbA1c Level", value=5.0, min_value=3.0, max_value=20.0, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level", value=100, min_value=0, max_value=500)

# Create a dictionary of input features
input_features = {
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_history],
    'bmi': [bmi],
    'HbA1c_level': [HbA1c_level],
    'blood_glucose_level': [blood_glucose_level]
}

# Convert input to DataFrame and reshape for prediction
input_df = pd.DataFrame(input_features)

# Prediction
if st.button("Predict"):
    st.write("### Prediction")
    for name, classifier in classifiers:
        model = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('classifier', classifier)  # Add the classifier
            ]
        )

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Make prediction
        prediction = model.predict(input_df)
        st.write(f"Prediction using {name}: {prediction[0]}")

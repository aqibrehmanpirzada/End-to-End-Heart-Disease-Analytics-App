# from django.forms import SelectDateWidget
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
# Load the pre-trained models
# log_reg_model = joblib.load('log_reg_model.joblib')
# svm_model = joblib.load('svm_model.joblib')
# random_forest_model = joblib.load('random_forest_model.joblib')
# knn_model = joblib.load('knn_model.joblib')

# Function to predict using the SelectDateWidget model
# def predict(model, input_data):
#     return model.predict(input_data)

# Load the dataset
@st.cache
def load_data():
    # Load your dataset here
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    return df

# Sidebar to select plot type
st.sidebar.header('Explore Data')
plot_type = st.sidebar.selectbox('Select Plot Type', ('Age Factor Distribution', 'Box Plot', 'Violin Plot',
                                                       'Scatter Plot', 'Bar Chart', 'Pie Chart', 'Count Plot',
                                                       'Histogram', 'Pair Plot', 'Descriptive Analysis of the Dataset'))

# Load data
df = load_data()

# Display plot based on user selection
st.header('Exploratory Data Analysis of Diabetes Prediction Report')
numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
categorical_features = ['gender', 'smoking_history']
if plot_type == 'Age Factor Distribution':
    fig = px.histogram(df, x='age',y='hypertension', nbins=30, title='Age Factor Distribution 1')
    st.plotly_chart(fig)
    fig = px.histogram(df, x='age',y='heart_disease', nbins=30, title='Age Factor Distribution 2')
    st.plotly_chart(fig)

elif plot_type == 'Box Plot':
    numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']


    for feature in numerical_features:
        fig = px.box(df, x='smoking_history', y=feature, title=f'Box Plot of {feature} by Smoking History')
        st.plotly_chart(fig)
    categorical_features = ['gender', 'smoking_history']

    for feature in categorical_features:
        fig = px.box(df, x=feature, y='age', title=f'Box Plots of Age History by {feature}')
        st.plotly_chart(fig)
elif plot_type == 'Violin Plot':
   
    for numerical_feature in numerical_features:
        for categorical_feature in categorical_features:
            fig = px.violin(df, y=numerical_feature, x=categorical_feature, box=True,
                            title=f'Violin Plot of {numerical_feature} by {categorical_feature}')
            st.plotly_chart(fig)

elif plot_type == 'Scatter Plot':
    fig = px.scatter(df, x='age', y='hypertension', title='Scatter Plot')
    st.plotly_chart(fig)

elif plot_type == 'Bar Chart':
    fig = px.bar(df, x='gender', y='bmi', title='Bar Chart by Gender Over BMI')
    st.plotly_chart(fig)

elif plot_type == 'Pie Chart':
    fig = px.pie(df, names='gender', title='Pie Chart of Gender Distribution')
    st.plotly_chart(fig)

elif plot_type == 'Count Plot':
    fig = px.histogram(df, x='smoking_history', title='Count Plot of Smoking History')
    st.plotly_chart(fig)
    fig = px.histogram(df, x='gender', title='Count Plot of Gender')
    st.plotly_chart(fig)

elif plot_type == 'Histogram':
    fig = px.histogram(df, x='heart_disease', nbins=30, title='Histogram of Average Cars at Home')
    st.plotly_chart(fig)

elif plot_type == 'Pair Plot':
    fig = px.scatter_matrix(df, dimensions=numerical_features,
                             title='Pair Plot')
    st.plotly_chart(fig)

elif plot_type == 'Descriptive Analysis of the Dataset':
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_cols].corr()
    fig = px.imshow(correlation_matrix, color_continuous_scale='magma', title='Correlation Heatmap')
    st.plotly_chart(fig)
    # Summary statistics (e.g., mean, median, etc.)
    summary_stats = df.describe()

    # Display using a bar plot
    st.subheader('Summary Statistics - Bar Plot')
    fig_bar = px.bar(summary_stats, title='Summary Statistics', orientation='h')
    st.plotly_chart(fig_bar)

    # Display using a pie chart
    st.subheader('Summary Statistics - Pie Chart')
    fig_pie = px.pie(names=summary_stats.columns, values=summary_stats.iloc[0], title='Summary Statistics')
    st.plotly_chart(fig_pie)
    # Calculate summary statistics
    summary_stats = df.describe().T

    # Create a bar plot for the mean
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary_stats.index,
        y=summary_stats['mean'],
        name='Mean',
        marker_color='#205ff2'
    ))

    # Create a heatmap for standard deviation
    fig.add_trace(go.Heatmap(
        z=[summary_stats['std']],
        colorscale='Reds',
        name='Standard Deviation'
    ))

    # Create a heatmap for 50th percentile (median)
    fig.add_trace(go.Heatmap(
        z=[summary_stats['50%']],
        colorscale='twilight',
        name='50th Percentile'
    ))

    fig.update_layout(
        title='Summary Statistics',
        xaxis_title='Statistics',
        yaxis_title='Values'
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
# Create a bar plot for the std
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary_stats.index,
        y=summary_stats['std'],
        name='Mean',
        marker_color='#205ff2'
    ))

    # Create a heatmap for standard deviation
    fig.add_trace(go.Heatmap(
        z=[summary_stats['mean']],
        colorscale='Reds',
        name='Mean'
    ))

    # Create a heatmap for 50th percentile (median)
    fig.add_trace(go.Heatmap(
        z=[summary_stats['75%']],
        colorscale='twilight',
        name='75th Percentile'
    ))

    fig.update_layout(
        title='Summary Statistics',
        xaxis_title='Statistics',
        yaxis_title='Values'
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)
        
    # Create a Streamlit DataFrame to display the summary statistics
    summary_stats_styled = summary_stats.style.bar(subset=['mean'], color='#205ff2')\
                                        .background_gradient(subset=['std'], cmap='Reds')\
                                        .background_gradient(subset=['50%'], cmap='coolwarm')

    # Display the styled summary statistics in Streamlit
    st.header('Summary Statistics')
    st.write('Summary statistics of the dataset:')
    st.dataframe(summary_stats_styled)






# # Sidebar to select plot type
# st.sidebar.header('Explore Data')
# plot_type = st.sidebar.selectbox('Select Plot Type', ('Age Factor Distribution', 'Box Plot', 'Violin Plot',
#                                                        'Scatter Plot', 'Bar Chart', 'Pie Chart', 'Count Plot',
#                                                        'Histogram', 'Pair Plot', 'Descriptive Analysis of the Dataset'))
# Streamlit app
st.sidebar.header("Machine Learning Model Predictor")
model_choice = st.sidebar.selectbox("Select a model and enter input for prediction:", ["Logistic Regression", "SVM", "Random Forest", "KNN", "Decision Tree"])


# User input form
st.write("Enter input for prediction:")
input_data = {}
if model_choice == "Logistic Regression":
    # Add input fields relevant to Logistic Regression model
    input_data['feature1'] = st.number_input("Feature 1")
    input_data['feature2'] = st.number_input("Feature 2")
elif model_choice == "SVM":
    # Add input fields relevant to SVM model
    input_data['feature1'] = st.number_input("Feature 1")
    input_data['feature2'] = st.number_input("Feature 2")
elif model_choice == "Random Forest":
    # Add input fields relevant to Random Forest model
    input_data['feature1'] = st.number_input("Feature 1")
    input_data['feature2'] = st.number_input("Feature 2")
elif model_choice == "KNN":
    # Add input fields relevant to KNN model
    input_data['feature1'] = st.number_input("Feature 1")
    input_data['feature2'] = st.number_input("Feature 2")
elif model_choice == "Decision Tree":
    # Add input fields relevant to Decision Tree model
    input_data['feature1'] = st.number_input("Feature 1")
    input_data['feature2'] = st.number_input("Feature 2")

# Convert input to DataFrame and reshape for prediction
input_df = pd.DataFrame(input_data, index=[0])

# # Prediction
# if st.button("Predict"):
#     if model_choice == "Logistic Regression":
#         prediction = predict(log_reg_model, input_df)
#     elif model_choice == "SVM":
#         prediction = predict(svm_model, input_df)
#     elif model_choice == "Random Forest":
#         prediction = predict(random_forest_model, input_df)
#     elif model_choice == "KNN":
#         prediction = predict(knn_model, input_df)
#     elif model_choice == "Decision Tree":
#         prediction = predict(decision_tree_model, input_df)

#     st.write("Prediction:", prediction)

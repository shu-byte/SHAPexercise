import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model training
clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer (new interface)
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)  # note: returns a shap.Explanation object

# Streamlit app
st.title("SHAP Analysis for Breast Cancer Prediction")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.dataframe(classification_report(y_test, y_pred, output_dict=True))

# Summary plot for class 1 (malignant)
st.subheader("Summary Plot for Malignant Class (1)")
fig, ax = plt.subplots()
# shap_values.values shape: (num_samples, num_features, num_classes)
# We plot class 1 shap values for all samples
shap.summary_plot(shap_values.values[:, :, 1], X_test, show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    mean_val = float(X_test[feature].mean())
    input_data[feature] = st.number_input(f"Enter {feature}:", value=mean_val)

input_df = pd.DataFrame(input_data, index=[0])

# Prediction and probability
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1]

st.write(f"**Prediction:** {'Malignant' if prediction == 1 else 'Benign'}")
st.write(f"**Malignant Probability:** {probability:.2f}")

# SHAP values for input instance
shap_values_input = explainer(input_df)

# Force plot for class 1
st.subheader("Force Plot for Malignant Class")
st_shap(shap.force_plot(explainer.expected_value[1], shap_values_input.values[0, :, 1], input_df), height=200, width=1000)

# Decision plot for class 1
st.subheader("Decision Plot for Malignant Class")
st_shap(shap.decision_plot(explainer.expected_value[1], shap_values_input.values[0, :, 1], input_df.columns))

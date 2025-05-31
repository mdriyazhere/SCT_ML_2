import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model and scaler
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# App title
st.title("🛍️ Customer Segmentation App (K-Means Clustering)")

st.write("Enter customer data below to predict their segment.")

# User input
income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=50)
score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Prepare and scale input
user_data = np.array([[income, score]])
user_data_scaled = scaler.transform(user_data)

# Predict cluster
if st.button("Predict Segment"):
    cluster = kmeans.predict(user_data_scaled)[0]
    st.success(f"🧠 Predicted Customer Segment: Cluster {cluster}")

    # Plot existing clusters
    # Optional: load sample data just for visualization
    sample_data = pd.read_csv("Mall_Customers.csv")
    X = sample_data[['Annual Income (k$)', 'Spending Score (1-100)']]
    X_scaled = scaler.transform(X)
    sample_data['Cluster'] = kmeans.predict(X_scaled)

    # Plotting
    fig, ax = plt.subplots()
    for c in range(kmeans.n_clusters):
        cluster_data = sample_data[sample_data['Cluster'] == c]
        ax.scatter(cluster_data['Annual Income (k$)'],
                   cluster_data['Spending Score (1-100)'],
                   label=f"Cluster {c}")
    ax.scatter(income, score, color='black', s=200, marker='X', label='New Customer')
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segments")
    ax.legend()
    st.pyplot(fig)

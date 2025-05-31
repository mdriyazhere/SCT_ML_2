#🛍️ Customer Segmentation using K-Means (Streamlit App)

This project uses the K-Means Clustering algorithm to segment customers of a retail store based on their Annual Income and Spending Score. A Streamlit web app is provided to interactively predict customer segments.

📁 Project Structure
customer-segmentation-app/

├── app.py                  # Streamlit app script

├── kmeans_model.pkl        # Trained KMeans model

├── Mall_Customers (1).csv  # Original dataset (optional, for training only)

├── README.md               # Project documentation


✅ Features
Input Annual Income and Spending Score

Predict customer cluster using trained K-Means model

Clean and interactive UI using Streamlit

📊 Dataset Used
Mall_Customers.csv
Columns:

CustomerID

Gender

Age

Annual Income (k$)

Spending Score (1-100)

Only Annual Income and Spending Score are used for clustering.

🧠 Model Training (Done Separately)
Model is trained using scikit-learn:

python
Copy
Edit
from sklearn.cluster import KMeans
import pandas as pd
import joblib

df = pd.read_csv("Mall_Customers (1).csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

joblib.dump(kmeans, "kmeans_model.pkl")
🚀 How to Run the App
Clone or download the repository.

Install required packages:

bash
Copy
Edit
pip install streamlit scikit-learn joblib
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
🖥️ App UI
Move sliders to select Annual Income and Spending Score

Click Predict Segment to see the assigned customer cluster

📦 Requirements
Python 3.7+

streamlit

scikit-learn

joblib

📌 Notes
This version assumes the model is trained without scaling.

If you want to use scaled features or show cluster plots, those can be added back with scaler.pkl and data loading.

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load log data
data = pd.read_csv("logs.csv")

# Select relevant features for anomaly detection
features = data[["cpu_usage", "error_count", "requests"]]

# Create and train anomaly detection model
model = IsolationForest(contamination=0.25, random_state=42)
data["anomaly"] = model.fit_predict(features)

print("\nAnomaly Detection Results:\n")
print(data)

# Visualize anomalies
plt.figure()
plt.scatter(range(len(data)), data["cpu_usage"], c=data["anomaly"])
plt.xlabel("Time Index")
plt.ylabel("CPU Usage")
plt.title("AI Log Anomaly Detection")
plt.show()

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# Paths
BASE_PATH = r"C:\Users\Keerthana\OneDrive\Desktop\customerchurn"
MODEL_PATH = os.path.join(BASE_PATH, "models", "churn_model.pkl")
FIG_PATH = os.path.join(BASE_PATH, "reports", "figures")

os.makedirs(FIG_PATH, exist_ok=True)

# Load model
pipeline = joblib.load(MODEL_PATH)

# Get feature names
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

cat_features = preprocessor.transformers_[0][1].get_feature_names_out()
num_features = preprocessor.transformers_[1][2]
feature_names = list(cat_features) + list(num_features)

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]

# Plot
plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top 15 Feature Importances for Customer Churn")
plt.tight_layout()
plt.savefig(os.path.join(FIG_PATH, "feature_importance.png"))
plt.close()

print("✅ Feature importance saved")

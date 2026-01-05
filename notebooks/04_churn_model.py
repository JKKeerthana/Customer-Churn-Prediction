import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

# Load data
df = pd.read_csv(
    r"C:\Users\Keerthana\OneDrive\Desktop\customerchurn\data\processed\cleaned_telco_churn.csv"
)

# Standardize column names
df.columns = df.columns.str.lower()

# Target
y = df["churn"]

# Drop non-features
X = df.drop(columns=["customerid", "churn", "tenure_group"], errors="ignore")

# Identify column types
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# Save model
MODEL_PATH = r"C:\Users\Keerthana\OneDrive\Desktop\customerchurn\models"
os.makedirs(MODEL_PATH, exist_ok=True)

joblib.dump(pipeline, os.path.join(MODEL_PATH, "churn_model.pkl"))

print("✅ Model saved successfully")

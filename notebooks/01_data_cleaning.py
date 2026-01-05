import pandas as pd

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = pd.read_csv(
    r"C:\Users\Keerthana\OneDrive\Desktop\customerchurn\data\raw\telco_churn.csv"
)

print("Data loaded successfully")
print("Original shape:", df.shape)
print("Columns in dataset:", df.columns.tolist())

# -------------------------------
# Step 2: Standardize column names
# -------------------------------
# lowercase, replace spaces with underscore
df.columns = df.columns.str.lower().str.replace(" ", "_")
print("Columns after cleaning:", df.columns.tolist())

# -------------------------------
# Step 3: Convert churn to binary
# -------------------------------
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# -------------------------------
# Step 4: Convert totalcharges to numeric
# -------------------------------
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

# -------------------------------
# Step 5: Handle missing values
# -------------------------------
# Fill missing totalcharges with median
df['totalcharges'].fillna(df['totalcharges'].median(), inplace=True)

# -------------------------------
# Step 6: Create tenure groups
# -------------------------------
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-1 year', '1-2 years', '2-4 years', '4+ years']
)

# -------------------------------
# Step 7: Save cleaned data
# -------------------------------
df.to_csv(
    r"C:\Users\Keerthana\OneDrive\Desktop\customerchurn\data\processed\cleaned_telco_churn.csv",
    index=False
)

print("Cleaned data saved successfully")
print("Cleaned data shape:", df.shape)
print(df.head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(
    r"C:\Users\Keerthana\OneDrive\Desktop\customerchurn\data\processed\cleaned_telco_churn.csv"
)

# ---- STANDARDIZE COLUMNS ----
# Ensure lowercase column names
df.columns = df.columns.str.lower()

# ---- FIX CHURN COLUMN ----
# Handle Yes/No, yes/no, 0/1 safely
if df["churn"].dtype == "object":
    df["churn"] = (
        df["churn"]
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

# ---- CHECK ----
print("Churn distribution:")
print(df["churn"].value_counts(normalize=True))

# ---- PLOTS ----
sns.barplot(x="contract", y="churn", data=df)
plt.title("Churn Rate by Contract Type")
plt.show()

sns.barplot(x="tenure_group", y="churn", data=df)
plt.title("Churn Rate by Tenure Group")
plt.show()

sns.boxplot(x="churn", y="monthlycharges", data=df)
plt.title("Monthly Charges by Churn")
plt.show()

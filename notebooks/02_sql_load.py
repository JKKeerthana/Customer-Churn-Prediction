import sqlite3
import pandas as pd
import os

# Absolute paths
BASE_PATH = r"C:\Users\Keerthana\OneDrive\Desktop\customerchurn\data\processed"
DB_PATH = os.path.join(BASE_PATH, "churn_db.sqlite")
CSV_PATH = os.path.join(BASE_PATH, "cleaned_telco_churn.csv")

# Connect to SQLite database (will create file if folder exists)
conn = sqlite3.connect(DB_PATH)

# Load cleaned data
df = pd.read_csv(CSV_PATH)

# Write to SQL table
df.to_sql("customers", conn, if_exists="replace", index=False)

conn.close()

print("✅ Data loaded into SQLite successfully")
print(f"📁 Database created at: {DB_PATH}")

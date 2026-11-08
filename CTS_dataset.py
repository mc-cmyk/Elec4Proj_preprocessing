import pandas as pd



df = pd.read_csv("C:/Users/Mc Marteen/Downloads/customer orders.csv")

print("===== DATASET STRUCTURE =====")
df.info()

print("\n===== SUMMARY STATISTICS =====")
print(df.describe(include="all"))

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== NUMBER OF ROWS AND COLUMNS =====")
print("Rows and Columns:", df.shape)

print("\n===== RANDOM 30 ROWS =====")
print(df.sample(30, random_state=42))


# 2. Check duplicates, view duplicate rows, and drop duplicates
print("\n===== DUPLICATE COUNT =====")
print("Number of duplicate rows:", df.duplicated().sum())

print("\n===== DUPLICATE ROWS =====")
print(df[df.duplicated()])

df = df.drop_duplicates()


# 3. Missing values per column and fill missing values
print("\n===== MISSING VALUES PER COLUMN =====")
print(df.isnull().sum())

# Fill missing Rating with median
df["Rating"] = df["Rating"].fillna(df["Rating"].median())

# Fill missing Customer Names with "Unknown"
df["Customer"] = df["Customer"].fillna("Unknown")

# Forward fill missing Remarks
df["Remarks"] = df["Remarks"].ffill()


# 4. Replace inconsistent values
df["PaymentMethod"] = df["PaymentMethod"].replace({
    "Gcash": "GCASH",
    "gcash": "GCASH",
    "GCash": "GCASH"
})


# 5. Convert TransactionDate to datetime and Rating to int
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["Rating"] = df["Rating"].astype(int)

print("\n===== DATA TYPES AFTER CONVERSION =====")
print(df.dtypes)


# 6. Drop Remarks column and remove rows where Quantity is 0 or negative
df = df.drop(columns=["Remarks"])
df = df[df["Quantity"] > 0]


# 7. Create calculated columns
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["FinalPrice"] = df["TotalPrice"] * 1.12
df["SalesCategory"] = df["FinalPrice"].apply(lambda x: "High" if x > 3000 else "Low")


# 8. Show all rows in the dataset and its structure
print("\n===== FINAL DATASET =====")
print(df.to_string(index=False))

print("\n===== FINAL DATASET STRUCTURE =====")
df.info()
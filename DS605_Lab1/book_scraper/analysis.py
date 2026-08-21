import pandas as pd

# Load cleaned dataset
df = pd.read_csv("books_cleaned.csv")

print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSummary Statistics:")
print(df.describe())

print("\n")

# ---------------------------------------
# Category Counts
# ---------------------------------------
print("=" * 60)
print("BOOKS PER CATEGORY")
print("=" * 60)

print(df["Category"].value_counts())

print("\n")

# ---------------------------------------
# Average Price by Category
# ---------------------------------------
print("=" * 60)
print("AVERAGE PRICE BY CATEGORY")
print("=" * 60)

print(
    df.groupby("Category")["Price"]
    .mean()
    .sort_values(ascending=False)
)

print("\n")

# ---------------------------------------
# Rating Distribution
# ---------------------------------------
print("=" * 60)
print("RATING DISTRIBUTION")
print("=" * 60)

print(df["Rating"].value_counts().sort_index())

print("\n")

# ---------------------------------------
# Top 10 Expensive Books
# ---------------------------------------
print("=" * 60)
print("TOP 10 MOST EXPENSIVE BOOKS")
print("=" * 60)

print(
    df[["Title","Category","Price"]]
    .sort_values("Price",ascending=False)
    .head(10)
)

print("\n")

# ---------------------------------------
# Best Value Books
# ---------------------------------------
print("=" * 60)
print("TOP 10 VALUE BOOKS")
print("=" * 60)

print(
    df[["Title","Price","Rating","value_score"]]
    .sort_values("value_score",ascending=False)
    .head(10)
)

print("\n")

# ---------------------------------------
# Recommended Books
# ---------------------------------------
print("=" * 60)
print("RECOMMENDED BOOKS")
print("=" * 60)

print(df["recommended"].value_counts())

print("\n")

# ---------------------------------------
# Stock Information
# ---------------------------------------
print("=" * 60)
print("AVERAGE STOCK")
print("=" * 60)

print("Average Stock:", round(df["Stock"].mean(),2))

print("Maximum Stock:", df["Stock"].max())

print("Minimum Stock:", df["Stock"].min())

print("\nAnalysis Completed Successfully!")
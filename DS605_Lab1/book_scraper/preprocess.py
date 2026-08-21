import pandas as pd

# Load dataset
df = pd.read_csv("books.csv")

# -------------------------
# Dataset Information
# -------------------------
print("=" * 50)
print("RAW DATA REPORT")
print("=" * 50)

print("Total Records:", len(df))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate UPCs:")
print(df.duplicated(subset="UPC").sum())

# -------------------------
# Remove extra spaces
# -------------------------
for column in df.select_dtypes(include="object"):
    df[column] = df[column].str.strip()

# -------------------------
# Remove duplicate books
# -------------------------
df = df.drop_duplicates(subset="UPC")

# -------------------------
# Handle missing descriptions
# -------------------------
df["Description"] = df["Description"].fillna("No description available")

# -------------------------
# Convert Price
# -------------------------
df["Price"] = (
    df["Price"]
    .str.replace("£", "", regex=False)
    .astype(float)
)

# -------------------------
# Convert Rating
# -------------------------
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["Rating"] = df["Rating"].map(rating_map)

# -------------------------
# Extract Stock Count
# -------------------------
df["Stock"] = (
    df["Availability"]
    .str.extract(r"(\d+)", expand=False)
    .fillna(0)
    .astype(int)
)

# -------------------------
# Feature 1
# -------------------------
df["description_word_count"] = df["Description"].apply(
    lambda x: len(str(x).split())
)

# -------------------------
# Feature 2
# -------------------------
def price_band(price):
    if price < 20:
        return "Cheap"
    elif price < 40:
        return "Medium"
    else:
        return "Expensive"

df["price_band"] = df["Price"].apply(price_band)

# -------------------------
# Feature 3
# -------------------------
df["affordability_score"] = (
    100 - df["Price"]
).round(2)

# -------------------------
# Feature 4
# -------------------------
df["value_score"] = (
    df["Rating"] / df["Price"]
).round(2)

# -------------------------
# Feature 5
# -------------------------
df["recommended"] = df.apply(
    lambda row: "Yes"
    if row["Rating"] >= 4 and row["Price"] < 30
    else "No",
    axis=1
)

# -------------------------
# Save Cleaned Dataset
# -------------------------
df.to_csv("books_cleaned.csv", index=False)

print("\nCleaning Completed Successfully!")
print("Final Records:", len(df))
print("Saved as books_cleaned.csv")
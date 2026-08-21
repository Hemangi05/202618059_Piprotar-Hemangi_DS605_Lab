import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load cleaned dataset
df = pd.read_csv("books_cleaned.csv")

# ----------------------------
# 1. Price Distribution
# ----------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["Price"], bins=15, edgecolor="black")
plt.title("Price Distribution")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")
plt.grid(True)
plt.savefig("price_distribution.png")
plt.close()

# ----------------------------
# 2. Rating Distribution
# ----------------------------
plt.figure(figsize=(6, 4))
df["Rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig("rating_distribution.png")
plt.close()

# ----------------------------
# 3. Average Price by Category
# ----------------------------
plt.figure(figsize=(10, 6))
avg_price = df.groupby("Category")["Price"].mean().sort_values()
avg_price.plot(kind="barh")
plt.title("Average Price by Category")
plt.xlabel("Average Price (£)")
plt.tight_layout()
plt.savefig("average_price_category.png")
plt.close()

# ----------------------------
# 4. Price vs Rating
# ----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(df["Price"], df["Rating"])
plt.title("Price vs Rating")
plt.xlabel("Price (£)")
plt.ylabel("Rating")
plt.grid(True)
plt.savefig("price_vs_rating.png")
plt.close()

# ----------------------------
# 5. Word Cloud
# ----------------------------
text = " ".join(df["Description"].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Book Descriptions")
plt.savefig("wordcloud.png")
plt.close()

print("All plots generated successfully!")
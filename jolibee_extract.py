"""
Build the clean branch-level dataset for the sentiment analytics project.
=========================================================================
Input : bwandowando/food-panda-restaurant-reviews (v13)
Output: five CSVs in ./output ready for preprocessing, modelling and the dashboard.

Deliberately NOT done here: tokenization, stopword removal, stemming.
That is Objective 1 on your deck and should live in your own preprocessing
notebook so you can show the work. This script only selects, labels and splits.

Change BRAND at the top to switch chains (McDonald's, Mang Inasal, KFC, Chowking...).
"""

import os
import re

import kagglehub
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- settings
BRAND = "Jollibee"
OUTDIR = "output"
MIN_REVIEWS_PER_BRANCH = 30     # branches below this are flagged, not deleted
NEGATIVE_MAX_STAR = 2
NEUTRAL_STAR = 3
TEST_FRACTION = 0.20
SEED = 42

INVISIBLE = r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)
os.makedirs(OUTDIR, exist_ok=True)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def clean_text(s):
    """Strip invisible characters and normalise whitespace. Keeps case and emoji."""
    if pd.isna(s):
        return ""
    s = re.sub(INVISIBLE, "", str(s))
    return re.sub(r"\s+", " ", s).strip()


def branch_label(complete_name, brand):
    """'Jollibee - SM Megamall' -> 'SM Megamall'. Tolerates missing/odd separators."""
    c = clean_text(complete_name)
    stripped = re.sub(rf"^\s*{re.escape(brand)}\s*[-\u2013\u2014]?\s*", "", c, flags=re.I).strip()
    return stripped if stripped else c


# ------------------------------------------------------------------- load
rule("LOAD")
path = kagglehub.dataset_download("bwandowando/food-panda-restaurant-reviews")
restos = pd.read_csv(os.path.join(path, "ph_restos_2025.csv"), low_memory=False)
reviews = pd.read_csv(os.path.join(path, "ph_reviews_2025.csv"), low_memory=False)
print(f"stores  {restos.shape}")
print(f"reviews {reviews.shape}")

restos["CompleteStoreName"] = restos["CompleteStoreName"].map(clean_text)


# ------------------------------------------------- duplicate StoreId audit
# 67 StoreIds appear twice with DIFFERENT branch names. Blind deduping throws
# away a real branch, so check whether this brand is affected before dropping.
rule("DUPLICATE StoreId AUDIT")

dup_mask = restos.duplicated("StoreId", keep=False)
brand_prefix = restos["CompleteStoreName"].str.match(rf"^\s*{re.escape(BRAND)}\b", case=False, na=False)
affected = restos[dup_mask & brand_prefix]

print(f"Duplicate StoreId rows dataset-wide: {dup_mask.sum()}")
print(f"...of which belong to {BRAND}: {len(affected)}")
if len(affected):
    print("\n!! These branches share an id. One will be lost on dedupe -- review manually:")
    print(affected.sort_values("StoreId")[["StoreId", "CompleteStoreName", "City"]].to_string(index=False))
else:
    print(f">> {BRAND} is unaffected. Safe to dedupe.")

restos = restos.drop_duplicates(subset="StoreId", keep="first").copy()


# --------------------------------------------------------------- brand filter
# Prefix match, not a " - " split: it survives "Jollibee- X" and "Jollibee X"
# variants that broke the generic splitter.
rule(f"FILTER TO {BRAND}")

stores = restos[
    restos["CompleteStoreName"].str.match(rf"^\s*{re.escape(BRAND)}\b", case=False, na=False)
].copy()
stores["Brand"] = BRAND
stores["Branch"] = stores["CompleteStoreName"].map(lambda c: branch_label(c, BRAND))

print(f"Matched {len(stores)} stores.")
print("\nEYEBALL THIS LIST -- a prefix match can catch unrelated names:")
print(stores[["StoreId", "CompleteStoreName", "Branch", "City"]]
      .sort_values("Branch").to_string(index=False))


# ------------------------------------------------------------------- join
rule("JOIN")

df = reviews.merge(
    stores[["StoreId", "Brand", "Branch", "CompleteStoreName", "City", "FoodType"]],
    on="StoreId", how="inner",
)
print(f"Reviews for {BRAND}: {len(df)}")

before = len(df)
df = df.drop_duplicates(subset="uuid", keep="first")
print(f"Duplicate uuids removed: {before - len(df)}")


# ------------------------------------------------------------------ labels
rule("LABELS")

df["rating"] = pd.to_numeric(df["overall"], errors="coerce")
df = df[df["rating"].notna()].copy()


def to_sentiment(x):
    if x <= NEGATIVE_MAX_STAR:
        return "Negative"
    if x == NEUTRAL_STAR:
        return "Neutral"
    return "Positive"


df["sentiment"] = df["rating"].map(to_sentiment)

# rider is effectively binary (1 or 5), so treat it as a flag, not a scale.
df["rider_rating"] = pd.to_numeric(df["rider"], errors="coerce")
df["rider_blamed"] = df["rider_rating"] <= 2
# True when the complaint is NOT attributable to the delivery rider.
# Missing rider score counts as attributable to the branch.
df["branch_attributable"] = ~df["rider_blamed"].fillna(False)

df["text_raw"] = df["text"]
df["text_clean"] = df["text"].map(clean_text)
df["word_count"] = df["text_clean"].str.split().str.len()
df["is_short"] = df["word_count"] < 3          # "No spoon", "Took long" -- kept, flagged
df["review_date"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

df = df[df["text_clean"].str.len() > 0].copy()

print(df["sentiment"].value_counts().to_frame("n")
      .assign(pct=lambda d: d.n / d.n.sum()).to_string())
print(f"\nShort reviews (<3 words), flagged not dropped: {df['is_short'].sum()} ({df['is_short'].mean():.1%})")
print(f"Rider-blamed reviews                          : {df['rider_blamed'].sum()}")
print(f"Date range: {df['review_date'].min().date()} -> {df['review_date'].max().date()}")


# ----------------------------------------------------------- branch quality
rule("BRANCH VIABILITY")

branch_stats = df.groupby(["StoreId", "Branch", "City"]).agg(
    n_reviews=("uuid", "size"),
    n_negative=("sentiment", lambda s: (s == "Negative").sum()),
    n_neutral=("sentiment", lambda s: (s == "Neutral").sum()),
    n_positive=("sentiment", lambda s: (s == "Positive").sum()),
    mean_rating=("rating", "mean"),
).reset_index()

branch_stats["pct_negative"] = branch_stats.n_negative / branch_stats.n_reviews
branch_stats["usable"] = branch_stats.n_reviews >= MIN_REVIEWS_PER_BRANCH
branch_stats = branch_stats.sort_values("n_reviews", ascending=False)

df["branch_usable"] = df["StoreId"].isin(branch_stats.loc[branch_stats.usable, "StoreId"])

print(f"Branches: {len(branch_stats)}   usable (>={MIN_REVIEWS_PER_BRANCH} reviews): "
      f"{branch_stats.usable.sum()}")
print(branch_stats.to_string(index=False))


# ---------------------------------------------------------- train/test split
rule("TRAIN / TEST SPLIT")

model_df = df.copy()   # was: df[~df["is_short"]].copy()
test = model_df.groupby("sentiment", group_keys=False).sample(frac=TEST_FRACTION, random_state=SEED)
train = model_df.drop(test.index)

print(f"Modelling pool (short reviews excluded): {len(model_df)}")
print(f"train {len(train)}  |  test {len(test)}")
print("\nClass balance preserved:")
print(pd.DataFrame({
    "train": train.sentiment.value_counts(normalize=True),
    "test": test.sentiment.value_counts(normalize=True),
}).to_string())

baseline = test.sentiment.value_counts(normalize=True).max()
print(f"\n>> Majority-class baseline on the test set: {baseline:.1%}")
print(">> Report macro-F1 and per-class recall. Neutral will be your weak class.")


# ------------------------------------------------------- complaint corpus
rule("COMPLAINT CORPUS")

# Negative AND not the rider's fault -- this is what the keyword ranking runs on.
complaints = df[(df.sentiment == "Negative") & df.branch_attributable & df.branch_usable].copy()
print(f"Negative reviews attributable to the branch, in usable branches: {len(complaints)}")
print("\nPer branch:")
print(complaints.groupby("Branch").size().sort_values(ascending=False).to_string())


# ---------------------------------------------------------------- write out
rule("WRITING FILES")

cols = ["uuid", "StoreId", "Brand", "Branch", "CompleteStoreName", "City", "FoodType",
        "review_date", "text_raw", "text_clean", "word_count", "is_short",
        "rating", "sentiment", "rider_rating", "rider_blamed", "branch_attributable",
        "branch_usable"]

targets = {
    f"{BRAND.lower()}_all.csv": df[cols],
    f"{BRAND.lower()}_train.csv": train[cols],
    f"{BRAND.lower()}_test.csv": test[cols],
    f"{BRAND.lower()}_complaints.csv": complaints[cols],
    f"{BRAND.lower()}_branch_summary.csv": branch_stats,
}
for name, frame in targets.items():
    p = os.path.join(OUTDIR, name)
    # utf-8-sig so Excel renders Filipino text and emoji correctly on Windows
    frame.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  {p}  ({len(frame)} rows)")


rule("FOR THE DECK")
print(f"""
Slide 5   {len(df)} reviews, {branch_stats.usable.sum()} branches, {BRAND}
          {df['review_date'].min().date()} to {df['review_date'].max().date()}
Slide 10  majority-class baseline {baseline:.1%} on the held-out test set
          train {len(train)} / test {len(test)}, stratified by sentiment
Slide 8   complaint ranking runs on {len(complaints)} branch-attributable
          negative reviews ({df['rider_blamed'].sum()} rider complaints excluded)

Next: preprocessing (Objective 1) on text_clean, then TF-IDF + classifier.
Keep is_short and rider_blamed as columns so you can justify every exclusion.
""")
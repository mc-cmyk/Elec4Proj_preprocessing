"""
Foodpanda PH recon -- FIX PASS
==============================
Corrects three failures in the first run:

  BUG 1  ph_restos_2025.csv has no StoreName column, so 'brand' resolved to the
         full "Brand - Branch" string and every branch looked unique.
         FIX: derive brand by splitting CompleteStoreName on " - ",
              and VALIDATE that split against restos.csv, which has both columns.

  BUG 2  The rating column is called 'overall', not 'rating'. Q5 and Q7 never ran.
         FIX: use overall / restaurant_food / rider explicitly.

  BUG 3  Duplicate StoreIds inflated the merge (gained 1187 rows).
         FIX: drop_duplicates on StoreId before joining.

NEW: separates food complaints from rider complaints, and separates
     Tagalog from Bisaya.
"""

import os
import re

import kagglehub
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 90)

MIN_REVIEWS_PER_BRANCH = 30
MIN_NEGATIVES_PER_BRANCH = 15
NEGATIVE_MAX_STAR = 2
NEUTRAL_STAR = 3

# Zero-width and invisible characters seen in this data (e.g. "\u2060PICKUP COFFEE").
INVISIBLE = r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]"


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def clean(s):
    s = re.sub(INVISIBLE, "", str(s))
    return re.sub(r"\s+", " ", s).strip()


def brand_of(complete_name):
    """'Jollibee - SM Megamall' -> 'Jollibee'.  Single-location shops pass through."""
    return clean(complete_name).split(" - ")[0].strip()


path = kagglehub.dataset_download("bwandowando/food-panda-restaurant-reviews")
print("Path:", path)

restos = pd.read_csv(os.path.join(path, "ph_restos_2025.csv"), low_memory=False)
reviews = pd.read_csv(os.path.join(path, "ph_reviews_2025.csv"), low_memory=False)
legacy = pd.read_csv(os.path.join(path, "restos.csv"), low_memory=False)  # has real StoreName


# ------------------------------------------------- BUG 1: derive + validate brand
rule("BUG 1 FIX  Deriving brand, and checking the derivation is actually right")

legacy["_derived"] = legacy["CompleteStoreName"].map(brand_of)
legacy["_actual"] = legacy["StoreName"].map(clean)
match = (legacy["_derived"].str.casefold() == legacy["_actual"].str.casefold())
print(f"restos.csv has both StoreName and CompleteStoreName -- using it as ground truth.")
print(f"Split on ' - ' reproduces the real StoreName: {match.sum()}/{len(legacy)} ({match.mean():.1%})")
if (~match).any():
    print("\nMismatches (check these are harmless before trusting the split):")
    print(legacy.loc[~match, ["CompleteStoreName", "_actual", "_derived"]].head(15).to_string(index=False))


# ---------------------------------------------------- BUG 3: dedupe before merge
rule("BUG 3 FIX  Deduplicating the store table")

print(f"ph_restos_2025 rows: {len(restos)}   unique StoreId: {restos['StoreId'].nunique()}")
dupes = restos[restos.duplicated("StoreId", keep=False)].sort_values("StoreId")
print(f"Duplicate StoreId rows: {len(dupes)}")
if len(dupes):
    print(dupes.head(8).to_string(index=False))

restos = restos.drop_duplicates(subset="StoreId", keep="first").copy()
restos["Brand"] = restos["CompleteStoreName"].map(brand_of)
restos["BrandKey"] = restos["Brand"].str.casefold()
restos["Branch"] = restos["CompleteStoreName"].map(clean)
print(f"After dedupe: {len(restos)} stores, {restos['BrandKey'].nunique()} distinct brands")


# --------------------------------------------------------------- Q2 REDO: depth
rule("Q2 REDO  CHAIN DEPTH")

depth = (restos.groupby("BrandKey")
         .agg(brand=("Brand", "first"), branches=("StoreId", "nunique"))
         .sort_values("branches", ascending=False))

print(f"Brands with >1 branch : {(depth.branches > 1).sum()}")
print(f"Brands with >=5       : {(depth.branches >= 5).sum()}")
print(f"Brands with >=10      : {(depth.branches >= 10).sum()}")
print("\nDeepest 30 chains:")
print(depth.head(30).to_string(index=False))


# ----------------------------------------------------------- BUG 2 / Q5: ratings
rule("Q5 REDO  RATING DISTRIBUTION  (column is 'overall')")

for col in ["overall", "restaurant_food", "rider"]:
    s = pd.to_numeric(reviews[col], errors="coerce")
    print(f"\n{col}:  present {s.notna().sum()} ({s.notna().mean():.1%})   mean {s.mean():.2f}")
    print(s.value_counts().sort_index().to_string())


def to_sent(x):
    if pd.isna(x):
        return np.nan
    if x <= NEGATIVE_MAX_STAR:
        return "Negative"
    if x == NEUTRAL_STAR:
        return "Neutral"
    return "Positive"


reviews["_overall"] = pd.to_numeric(reviews["overall"], errors="coerce")
reviews["_food"] = pd.to_numeric(reviews["restaurant_food"], errors="coerce")
reviews["_rider"] = pd.to_numeric(reviews["rider"], errors="coerce")
reviews["_sent"] = reviews["_overall"].map(to_sent)

dist = reviews["_sent"].value_counts()
print("\n3-class distribution (from 'overall'):")
print(dist.to_frame("n").assign(pct=lambda d: d.n / d.n.sum()).to_string())

baseline = dist.max() / dist.sum()
print(f"\n>> MAJORITY-CLASS BASELINE = {baseline:.1%}  (always predict '{dist.idxmax()}')")
print(">> Put this number on slide 10. Any accuracy you report must beat it.")
print(">> Lead with macro-F1 + per-class recall, not accuracy.")


# ------------------------------------------------------ NEW: food vs rider blame
rule("NEW  FOOD COMPLAINTS vs DELIVERY COMPLAINTS")

both = reviews["_food"].notna() & reviews["_rider"].notna()
print(f"Reviews scoring BOTH food and rider: {both.sum()} ({both.mean():.1%})")

food_bad = reviews["_food"] <= NEGATIVE_MAX_STAR
rider_bad = reviews["_rider"] <= NEGATIVE_MAX_STAR
print(f"\nFood rated negative        : {food_bad.sum()}")
print(f"Rider rated negative       : {rider_bad.sum()}")
print(f"Food bad BUT rider fine    : {(food_bad & (reviews['_rider'] >= 4)).sum()}")
print(f"Rider bad BUT food fine    : {(rider_bad & (reviews['_food'] >= 4)).sum()}")
print("\n>> The 'rider bad, food fine' rows are delivery complaints wrongly counted")
print(">> against the branch. Exclude them from branch complaint ranking.")

print("\nSample FOOD-only complaints (what your keyword extraction should see):")
for s in reviews.loc[food_bad & (reviews["_rider"] >= 4), "text"].dropna().head(8):
    print("  -", clean(s)[:150])


# --------------------------------------------------------- Q7 REDO: the decision
rule("Q7 REDO  CHAIN VIABILITY  <-- THE DECIDING TABLE")

reviews["_txt_ok"] = reviews["text"].notna() & (reviews["text"].astype(str).str.strip() != "")
m = reviews[reviews["_txt_ok"]].merge(
    restos[["StoreId", "Brand", "BrandKey", "Branch", "City", "FoodType"]],
    on="StoreId", how="inner",
)
print(f"Merged: {len(m)} rows from {reviews['_txt_ok'].sum()} reviews with text")
print(f"(should be equal or slightly fewer -- if larger, dedupe failed)")

m["_neg"] = m["_overall"] <= NEGATIVE_MAX_STAR
m["_food_neg"] = m["_food"] <= NEGATIVE_MAX_STAR

per_branch = m.groupby(["BrandKey", "StoreId"]).agg(
    brand=("Brand", "first"),
    branch=("Branch", "first"),
    city=("City", "first"),
    n_reviews=("_txt_ok", "sum"),
    n_neg=("_neg", "sum"),
    n_food_neg=("_food_neg", "sum"),
).reset_index()

summary = per_branch.groupby("BrandKey").agg(
    brand=("brand", "first"),
    branches=("StoreId", "nunique"),
    total_reviews=("n_reviews", "sum"),
    total_neg=("n_neg", "sum"),
    median_neg=("n_neg", "median"),
)
summary["ok_branches"] = per_branch.groupby("BrandKey").apply(
    lambda g: int(((g.n_reviews >= MIN_REVIEWS_PER_BRANCH) &
                   (g.n_neg >= MIN_NEGATIVES_PER_BRANCH)).sum())
)

viable = summary[summary.ok_branches >= 3].sort_values(
    ["ok_branches", "total_neg"], ascending=False)

print(f"\nBar: >=3 branches each with >={MIN_REVIEWS_PER_BRANCH} reviews "
      f"and >={MIN_NEGATIVES_PER_BRANCH} negatives\n")
if len(viable) == 0:
    print("!! NO CHAIN CLEARS THE BAR. Closest candidates:")
    print(summary.sort_values("total_neg", ascending=False).head(20).to_string())
else:
    print(f"{len(viable)} chain(s) clear it:\n")
    print(viable.head(20).to_string())
    best = viable.index[0]
    print(f"\n>> RECOMMENDED CHAIN: {viable.loc[best, 'brand']}")
    print(per_branch[per_branch.BrandKey == best]
          .sort_values("n_reviews", ascending=False)
          .loc[:, ["branch", "city", "n_reviews", "n_neg", "n_food_neg"]]
          .to_string(index=False))


# --------------------------------------------------- Q8 REDO: Tagalog vs Bisaya
rule("Q8 REDO  TAGALOG vs BISAYA  (your samples had both)")

TAGALOG = ["ang", "ng", "mga", "ay", "po", "hindi", "walang", "sobrang", "talaga",
           "lang", "yung", "ako", "namin", "kasi", "masarap", "malamig", "matagal",
           "kulang", "mali", "sana", "grabe", "salamat", "naman", "dito", "kami"]
BISAYA = ["nindot", "kaayo", "kaayong", "kaayug", "daghan", "pud", "gyud", "lami",
          "wala", "naa", "unsa", "pila", "sulod", "ila", "ako", "nimo", "ninyo",
          "maayo", "dili", "pag", "akong", "imong"]

txt = m["text"].astype(str)
tag_pat = r"\b(?:" + "|".join(TAGALOG) + r")\b"
bis_pat = r"\b(?:" + "|".join(BISAYA) + r")\b"
has_tag = txt.str.contains(tag_pat, case=False, regex=True)
has_bis = txt.str.contains(bis_pat, case=False, regex=True)

print(f"Tagalog markers : {has_tag.sum()} ({has_tag.mean():.1%})")
print(f"Bisaya markers  : {has_bis.sum()} ({has_bis.mean():.1%})")
print(f"Both            : {(has_tag & has_bis).sum()}")
print(f"Pure English    : {(~has_tag & ~has_bis).sum()} ({(~has_tag & ~has_bis).mean():.1%})")

print("\nBisaya share by city (top 12 cities by volume):")
m["_bis"] = has_bis.values
top_cities = m.City.value_counts().head(12).index
print(m[m.City.isin(top_cities)].groupby("City")
      .agg(n=("text", "size"), bisaya_pct=("_bis", "mean"))
      .sort_values("n", ascending=False).to_string())
print("\n>> If Bisaya is concentrated in a few cities, filtering to NCR keeps")
print(">> your 'English + Taglish' scope claim honest without losing much data.")
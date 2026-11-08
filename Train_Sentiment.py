"""
Objectives 1, 2 and 5: preprocessing, sentiment classification, evaluation.
==========================================================================
Input : output/jollibee_train.csv, output/jollibee_test.csv
Output: output/model.joblib, output/predictions_test.csv, output/cv_results.csv

Requires: pip install scikit-learn joblib

Three preprocessing decisions here are non-obvious and you will be asked about
them. Each is explained at the point it happens. The short version:

  1. Negation words are NEVER removed as stopwords. "hindi masarap" means the
     opposite of "masarap"; dropping "hindi" inverts the label.
  2. No stemming. Porter stemming assumes English morphology and mangles
     Tagalog affixes (na-, ma-, pag-, napaka-) and reduplication.
  3. Character n-grams alongside word n-grams. This data is full of typos
     ("kolang", "poang", "apealing") and Tagalog affixation; char n-grams
     absorb both, word n-grams alone do not.
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, make_scorer)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

OUTDIR = "output"
SEED = 42
CLASSES = ["Negative", "Neutral", "Positive"]

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)


def rule(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# =========================================================== STOPWORDS
# Built from English + Tagalog + Bisaya function words, then a KEEP list is
# SUBTRACTED. Standard Tagalog stopword lists (stopwords-iso and friends)
# include "hindi", "walang", "kulang", "masyado" -- all of which carry
# sentiment in a review. Using such a list unmodified is a real and common
# mistake; it silently destroys the signal you are trying to classify.

ENGLISH = """
a about above after again all am an and any are as at be because been before
being below between both but by can did do does doing down during each few for
from further had has have having he her here hers herself him himself his how i
if in into is it its itself just me more most my myself of off on once only or
other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will
with you your yours yourself yourselves
""".split()

TAGALOG = """
akin aking ako alin amin aming ang ano anumang apat at atin ating ay bababa bago
bakit bawat bilang dahil dalawa dapat din dito doon gagawin gayunman ginagawa
ginawa ginawang gumawa habang hanggang ikaw ilagay ilalim ilan inyong isa isang
itaas ito iyo iyon iyong ka kahit kailangan kailanman kami kanila kanilang
kanino kanya kanyang kapag kapwa karamihan katulad kaya kaysa ko kong kumuha
kung laban lahat lamang likod lima maaari maaaring maging makita marami may
mayroon mga minsan mismo mula muli na naging nagkaroon nais nakita namin narito
nasaan ng ngayon ni nila nilang nito niya niyang noon o pa paano pababa paggawa
pagitan pagkakaroon pagkatapos palabas pamamagitan panahon pangalawa para paraan
pareho pataas pero pumunta sa saan sabi sabihin sarili sila sino siya tatlo tayo
tulad tungkol una yan yung nyo niyo po opo ba naman nga lang daw raw kasi eh
""".split()

BISAYA = """
ako akong amo among ang anha anhi ani aron ba bisan dayon diay diha didto dinhi
ug uban unya usa usab siya sila silang sa nga ni nga kini kana kanang kang kay
ko ka kita kamo nimo ninyo niya nila ang mga pud pod man lagi gud kaayo
""".split()

# Never remove these. Negations flip polarity; intensifiers and complaint nouns
# ARE the signal for the keyword extraction in Objective 3.
KEEP = set("""
no not never none cannot cant dont didnt wasnt isnt wont doesnt couldnt
shouldnt wouldnt nothing without hardly barely
hindi di wala walang huwag wag ayaw ayoko
dili way wa walay
kulang sobra sobrang masyado napaka grabe very too really so much
bad good worse worst
""".split())

STOPWORDS = sorted((set(ENGLISH) | set(TAGALOG) | set(BISAYA)) - KEEP)
print(f"Stopwords: {len(STOPWORDS)}  (protected from removal: {len(KEEP)})")


# ====================================================== PREPROCESSING
URL = re.compile(r"https?://\S+|www\.\S+")
ELONG = re.compile(r"(.)\1{2,}")            # "sooooobrang" -> "soobrang"
MULTISPACE = re.compile(r"\s+")
# Keep emoji: sklearn's default token pattern (\w\w+) drops them, and they
# carry sentiment here (😟 😭 🥴 appear throughout the negative reviews).
EMOJI = "\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F"
TOKEN_PATTERN = rf"(?u)\b\w\w+\b|[{EMOJI}]"


def normalise(text):
    """Light normalisation only. Deliberately NOT stemming -- see module docstring."""
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = URL.sub(" ", t)
    t = ELONG.sub(r"\1\1", t)               # cap character runs at 2
    t = MULTISPACE.sub(" ", t)
    return t.strip()


def tokens_after_stopwords(text):
    """Only for display/documentation of Objective 1 -- the model uses the
    vectorizer's own stop_words parameter so the pipeline stays leak-free."""
    toks = re.findall(TOKEN_PATTERN, normalise(text))
    sw = set(STOPWORDS)
    return " ".join(t for t in toks if t not in sw)


# ============================================================== LOAD
rule("LOAD")
train = pd.read_csv(os.path.join(OUTDIR, "jollibee_train.csv"))
test = pd.read_csv(os.path.join(OUTDIR, "jollibee_test.csv"))
print(f"train {train.shape}   test {test.shape}")

for df in (train, test):
    df["text_norm"] = df["text_clean"].map(normalise)
    df["text_tokens"] = df["text_clean"].map(tokens_after_stopwords)

X_train, y_train = train["text_norm"], train["sentiment"]
X_test, y_test = test["text_norm"], test["sentiment"]

rule("PREPROCESSING EXAMPLES  (put 2-3 of these on your Objective 1 slide)")
for _, r in train.sample(6, random_state=SEED).iterrows():
    print(f"\n[{r.sentiment}] raw   : {r.text_clean[:90]}")
    print(f"           norm  : {r.text_norm[:90]}")
    print(f"           tokens: {r.text_tokens[:90]}")

# Sanity check that the negation guard actually works.
rule("NEGATION GUARD CHECK")
for probe in ["hindi masarap", "walang sauce", "dili lami", "not worth it"]:
    print(f"  {probe!r:22} -> {tokens_after_stopwords(probe)!r}")
print("\n>> If any of these came back empty, a negation leaked into the stoplist.")


# =========================================================== FEATURES
# Word n-grams capture phrases ("wrong order", "ang tagal"). Char n-grams
# capture misspellings and Tagalog affixation that word n-grams miss entirely.
features = FeatureUnion([
    ("word", TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), token_pattern=TOKEN_PATTERN,
        stop_words=STOPWORDS, min_df=2, sublinear_tf=True)),
    ("char", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
])

models = {
    "MultinomialNB": MultinomialNB(alpha=0.5),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", C=4.0, random_state=SEED),
    "LinearSVC": LinearSVC(class_weight="balanced", C=0.5, random_state=SEED),
}


# ================================================= CROSS-VALIDATION
# 5-fold CV rather than a single split: Neutral is ~10% of the data, so on one
# holdout its per-class scores swing wildly with the random seed. Report the
# mean and standard deviation.
rule("5-FOLD CROSS-VALIDATION ON TRAIN")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {
    "macro_f1": make_scorer(f1_score, average="macro"),
    "neutral_f1": make_scorer(f1_score, labels=["Neutral"], average="macro", zero_division=0),
}

rows = []
for name, clf in models.items():
    pipe = Pipeline([("features", features), ("clf", clf)])
    res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    rows.append({
        "model": name,
        "macro_f1_mean": res["test_macro_f1"].mean(),
        "macro_f1_std": res["test_macro_f1"].std(),
        "neutral_f1_mean": res["test_neutral_f1"].mean(),
        "neutral_f1_std": res["test_neutral_f1"].std(),
    })
    print(f"{name:22} macro-F1 {rows[-1]['macro_f1_mean']:.3f} "
          f"(+/- {rows[-1]['macro_f1_std']:.3f})   "
          f"Neutral-F1 {rows[-1]['neutral_f1_mean']:.3f} "
          f"(+/- {rows[-1]['neutral_f1_std']:.3f})")

cv_df = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
cv_df.to_csv(os.path.join(OUTDIR, "cv_results.csv"), index=False)
best_name = cv_df.iloc[0]["model"]
print(f"\n>> Best by macro-F1: {best_name}")


# ================================================== FINAL EVALUATION
rule("HELD-OUT TEST EVALUATION")

best = Pipeline([("features", features), ("clf", models[best_name])])
best.fit(X_train, y_train)
pred = best.predict(X_test)

baseline = y_test.value_counts(normalize=True).max()
acc = (pred == y_test).mean()
macro = f1_score(y_test, pred, average="macro")

print(f"Majority-class baseline : {baseline:.1%}")
print(f"Accuracy                : {acc:.1%}   (lift +{acc - baseline:.1%})")
print(f"Macro-F1                : {macro:.3f}\n")
print(classification_report(y_test, pred, labels=CLASSES, digits=3, zero_division=0))

cm = pd.DataFrame(
    confusion_matrix(y_test, pred, labels=CLASSES),
    index=[f"true_{c}" for c in CLASSES],
    columns=[f"pred_{c}" for c in CLASSES],
)
print("Confusion matrix:")
print(cm.to_string())
print("\n>> Read the Neutral row. Expect it to spill into both neighbours --")
print(">> that is the U-shaped rating distribution, not a bug in your model.")


# ------------------------------------------- short vs long (the promised check)
rule("SHORT vs LONG REVIEWS")

res = test.copy()
res["pred"] = pred
res["correct"] = res["pred"] == res["sentiment"]
by_len = res.groupby(res["is_short"].map({True: "<3 words", False: ">=3 words"})).agg(
    n=("correct", "size"), accuracy=("correct", "mean"))
print(by_len.to_string())
print("\n>> This is why short reviews stayed in. If accuracy holds up, say so on")
print(">> slide 10 -- it justifies the decision instead of hiding it.")


# ------------------------------------------------------- interpretability
rule("TOP FEATURES PER CLASS")

clf = best.named_steps["clf"]
if hasattr(clf, "coef_"):
    names = np.array(best.named_steps["features"].get_feature_names_out())
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        top = np.argsort(coefs)[-15:][::-1]
        terms = [t for t in names[top] if not t.startswith(("char__",))][:12]
        print(f"\n{cls}:")
        print("  " + ", ".join(names[top][:12]))
    print("\n>> Word-level terms here are quotable evidence in your write-up.")
    print(">> char__ features are n-gram fragments -- expected, not meaningful alone.")


# ================================================================ SAVE
rule("SAVE")
joblib.dump(best, os.path.join(OUTDIR, "model.joblib"))
res[["uuid", "Branch", "City", "text_clean", "sentiment", "pred", "correct",
     "is_short", "word_count"]].to_csv(
    os.path.join(OUTDIR, "predictions_test.csv"), index=False, encoding="utf-8-sig")
print(f"  {OUTDIR}/model.joblib")
print(f"  {OUTDIR}/predictions_test.csv")
print(f"  {OUTDIR}/cv_results.csv")

rule("FOR SLIDE 10")
print(f"""
Model      : {best_name} on TF-IDF (word 1-2 grams + char_wb 3-5 grams)
Validation : stratified 5-fold CV, macro-F1 {cv_df.iloc[0]['macro_f1_mean']:.3f} +/- {cv_df.iloc[0]['macro_f1_std']:.3f}
Holdout    : accuracy {acc:.1%} vs {baseline:.1%} majority baseline, macro-F1 {macro:.3f}
Weak class : Neutral -- report its recall explicitly, do not hide it in the average

Say "macro-F1" not "accuracy" as your headline number, and show the confusion
matrix. An examiner who asks "what about class imbalance?" is then already answered.
""")
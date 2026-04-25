# =============================================================================
# Flight Delay Prediction - ML Pipeline
# CS 4365 IEC | Group 10
# Schema-accurate version based on flight_db_create.sql
# Target: arr_del15 (already a boolean in the flight table)
# =============================================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1: LOAD DATA FROM SQLITE
# =============================================================================
# Adjust this path to where your flights.db actually lives
# Set working directory to repo root so all relative paths resolve correctly
import pathlib as _pl, os as _os
_os.chdir(_pl.Path(__file__).resolve().parent.parent.parent)

DB_PATH = "data/CleanedData/flights.db"

conn = sqlite3.connect(DB_PATH)

query = """
    SELECT
        f.flight_date,
        f.origin,
        f.dest,
        f.distance,
        f.crs_dep_time,
        f.carrier,
        f.dep_del15,
        f.arr_del15,
        a.state AS origin_state
    FROM flight f
    LEFT JOIN airport a ON f.origin = a.iata_code
    WHERE f.arr_del15 IS NOT NULL
"""

# For fast testing, add: LIMIT 500000
# Remove the limit for your full 6.4M row run

df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df):,} rows")

# Downcast float64 columns to float32 — halves RAM usage with no accuracy loss
float_cols = df.select_dtypes(include="float64").columns
df[float_cols] = df[float_cols].astype("float32")
print(f"Memory usage after downcast: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Sample to 1M rows — statistically sufficient for both models
df = df.sample(n=1_000_000, random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df):,} rows")

print(df.dtypes)
print(f"\nClass balance (arr_del15):\n{df['arr_del15'].value_counts(normalize=True).round(3)}")


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================
# --- Date features ---
df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
df["month"]       = df["flight_date"].dt.month          # 1-12
df["day_of_week"] = df["flight_date"].dt.dayofweek      # 0=Mon, 6=Sun

# --- Departure hour ---
# crs_dep_time is stored as TIME in SQLite (e.g. "14:30:00")
# Extract just the hour as an integer
df["dep_hour"] = pd.to_datetime(df["crs_dep_time"], format="%H:%M:%S", errors="coerce").dt.hour

# If the above returns all NaT, SQLite may store time differently — try this instead:
# df["dep_hour"] = df["crs_dep_time"].astype(str).str[:2].astype(int)

print("\nSample of engineered features:")
print(df[["flight_date", "month", "day_of_week", "dep_hour", "origin", "distance", "arr_del15"]].head())


# =============================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =============================================================================
le_origin  = LabelEncoder()
le_dest    = LabelEncoder()
le_state   = LabelEncoder()
le_carrier = LabelEncoder()

df["origin_enc"]  = le_origin.fit_transform(df["origin"].astype(str))
df["dest_enc"]    = le_dest.fit_transform(df["dest"].astype(str))
df["state_enc"]   = le_state.fit_transform(df["origin_state"].fillna("UNKNOWN").astype(str))
df["carrier_enc"] = le_carrier.fit_transform(df["carrier"].astype(str))


# =============================================================================
# STEP 4: DEFINE FEATURES & SPLIT
# =============================================================================
# These map directly to your schema columns — no guessing
FEATURES = [
    "month",          # seasonal pattern (your EDA confirmed summer peaks)
    "day_of_week",    # weekly patterns
    "dep_hour",       # ripple effect (your Chart 4 showed this clearly)
    "distance",       # flight distance is in the flight table directly
    "origin_enc",     # encoded origin airport
    "carrier_enc",    # encoded airline carrier (direct from flight table)
    "state_enc",      # encoded origin state (from airport join)
]
TARGET = "arr_del15"

model_df = df[FEATURES + [TARGET]].dropna()

X = model_df[FEATURES]
y = model_df[TARGET].astype(int)

# Stratified split preserves the on-time/delayed ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nFeatures used: {FEATURES}")
print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"Delayed in test set: {y_test.sum():,} ({y_test.mean():.1%})")


# =============================================================================
# STEP 5A: LOGISTIC REGRESSION (Baseline)
# =============================================================================
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # critical: dataset is imbalanced (~78% on time)
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]

lr_metrics = {
    "Accuracy":  accuracy_score(y_test, lr_preds),
    "Precision": precision_score(y_test, lr_preds),
    "Recall":    recall_score(y_test, lr_preds),
    "F1 Score":  f1_score(y_test, lr_preds),
    "AUC-ROC":   roc_auc_score(y_test, lr_proba),
}

print(pd.Series(lr_metrics).round(4).to_string())
print("\n", classification_report(y_test, lr_preds, target_names=["On Time (0)", "Delayed (1)"]))

# Which features push toward delay?
coef_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": lr.coef_[0]
}).sort_values("Coefficient", ascending=False)
print("\nCoefficients (positive = pushes toward predicting delay):")
print(coef_df.to_string(index=False))


# =============================================================================
# STEP 5B: RANDOM FOREST
# =============================================================================
print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=2
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_metrics = {
    "Accuracy":  accuracy_score(y_test, rf_preds),
    "Precision": precision_score(y_test, rf_preds),
    "Recall":    recall_score(y_test, rf_preds),
    "F1 Score":  f1_score(y_test, rf_preds),
    "AUC-ROC":   roc_auc_score(y_test, rf_proba),
}

print(pd.Series(rf_metrics).round(4).to_string())
print("\n", classification_report(y_test, rf_preds, target_names=["On Time (0)", "Delayed (1)"]))


# =============================================================================
# STEP 6: VISUALIZATIONS (saved as PNG for your report)
# =============================================================================

# --- 6A: Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title in zip(axes,
                             [lr_preds, rf_preds],
                             ["Logistic Regression", "Random Forest"]):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On Time", "Delayed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6B: ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(
    y_test, lr_proba,
    name=f"Logistic Regression (AUC={lr_metrics['AUC-ROC']:.3f})", ax=ax)
RocCurveDisplay.from_predictions(
    y_test, rf_proba,
    name=f"Random Forest (AUC={rf_metrics['AUC-ROC']:.3f})", ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
ax.set_title("ROC Curves — Flight Delay Classification", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6C: Feature Importance (Random Forest only) ---
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6D: Metrics Comparison Bar Chart ---
comparison = pd.DataFrame({
    "Logistic Regression": lr_metrics,
    "Random Forest": rf_metrics
}).T

fig, ax = plt.subplots(figsize=(9, 4))
comparison.plot(kind="bar", ax=ax, ylim=(0, 1), colormap="Set2", edgecolor="black")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.set_xticklabels(comparison.index, rotation=0)
ax.legend(loc="lower right")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAll charts saved: confusion_matrices.png, roc_curves.png, feature_importance.png, model_comparison.png")
print("\n=== Final Model Comparison ===")
print(comparison.round(4).to_string())


# =============================================================================
# NOTES
# =============================================================================
# 1. DB_PATH: update to your actual path. If your notebook is in the repo root,
#    it may just be "flights.db"
#
# 2. dep_hour parsing: SQLite stores TIME as text "HH:MM:SS". If pd.to_datetime
#    fails, check what df["crs_dep_time"].head() actually returns and adjust.
#
# 3. The delay cause columns (carrier_delay, weather_delay, nas_delay, etc.)
#    are available in your schema but intentionally excluded from features.
#    They are recorded AFTER the flight lands, so using them would be data
#    leakage — the model would be "predicting" delays it already knows about.
#
# 4. To add dest_enc as a feature, just append "dest_enc" to FEATURES above.
#    It may improve Random Forest accuracy slightly.
#
# 5. Full run on 6.4M rows: Random Forest with 200 trees may take 10-20 min.
#    Start with LIMIT 500000 to verify everything works first.
# =============================================================================
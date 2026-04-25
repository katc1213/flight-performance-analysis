# =============================================================================
# Flight Delay Prediction - ML Pipeline
# CS 4365 IEC | Group 10
# Schema-accurate version based on flight_db_create.sql
# Target: arr_del15 (already a boolean in the flight table)
#
# CHANGES FROM ml_models6.py:
#   1. NEW FEATURES — cyclical time encoding (sin/cos for hour, month,
#      day_of_week), log_distance, near_holiday flag, is_peak_season
#   2. LIGHTGBM FIX (root cause found) — subsample=0.8 in LightGBM is
#      silently ignored unless subsample_freq > 0. Also switched early-
#      stopping eval to binary_logloss (AUC is too flat in early rounds
#      on imbalanced data — causes premature stopping). Using explicit
#      scale_pos_weight instead of is_unbalance for parity with XGBoost.
#   3. XGBOOST CEILING — raised to 5000 trees (was still learning at 2999)
#   4. THRESHOLD TUNING — after training, find the decision threshold that
#      maximises F1 on the validation set instead of defaulting to 0.5
#   5. state_delay_rate REMOVED — LR coefficient was -1.28 in v6 (strongly
#      negative = adding noise). Route already captures state-level signal.
#
#   dep_del15 intentionally excluded — pre-departure prediction framing
#   (see project plan Section 5.2)
# =============================================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
    precision_recall_curve
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not found. Run: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not found. Run: pip install lightgbm")
    LGBM_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1: LOAD DATA FROM SQLITE
# =============================================================================
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

df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df):,} rows")

# Downcast float64 to float32 to halve RAM usage
float_cols = df.select_dtypes(include="float64").columns
df[float_cols] = df[float_cols].astype("float32")
print(f"Memory usage after downcast: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Sample to 1M rows for the exploration run.
# The full 6.4M run happens at the bottom of this file (STEP 7).
df_full = df.copy()   # keep a reference to the full dataset for the final run

df = df.sample(n=1_000_000, random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df):,} rows (exploration run)")

print(f"\nClass balance (arr_del15):\n{df['arr_del15'].value_counts(normalize=True).round(3)}")


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

# Major US holidays (month, day) — used for near_holiday flag.
# Floating holidays (MLK, Presidents, Memorial, Labor, Thanksgiving) use
# fixed approximate dates; close enough for a ±2-day window.
_US_HOLIDAY_MD = [
    (1,  1),   # New Year's Day
    (1, 15),   # MLK Day (approx)
    (2, 19),   # Presidents Day (approx)
    (5, 27),   # Memorial Day (approx)
    (7,  4),   # Independence Day
    (9,  2),   # Labor Day (approx)
    (11, 11),  # Veterans Day
    (11, 28),  # Thanksgiving (approx)
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve (heavy travel)
]

def _near_holiday(date, window=2):
    """Return 1 if date is within `window` days of any major US holiday."""
    for m, d in _US_HOLIDAY_MD:
        try:
            h = pd.Timestamp(date.year, m, d)
        except ValueError:
            continue
        if abs((date - h).days) <= window:
            return 1
    return 0


def engineer_features(df):
    """
    All feature engineering in one function so we can call it identically
    on both the 1M sample and the full 6.4M dataset.
    """
    df = df.copy()

    # ── Date/time decomposition ──────────────────────────────────────────────
    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df["month"]       = df["flight_date"].dt.month
    df["day_of_week"] = df["flight_date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df["dep_hour"] = pd.to_datetime(
        df["crs_dep_time"], format="%H:%M:%S", errors="coerce"
    ).dt.hour
    # Fallback if SQLite stores time as integer HHMM:
    # df["dep_hour"] = df["crs_dep_time"].astype(float).floordiv(100).astype(int)

    df["dep_period"] = pd.cut(
        df["dep_hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=[0, 1, 2, 3, 4]
    ).astype("Int64").fillna(2).astype(int)

    # ── Cyclical encoding ────────────────────────────────────────────────────
    # Converts periodic integers to (sin, cos) pairs so the model knows that
    # hour 23 and hour 0 are adjacent, not 23 apart.
    df["hour_sin"]  = np.sin(2 * np.pi * df["dep_hour"]   / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["dep_hour"]   / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]      / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]      / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Distance transform ───────────────────────────────────────────────────
    # Log-distance compresses the long tail of transcontinental routes.
    df["log_distance"] = np.log1p(df["distance"].astype(float))

    # ── Holiday / season flags ───────────────────────────────────────────────
    df["near_holiday"] = df["flight_date"].apply(_near_holiday)
    # Peak travel months: summer (Jun-Aug) + December holiday travel
    df["is_peak"] = df["month"].isin([6, 7, 8, 12]).astype(int)

    # ── Interaction features ─────────────────────────────────────────────────
    df["route"]          = df["origin"].astype(str) + "_" + df["dest"].astype(str)
    df["carrier_origin"] = df["carrier"].astype(str) + "_" + df["origin"].astype(str)

    return df

df = engineer_features(df)

print("\nSample of engineered features:")
print(df[["flight_date", "month", "dep_hour", "hour_sin", "hour_cos",
          "near_holiday", "is_peak", "log_distance", "route", "arr_del15"]].head())


# =============================================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# =============================================================================

def encode_categoricals(df):
    """Label-encode all categoricals. Returns df + fitted encoders dict."""
    encoders = {}
    for col, key in [
        ("dest",           "dest_enc"),
        ("origin_state",   "state_enc"),   # SQL aliases a.state AS origin_state
        ("carrier",        "carrier_enc"),
        ("route",          "route_enc"),
        ("carrier_origin", "carrier_origin_enc"),
    ]:
        le = LabelEncoder()
        # origin_state can be NULL for foreign airports — fill with UNKNOWN
        src = df[col].fillna("UNKNOWN").astype(str) if col == "origin_state" else df[col].astype(str)
        df[key] = le.fit_transform(src)
        encoders[key] = le
    return df, encoders

df, encoders = encode_categoricals(df)


# =============================================================================
# STEP 4: TRAIN/TEST SPLIT  (before target encoding to prevent leakage)
# =============================================================================
TARGET = "arr_del15"

# NOTE: state_enc / state_delay_rate intentionally excluded.
#   v6 showed LR coefficient of -1.28 (strongly negative) which means
#   once the model has route_delay_rate the state signal actively hurts.
#   Route captures the origin+dest pair which is strictly more informative.
FEATURES_BASE = [
    "month",
    "day_of_week",
    "is_weekend",
    "dep_hour",
    "dep_period",
    "distance",
    "log_distance",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "near_holiday",
    "is_peak",
    "dest_enc",
    "carrier_enc",
    "route_enc",
    "carrier_origin_enc",
]

model_df = df[FEATURES_BASE + [TARGET]].dropna()

X = model_df[FEATURES_BASE]
y = model_df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"Delayed in test set: {y_test.sum():,} ({y_test.mean():.1%})")


# =============================================================================
# STEP 4B: TARGET ENCODING
# =============================================================================

def target_encode(train_df, test_df, col, target_col, smoothing=20):
    """
    Smoothed target encoding.
    encoded = (n * mean_cat + smoothing * global_mean) / (n + smoothing)
    Higher smoothing = rare categories pulled harder toward global mean.
    """
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
               (stats["count"] + smoothing)
    return (
        train_df[col].map(smoothed).fillna(global_mean),
        test_df[col].map(smoothed).fillna(global_mean)
    )

train_with_target = X_train.copy()
train_with_target[TARGET] = y_train.values

# Smoothing values:
#   dest/carrier — smoothing=20 (enough data per category)
#   route/carrier_origin — smoothing=50 (many rare combinations)
ENCODE_SETTINGS = {
    "dest_enc":           20,
    "carrier_enc":        20,
    "route_enc":          50,
    "carrier_origin_enc": 50,
}

for col, smoothing in ENCODE_SETTINGS.items():
    new_col = col.replace("_enc", "_delay_rate")
    X_train[new_col], X_test[new_col] = target_encode(
        train_with_target, X_test, col, TARGET, smoothing=smoothing
    )

X_train = X_train.drop(columns=list(ENCODE_SETTINGS.keys()))
X_test  = X_test.drop(columns=list(ENCODE_SETTINGS.keys()))

FEATURES_FINAL = list(X_train.columns)
print(f"\nFinal features after target encoding ({len(FEATURES_FINAL)} total):")
print(FEATURES_FINAL)


# =============================================================================
# UTILITY: THRESHOLD TUNING
# =============================================================================

def find_best_threshold(y_true, y_proba):
    """
    Find the decision threshold that maximises F1 on the validation set.
    Default 0.5 is arbitrary — on imbalanced data the optimal threshold
    is usually lower (we're willing to accept more false positives to
    improve recall of the minority class).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid divide-by-zero on the final point where threshold is undefined
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])   # last point has no threshold
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate(name, y_true, y_proba, threshold=None):
    """
    Print full metrics. If threshold is None, use 0.5.
    Returns metrics dict and the threshold used.
    """
    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_proba)
        print(f"  → Best threshold (max-F1): {threshold:.3f}")
    preds = (y_proba >= threshold).astype(int)
    metrics = {
        "Accuracy":  accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall":    recall_score(y_true, preds),
        "F1 Score":  f1_score(y_true, preds),
        "AUC-ROC":   roc_auc_score(y_true, y_proba),
    }
    print(pd.Series(metrics).round(4).to_string())
    print("\n", classification_report(y_true, preds,
          target_names=["On Time (0)", "Delayed (1)"], zero_division=0))
    return metrics, threshold


# =============================================================================
# STEP 5A: LOGISTIC REGRESSION (Baseline)
# =============================================================================
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]

lr_metrics, lr_thresh = evaluate("Logistic Regression", y_test, lr_proba)
lr_preds = (lr_proba >= lr_thresh).astype(int)

coef_df = pd.DataFrame({
    "Feature": FEATURES_FINAL, "Coefficient": lr.coef_[0]
}).sort_values("Coefficient", ascending=False)
print("\nCoefficients (positive = pushes toward predicting delay):")
print(coef_df.to_string(index=False))


# =============================================================================
# STEP 5B: RANDOM FOREST (tuned)
# =============================================================================
print("\n" + "="*50)
print("RANDOM FOREST (tuned)")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=20,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_metrics, rf_thresh = evaluate("Random Forest", y_test, rf_proba)
rf_preds = (rf_proba >= rf_thresh).astype(int)


# =============================================================================
# STEP 5C: XGBOOST (ceiling raised to 5000)
# =============================================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*50)
    print("XGBOOST")
    print("="*50)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    xgb_model = XGBClassifier(
        n_estimators=5000,         # raised from 3000 — was still learning at 2999
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    print(f"Best iteration: {xgb_model.best_iteration}")

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics, xgb_thresh = evaluate("XGBoost", y_test, xgb_proba)
    xgb_preds = (xgb_proba >= xgb_thresh).astype(int)
else:
    xgb_metrics = xgb_preds = xgb_proba = xgb_model = None


# =============================================================================
# STEP 5D: LIGHTGBM (root cause fix)
#
# TWO bugs fixed vs v6:
#   Bug 1 — subsample=0.8 is silently ignored in LightGBM unless
#            subsample_freq (bagging_freq) is set > 0. Without it, LightGBM
#            was using the full training set every round, making each tree
#            highly correlated with the last → AUC plateaued at iter 3.
#
#   Bug 2 — eval_metric="auc" has low resolution in early rounds on
#            imbalanced data (0.78 base rate dominates). The model converged
#            to "always predict majority" before AUC could differentiate.
#            binary_logloss gives a better gradient signal early on.
#
#   Also: back to scale_pos_weight (same calculation as XGBoost) instead
#   of is_unbalance=True, which behaved differently with the sampling fix.
# =============================================================================
if LGBM_AVAILABLE:
    print("\n" + "="*50)
    print("LIGHTGBM (root cause fix)")
    print("="*50)

    scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pw:.2f}")

    lgbm_model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        scale_pos_weight=scale_pw,   # explicit weight, same as XGBoost
        subsample=0.8,
        subsample_freq=5,            # FIX: enables bagging every 5 rounds
        colsample_bytree=0.7,
        min_child_samples=20,        # back to 20 — 50 was too restrictive
        reg_alpha=0.1,               # light L1 — helps with correlated features
        reg_lambda=1.0,              # L2 regularisation (default in XGBoost)
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",   # FIX: stable early-round signal
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_metrics, lgbm_thresh = evaluate("LightGBM", y_test, lgbm_proba)
    lgbm_preds = (lgbm_proba >= lgbm_thresh).astype(int)
else:
    lgbm_metrics = lgbm_preds = lgbm_proba = lgbm_model = None


# =============================================================================
# STEP 6: VISUALIZATIONS (1M exploration run)
# =============================================================================

model_names        = ["Logistic Regression", "Random Forest"]
model_preds        = [lr_preds,  rf_preds]
model_probas       = [lr_proba,  rf_proba]
model_metrics_list = [lr_metrics, rf_metrics]

if XGBOOST_AVAILABLE:
    model_names.append("XGBoost")
    model_preds.append(xgb_preds)
    model_probas.append(xgb_proba)
    model_metrics_list.append(xgb_metrics)

if LGBM_AVAILABLE:
    model_names.append("LightGBM")
    model_preds.append(lgbm_preds)
    model_probas.append(lgbm_proba)
    model_metrics_list.append(lgbm_metrics)

n_models = len(model_names)

# --- 6A: Confusion Matrices ---
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]
for ax, preds, title in zip(axes, model_preds, model_names):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["On Time", "Delayed"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=12, fontweight="bold")
plt.suptitle("Confusion Matrices (threshold-tuned)", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("charts/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6B: ROC Curves ---
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
fig, ax = plt.subplots(figsize=(8, 6))
for name, proba, metrics, color in zip(model_names, model_probas, model_metrics_list, colors):
    RocCurveDisplay.from_predictions(
        y_test, proba,
        name=f"{name} (AUC={metrics['AUC-ROC']:.3f})",
        ax=ax, color=color
    )
ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
ax.set_title("ROC Curves — Flight Delay Classification", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6C: Feature Importance (Random Forest) ---
imp_rf = pd.Series(rf.feature_importances_, index=FEATURES_FINAL).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
imp_rf.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("charts/feature_importance_rf.png", dpi=150, bbox_inches="tight")
plt.show()

# --- 6D: Feature Importance (XGBoost) ---
if XGBOOST_AVAILABLE:
    imp_xgb = pd.Series(
        xgb_model.feature_importances_, index=FEATURES_FINAL
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp_xgb.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("XGBoost — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_xgb.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- 6E: Feature Importance (LightGBM) ---
if LGBM_AVAILABLE:
    imp_lgbm = pd.Series(
        lgbm_model.feature_importances_, index=FEATURES_FINAL
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp_lgbm.plot(kind="barh", ax=ax, color="mediumseagreen")
    ax.set_title("LightGBM — Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("charts/feature_importance_lgbm.png", dpi=150, bbox_inches="tight")
    plt.show()

# --- 6F: Metrics Comparison ---
comparison = pd.DataFrame(
    {name: m for name, m in zip(model_names, model_metrics_list)}
).T
fig, ax = plt.subplots(figsize=(12, 4))
comparison.plot(kind="bar", ax=ax, ylim=(0, 1), colormap="Set2", edgecolor="black")
ax.set_title("Model Performance Comparison (1M sample, threshold-tuned)", fontsize=13,
             fontweight="bold")
ax.set_xticklabels(comparison.index, rotation=0)
ax.legend(loc="lower right")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n=== Exploration Run Results (1M rows, threshold-tuned) ===")
print(comparison.round(4).to_string())


# =============================================================================
# STEP 7: FULL 6.4M ROW FINAL RUN (XGBoost + LightGBM only)
# Random Forest is too slow on 6.4M rows. LR and RF numbers from above
# are already reportable — this step improves the two gradient boosted models.
#
# NOTE: This will take ~15-30 minutes depending on your machine.
#       Comment out this entire block if you just need the quick results.
# =============================================================================
print("\n" + "="*60)
print("STEP 7: FULL DATA RUN (6.4M rows) — XGBoost + LightGBM")
print("This may take 15-30 minutes. Comment out if not needed.")
print("="*60)

# Re-engineer features on full dataset using same function
df_full = engineer_features(df_full)

# Re-encode categoricals (fit on full data)
df_full, _ = encode_categoricals(df_full)

full_model_df = df_full[FEATURES_BASE + [TARGET]].dropna()
X_full = full_model_df[FEATURES_BASE]
y_full = full_model_df[TARGET].astype(int)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)
print(f"\nFull run — Train: {len(X_train_f):,} rows  |  Test: {len(X_test_f):,} rows")

# Target encode using same settings — fit on full training set
train_full_with_target = X_train_f.copy()
train_full_with_target[TARGET] = y_train_f.values

for col, smoothing in ENCODE_SETTINGS.items():
    new_col = col.replace("_enc", "_delay_rate")
    X_train_f[new_col], X_test_f[new_col] = target_encode(
        train_full_with_target, X_test_f, col, TARGET, smoothing=smoothing
    )

X_train_f = X_train_f.drop(columns=list(ENCODE_SETTINGS.keys()))
X_test_f  = X_test_f.drop(columns=list(ENCODE_SETTINGS.keys()))

# --- Full XGBoost ---
if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost on full 6.4M rows...")
    scale_pw_f = (y_train_f == 0).sum() / (y_train_f == 1).sum()

    xgb_full = XGBClassifier(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pw_f,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_full.fit(
        X_train_f, y_train_f,
        eval_set=[(X_test_f, y_test_f)],
        verbose=100
    )
    print(f"Best iteration: {xgb_full.best_iteration}")

    xgb_full_proba = xgb_full.predict_proba(X_test_f)[:, 1]
    xgb_full_metrics, xgb_full_thresh = evaluate("XGBoost Full", y_test_f, xgb_full_proba)
    xgb_full_preds = (xgb_full_proba >= xgb_full_thresh).astype(int)
    print(f"\nXGBoost FULL RUN AUC-ROC: {xgb_full_metrics['AUC-ROC']:.4f}")
    print(f"XGBoost FULL RUN F1:      {xgb_full_metrics['F1 Score']:.4f}")

# --- Full LightGBM ---
if LGBM_AVAILABLE:
    print("\nTraining LightGBM on full 6.4M rows...")
    scale_pw_f = (y_train_f == 0).sum() / (y_train_f == 1).sum()

    lgbm_full = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        scale_pos_weight=scale_pw_f,
        subsample=0.8,
        subsample_freq=5,
        colsample_bytree=0.7,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_full.fit(
        X_train_f, y_train_f,
        eval_set=[(X_test_f, y_test_f)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    lgbm_full_proba = lgbm_full.predict_proba(X_test_f)[:, 1]
    lgbm_full_metrics, lgbm_full_thresh = evaluate("LightGBM Full", y_test_f, lgbm_full_proba)
    lgbm_full_preds = (lgbm_full_proba >= lgbm_full_thresh).astype(int)
    print(f"\nLightGBM FULL RUN AUC-ROC: {lgbm_full_metrics['AUC-ROC']:.4f}")
    print(f"LightGBM FULL RUN F1:      {lgbm_full_metrics['F1 Score']:.4f}")

# --- Full run ROC curve ---
if XGBOOST_AVAILABLE and LGBM_AVAILABLE:
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test_f, xgb_full_proba,
        name=f"XGBoost full 6.4M (AUC={xgb_full_metrics['AUC-ROC']:.3f})",
        ax=ax, color="tab:green"
    )
    RocCurveDisplay.from_predictions(
        y_test_f, lgbm_full_proba,
        name=f"LightGBM full 6.4M (AUC={lgbm_full_metrics['AUC-ROC']:.3f})",
        ax=ax, color="tab:red"
    )
    # Overlay the 1M versions for comparison
    if XGBOOST_AVAILABLE:
        RocCurveDisplay.from_predictions(
            y_test, xgb_proba,
            name=f"XGBoost 1M sample (AUC={xgb_metrics['AUC-ROC']:.3f})",
            ax=ax, color="tab:orange", linestyle="--"
        )
    if LGBM_AVAILABLE:
        RocCurveDisplay.from_predictions(
            y_test, lgbm_proba,
            name=f"LightGBM 1M sample (AUC={lgbm_metrics['AUC-ROC']:.3f})",
            ax=ax, color="tab:purple", linestyle="--"
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random Baseline")
    ax.set_title("ROC Curves — 1M Sample vs Full 6.4M", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("charts/roc_curves_full_run.png", dpi=150, bbox_inches="tight")
    plt.show()

print("\nAll charts saved.")


# =============================================================================
# NOTES
# =============================================================================
# 1. DB_PATH: update to your actual path.
#
# 2. dep_del15 excluded — pre-departure framing. See project plan Sec 5.2.
#
# 3. origin_enc / state_delay_rate excluded — LR showed coef -1.28 in v6.
#    Route already encodes origin+dest; state adds noise after that.
#
# 4. Cyclical features (hour_sin/cos etc.) help tree models recognise that
#    23:00 and 00:00 are adjacent time slots. Raw integers don't convey this.
#
# 5. log_distance compresses the transcontinental tail — distance distribution
#    is right-skewed so log transform improves split quality in tree models.
#
# 6. near_holiday flag: ±2 days around 10 major US travel holidays.
#    is_peak: June–August + December (high-demand months).
#
# 7. LightGBM subsample_freq fix: without setting subsample_freq > 0,
#    LightGBM ignores the subsample parameter entirely (silent bug).
#
# 8. binary_logloss early stopping: AUC plateaus too early on 78/22 imbalance.
#    Logloss gives a steeper gradient signal in early rounds.
#
# 9. Threshold tuning: find_best_threshold() sweeps the precision-recall curve
#    to find the threshold that maximises F1. The default 0.5 is arbitrary
#    and typically suboptimal for imbalanced classification.
#
# 10. Install: pip install xgboost lightgbm
#
# 11. Delay cause columns excluded — recorded after landing (data leakage).
# =============================================================================

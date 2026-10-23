# ============================================================
# AGRICULTURE CROP RECOMMENDATION SYSTEM (FAST + COMPLETE)
# Dataset  : Crop.csv
# Models   : Baseline RF, Optional Tuned RF (Auto-skip if accuracy high),
#            Voting Hybrid, Stacking Hybrid
# EDA      : Multiple charts
# Output   : best_crop_model.pkl
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pickle


# ---------------------------
# 0) SETTINGS
# ---------------------------
DATA_PATH = "Crop.csv"
RANDOM_STATE = 42

# Auto-skip tuning if baseline accuracy >= this value
SKIP_TUNING_IF_ACCURACY_GE = 0.995

# RandomizedSearch settings (fast)
TUNING_N_ITER = 10      # 10 candidates
TUNING_CV = 3           # 3 folds => 30 fits only

np.random.seed(RANDOM_STATE)


# ---------------------------
# 1) LOAD DATASET
# ---------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}. Place Crop.csv in this folder.")

data = pd.read_csv(DATA_PATH)

print("\n✅ Dataset Loaded Successfully!")
print("Shape:", data.shape)
print("\nFirst 5 rows:\n", data.head())
print("\n✅ Columns:", list(data.columns))
print("\n✅ Target Labels (Crops):", data['label'].nunique())
print("\n✅ Missing Values:\n", data.isnull().sum())

data = data.dropna().reset_index(drop=True)

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


# ---------------------------
# 2) EDA (Charts)
# ---------------------------

# Crop distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=data['label'], order=data['label'].value_counts().index)
plt.title("Crop Distribution (Count per Crop)")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data[feature_cols].corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Feature histograms
data[feature_cols].hist(figsize=(14, 10), bins=25)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# Top-10 crops for readable plots
top_crops = data['label'].value_counts().head(10).index
filtered = data[data['label'].isin(top_crops)].copy()

# Boxplots for N, P, K
for col in ['N', 'P', 'K']:
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='label', y=col, data=filtered)
    plt.title(f"{col} Distribution (Top 10 Crops)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Violin plot rainfall
plt.figure(figsize=(14, 6))
sns.violinplot(x='label', y='rainfall', data=filtered)
plt.title("Rainfall Distribution (Top 10 Crops)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter relationships (no legend to keep clean)
plt.figure(figsize=(7, 5))
sns.scatterplot(x='temperature', y='humidity', data=data)
plt.title("Temperature vs Humidity")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(x='ph', y='rainfall', data=data)
plt.title("pH vs Rainfall")
plt.tight_layout()
plt.show()


# ---------------------------
# 3) TRAIN / TEST SPLIT
# ---------------------------
X = data[feature_cols]
y = data['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n✅ Train shape:", Xtrain.shape, " Test shape:", Xtest.shape)


# ---------------------------
# 4) BASELINE RANDOM FOREST
# ---------------------------
rf_base = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE
)

rf_base.fit(Xtrain, Ytrain)
pred_rf = rf_base.predict(Xtest)

acc_rf = accuracy_score(Ytest, pred_rf)
print("\n================ BASELINE RANDOM FOREST ================")
print("Accuracy:", acc_rf)
print("\nClassification Report:\n", classification_report(Ytest, pred_rf))

# Cross validation baseline
cv_rf = cross_val_score(rf_base, X, y, cv=5, scoring="accuracy")
print("Cross Validation Scores:", cv_rf)
print("Mean CV Score:", cv_rf.mean())

# Confusion matrix baseline
labels_sorted = sorted(y.unique())
cm_rf = confusion_matrix(Ytest, pred_rf, labels=labels_sorted)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_rf, cmap="Blues")
plt.title("Confusion Matrix - Baseline RF")
plt.tight_layout()
plt.show()


# ---------------------------
# 5) OPTIONAL TUNING (AUTO SKIP IF ALREADY HIGH)
# ---------------------------
best_rf = rf_base
acc_best_rf = acc_rf

if acc_rf >= SKIP_TUNING_IF_ACCURACY_GE:
    print(f"\n⚡ Baseline accuracy is already >= {SKIP_TUNING_IF_ACCURACY_GE}.")
    print("⚡ Skipping tuning to save time .")
else:
    print("\n🔍 Running FAST RandomizedSearchCV tuning...")

    param_dist = {
        "n_estimators": [150, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False]
    }

    rand = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=TUNING_N_ITER,        # ✅ fast
        cv=TUNING_CV,                # ✅ fast
        scoring="accuracy",
        n_jobs=1,                    # ✅ stable on Windows
        verbose=2,
        random_state=RANDOM_STATE
    )

    rand.fit(Xtrain, Ytrain)
    best_rf = rand.best_estimator_

    pred_best_rf = best_rf.predict(Xtest)
    acc_best_rf = accuracy_score(Ytest, pred_best_rf)

    print("\n================ TUNED RANDOM FOREST ================")
    print("Best Params:", rand.best_params_)
    print("Best CV Score:", rand.best_score_)
    print("Test Accuracy:", acc_best_rf)


# ---------------------------
# 6) HYBRID MODEL 1: VOTING
# ---------------------------
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE)
knn = KNeighborsClassifier(n_neighbors=5)

voting_model = VotingClassifier(
    estimators=[("rf", best_rf), ("gb", gb), ("knn", knn)],
    voting="hard"
)

voting_model.fit(Xtrain, Ytrain)
pred_vote = voting_model.predict(Xtest)
acc_vote = accuracy_score(Ytest, pred_vote)

print("\n================ HYBRID MODEL: VOTING ================")
print("Test Accuracy:", acc_vote)
print("\nClassification Report:\n", classification_report(Ytest, pred_vote))


# ---------------------------
# 7) HYBRID MODEL 2: STACKING
# ---------------------------
stack_model = StackingClassifier(
    estimators=[
        ("rf", best_rf),
        ("gb", gb),
        ("knn", knn)
    ],
    final_estimator=LogisticRegression(max_iter=500)
)

stack_model.fit(Xtrain, Ytrain)
pred_stack = stack_model.predict(Xtest)
acc_stack = accuracy_score(Ytest, pred_stack)

print("\n================ HYBRID MODEL: STACKING ================")
print("Test Accuracy:", acc_stack)
print("\nClassification Report:\n", classification_report(Ytest, pred_stack))


# ---------------------------
# 8) FEATURE IMPORTANCE (Explainability)
# ---------------------------
importances = best_rf.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_cols)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()


# ---------------------------
# 9) SELECT BEST MODEL & SAVE
# ---------------------------
models = {
    "Baseline Random Forest": (rf_base, acc_rf),
    "Tuned/Best Random Forest": (best_rf, acc_best_rf),
    "Voting Hybrid": (voting_model, acc_vote),
    "Stacking Hybrid": (stack_model, acc_stack)
}

best_name = max(models, key=lambda k: models[k][1])
best_model = models[best_name][0]
best_acc = models[best_name][1]

print("\n✅ BEST MODEL SELECTED:", best_name)
print("✅ BEST TEST ACCURACY:", best_acc)

with open("best_crop_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Saved: best_crop_model.pkl")


# ---------------------------
# 10) SAMPLE PREDICTION
# ---------------------------
sample = np.array([[73, 57, 44, 20.87974371, 82.00274423, 6.502985292, 202.9355362]])
predicted_crop = best_model.predict(sample)

print("\n✅ Sample Input:", sample.tolist()[0])
print("✅ Recommended Crop:", predicted_crop[0])

print("\n🎉 Completed! ")

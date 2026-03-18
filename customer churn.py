# ----------------------------
# Customer Churn Prediction
# Fully working version
# ----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\python course(asra)\ecommerce_customer_churn.csv")

# Strip column spaces just in case
df.columns = df.columns.str.strip()

print("Columns in dataset:", df.columns)

# ----------------------------
# CREATE CHURN TARGET
# ----------------------------
df["Churn"] = (df["days_since_last_purchase"] > 30).astype(int)

# ----------------------------
# DROP LEAKAGE FEATURES
# ----------------------------
df = df.drop([
    "Customer_ID",
    "satisfaction_score",
    "days_since_last_purchase"  # Dominating feature already used for target
], axis=1, errors='ignore')

# ----------------------------
# DEFINE NUMERIC & CATEGORICAL FEATURES
# ----------------------------
numeric_cols = [
    "account_age_months",
    "avg_order_value",
    "total_orders",
    "discount_usage_rate",
    "return_rate",
    "customer_support_tickets",
    "browsing_frequency_per_week",
    "cart_abandonment_rate",
    "product_review_score_avg",
    "engagement_score",
    "price_sensitivity_index"
]

categorical_cols = ["loyalty_member"]  # Only categorical column

# ----------------------------
# SPLIT DATA
# ----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ----------------------------
# PREPROCESSING PIPELINE
# ----------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

pipeline = ImbPipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.01, # slower learning for realism
        max_depth=2, # shallow trees
        subsample=0.8,# stochastic gradient boosting
        max_features='sqrt',
        random_state=42
    ))
])

# ----------------------------
# CROSS-VALIDATION
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv)
print("\nCross Validation Accuracy:", round(cv_score.mean(), 3))

# ----------------------------
# TRAIN MODEL
# ----------------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ----------------------------
# EVALUATION
# ----------------------------
print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# CONFUSION MATRIX
# ----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Churn","Churn"],
            yticklabels=["Non-Churn","Churn"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# CORRELATION HEATMAP
# ----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
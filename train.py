# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import joblib
import os

# -------------------------------
# 1. Create folders
# -------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------
# 2. Load & Prepare Data
# -------------------------------
print("Loading data...")
df = pd.read_csv("data.csv")

# Target: Response = Yes → Converted = 1
df["Converted"] = (df["Response"] == "Yes").astype(int)

# Drop ID, date, original target
drop_cols = ["Customer", "Effective To Date", "Response"]
X = df.drop(columns=drop_cols + ["Converted"])
y = df["Converted"]

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Conversion Rate: {y.mean():.2%}")

# -------------------------------
# 3. Encode Categorical Features
# -------------------------------
encoders = {}
X_encoded = X.copy()

categorical_cols = X.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Save for app
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(X_encoded.columns.tolist(), "feature_names.pkl")

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

# -------------------------------
# 5. Model 1: XGBoost
# -------------------------------
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="auc"
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgboost_model.pkl")

# -------------------------------
# 6. Model 2: Random Forest
# -------------------------------
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=8, min_samples_split=10,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")

# -------------------------------
# 7. Model 3: CNN (1D) → .keras
# -------------------------------
print("Training CNN...")
X_train_cnn = X_train.values.reshape(-1, X_train.shape[1], 1)

cnn_model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
cnn_model.fit(X_train_cnn, y_train, epochs=15, batch_size=64, verbose=1, validation_split=0.1)

cnn_model.save("cnn_model.keras")

# -------------------------------
# 8. Model 4: SMOTE + XGBoost
# -------------------------------
print("Training SMOTE + XGBoost...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

gan_xgb = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42
)
gan_xgb.fit(X_res, y_res)
joblib.dump(gan_xgb, "gan_xgb_model.pkl")

# -------------------------------
# 9. Model 5: Stacking Ensemble (FIXED: NO scikeras)
# -------------------------------
print("Training Stacking Ensemble (XGB + RF only)...")

# Use only tree models to avoid scikeras bug
base_models = [
    ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42))
]

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,
    n_jobs=1,
    passthrough=False
)

stack.fit(X_train, y_train)
joblib.dump(stack, "stack_model.pkl")

# -------------------------------
# Done!
# -------------------------------
print("\nAll 5 models trained and saved!")
print("Next: Run `python compare.py`")
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

# -------------------------------
# 1. Load & Prepare Data
# -------------------------------
print("Loading data...")
df = pd.read_csv("data.csv")

# Target
df["Converted"] = (df["Response"] == "Yes").astype(int)

# Drop
drop_cols = ["Customer", "Effective To Date", "Response"]
X = df.drop(columns=drop_cols + ["Converted"])
y = df["Converted"]

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Conversion Rate: {y.mean():.2%}")

# -------------------------------
# 2. Encode
# -------------------------------
encoders = {}
X_encoded = X.copy()
cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Save at root
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(X_encoded.columns.tolist(), "feature_names.pkl")

# -------------------------------
# 3. Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

X_test.to_csv("X_test.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

# -------------------------------
# 4. Models
# -------------------------------
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgboost_model.pkl")

print("Training RF...")
rf_model = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")

print("Training CNN...")
X_train_cnn = X_train.values.reshape(-1, X_train.shape[1], 1)
cnn_model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=64, verbose=0)
cnn_model.save("cnn_model.keras")

print("Training SMOTE + XGB...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
gan_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=6, random_state=42)
gan_xgb.fit(X_res, y_res)
joblib.dump(gan_xgb, "gan_xgb_model.pkl")

print("Training Stacking...")
base = [
    ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42))
]
stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=1000), cv=3)
stack.fit(X_train, y_train)
joblib.dump(stack, "stack_model.pkl")

print("All models saved at root!")
# compare.py
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import tensorflow as tf

print("Loading test data...")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()

print("Loading models...")
models = {
    "XGBoost": joblib.load("xgboost_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl"),
    "CNN": tf.keras.models.load_model("cnn_model.keras"),
    "SMOTE + XGBoost": joblib.load("gan_xgb_model.pkl"),
    "Stacking (XGB+RF)": joblib.load("stack_model.pkl")
}

results = []
for name, model in models.items():
    print(f"Predicting with {name}...")
    if name == "CNN":
        X_cnn = X_test.values.reshape(-1, X_test.shape[1], 1)
        prob = model.predict(X_cnn, verbose=0).ravel()
    else:
        prob = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)
    top10_lift = np.mean(y_test[prob >= np.percentile(prob, 90)]) / y_test.mean()
    
    results.append({
        "Model": name,
        "AUC": round(auc, 4),
        "PR-AUC": round(ap, 4),
        "Lift@10%": round(top10_lift, 2)
    })

df = pd.DataFrame(results).sort_values("AUC", ascending=False)
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(df.to_string(index=False))

# Save best
best_name = df.iloc[0]["Model"]
best_model = models[best_name]
if "CNN" in best_name:
    best_model.save("best_model.keras")
else:
    joblib.dump(best_model, "best_model.pkl")
print(f"\nWINNER: {best_name} → saved as best_model.pkl")
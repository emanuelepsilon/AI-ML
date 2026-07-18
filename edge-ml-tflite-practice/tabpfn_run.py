import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

df = pd.read_csv(r"C:\Users\emanu\Downloads\student_performance.csv")


y = df["final_score"].astype(float)


student_ids = df["student_id"].copy()


X = df.drop(columns=["student_id", "final_score", "passed"])


num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(exclude=[np.number]).columns

for c in num_cols:
    X[c] = X[c].fillna(X[c].median())

for c in cat_cols:
    mode_val = X[c].mode(dropna=True)
    X[c] = X[c].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")


X = pd.get_dummies(X, drop_first=False)


X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, student_ids, test_size=0.2, random_state=42
)


reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
reg.fit(X_train, y_train)


print("=== MODEL TYPE ===")
print(type(reg))
print(type(reg.model_))
print("\n=== MODEL ARCHITECTURE (TEXT) ===")
print(reg.model_)


pred = reg.predict(X_test)


mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)


out = pd.DataFrame(
    {"student_id": id_test.values, "y_true": y_test.values, "y_pred": pred}
)
out["error"] = out["y_pred"] - out["y_true"]
out["abs_error"] = np.abs(out["error"])
out["abs_pct_error"] = np.where(
    out["y_true"] != 0, np.abs(out["error"] / out["y_true"]) * 100.0, np.nan
)


mape = out["abs_pct_error"].mean(skipna=True)
mdape = out["abs_pct_error"].median(skipna=True)
p90_ape = out["abs_pct_error"].quantile(0.90)
max_ape = out["abs_pct_error"].max(skipna=True)
bias = out["error"].mean()


print(f"\nMAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")
print(f"MAPE (mean)   : {mape:.2f}%")
print(f"MdAPE (median): {mdape:.2f}%")
print(f"P90 APE       : {p90_ape:.2f}%")
print(f"Max APE       : {max_ape:.2f}%")
print(f"Bias (mean error, y_pred - y_true): {bias:.4f}")


out.to_csv("tabpfn_regression_predictions.csv", index=False)

summary = pd.DataFrame(
    [
        {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape_mean_pct": mape,
            "mdape_median_pct": mdape,
            "p90_ape_pct": p90_ape,
            "max_ape_pct": max_ape,
            "bias_mean_error": bias,
        }
    ]
)
summary.to_csv("tabpfn_regression_metrics.csv", index=False)

with open("tabpfn_model_architecture.txt", "w", encoding="utf-8") as f:
    f.write(str(reg.model_))

print("\nSaved: tabpfn_regression_predictions.csv")
print("Saved: tabpfn_regression_metrics.csv")
print("Saved: tabpfn_model_architecture.txt")

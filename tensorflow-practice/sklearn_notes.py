"""Scikit-learn regression pipeline for student exam-score prediction."""

import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("StudentPerformanceFactors.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())

print("\nDtypes:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

X = df.drop("Exam_Score", axis=1)

y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

num_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

cat_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("y min/max:", y.min(), y.max())
print("y_test min/max:", y_test.min(), y_test.max())
print("pred min/max:", preds.min(), preds.max())

mae = mean_absolute_error(y_test, preds)

mse = mean_squared_error(y_test, preds)

rmse = mse**0.5

r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

cv_results = cross_validate(
    model,  # full pipeline (preprocessing + model) is CV'd correctly per fold
    X,
    y,
    cv=5,
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
    n_jobs=-1,  # use all CPU cores
)

cv_mae = -cv_results["test_neg_mean_absolute_error"]
cv_rmse = -cv_results["test_neg_root_mean_squared_error"]
cv_r2 = cv_results["test_r2"]

print("\nCV MAE mean/std:", cv_mae.mean(), cv_mae.std())
print("CV RMSE mean/std:", cv_rmse.mean(), cv_rmse.std())
print("CV R2 mean/std:", cv_r2.mean(), cv_r2.std())

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

coefs = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coef": coefs,
    }
).sort_values("coef", ascending=False)

print("\nTop 10 positive coefficients:")
print(coef_df.head(10))

print("\nTop 10 negative coefficients:")
print(coef_df.tail(10))

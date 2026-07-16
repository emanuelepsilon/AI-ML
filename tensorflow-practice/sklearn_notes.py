"""
This script trains and evaluates a regression model that predicts `Exam_Score`
from student-related features in `StudentPerformanceFactors.csv`.

The goal is educational clarity, so the code is heavily commented line by line.
"""

# pandas is the core tabular-data library we use for loading and inspecting CSV data.
import pandas as pd

# train_test_split: splits data into training and test sets.
# cross_validate: runs k-fold cross-validation and returns multiple metrics.
from sklearn.model_selection import train_test_split, cross_validate

# ColumnTransformer lets us apply different preprocessing to different column groups.
from sklearn.compose import ColumnTransformer

# Pipeline chains steps so preprocessing + model training happen consistently.
from sklearn.pipeline import Pipeline

# OneHotEncoder converts categorical text values into numeric indicator columns.
from sklearn.preprocessing import OneHotEncoder

# SimpleImputer fills missing values using strategies like median or most frequent.
from sklearn.impute import SimpleImputer

# LinearRegression is our baseline regression algorithm.
from sklearn.linear_model import LinearRegression

# Metrics used to evaluate regression performance.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# 1) Load and inspect data
# -------------------------

# Read the CSV file into a pandas DataFrame.
# A DataFrame is a 2D table with labeled rows and columns.
df = pd.read_csv("StudentPerformanceFactors.csv")

# shape returns (rows, columns), useful for a quick size check.
print("Shape:", df.shape)

# columns is an Index object; .tolist() converts it to a normal Python list.
print("\nColumns:")
print(df.columns.tolist())

# head() shows the first 5 rows by default so we can visually inspect sample data.
print("\nHead:")
print(df.head())

# dtypes tells us each column's inferred data type (int, float, object/string, etc.).
print("\nDtypes:")
print(df.dtypes)

# isnull() creates a boolean mask of missing values.
# sum() on booleans counts True values, so this gives missing-count per column.
print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------
# 2) Define features and target
# ------------------------------

# X (features): all columns except the target `Exam_Score`.
# drop(..., axis=1) means "drop a column" (axis=0 would mean rows).
X = df.drop("Exam_Score", axis=1)

# y (target): the column we want to predict.
y = df["Exam_Score"]

# ---------------------------------------------------
# 3) Split into training data and held-out test data
# ---------------------------------------------------

# test_size=0.2 => 20% test, 80% train.
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# 4) Detect numeric and categorical feature groups
# ------------------------------------------------

# select_dtypes(include=[...]) picks columns by data type.
# .columns returns just the column-name index for those selected columns.
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns

# -----------------------------------
# 5) Build preprocessing sub-pipelines
# -----------------------------------

# Numeric pipeline:
# - Fill missing numeric values with median of each numeric column.
# Median is robust to outliers compared to mean.
num_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

# Categorical pipeline:
# - Fill missing categories with most frequent category.
# - One-hot encode categories to numeric columns.
# handle_unknown="ignore" avoids errors if test data contains unseen categories.
# sparse_output=False returns a dense array, which is simpler for inspection.
cat_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# ColumnTransformer applies:
# - num_pipe to numeric columns
# - cat_pipe to categorical columns
# Then it concatenates all transformed features into one final matrix.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
)

# -------------------------------------
# 6) Build full modeling pipeline
# -------------------------------------

# Final pipeline has two steps:
# 1) preprocessor -> transforms raw input data
# 2) regressor -> learns relationship to predict Exam_Score
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# fit() learns:
# - imputer statistics
# - encoder category mappings
# - regression coefficients
# It learns only from training data, which helps avoid data leakage.
model.fit(X_train, y_train)

# predict() generates predicted Exam_Score values for each row in X_test.
preds = model.predict(X_test)

# Quick range sanity checks for true targets and predictions.
print("y min/max:", y.min(), y.max())
print("y_test min/max:", y_test.min(), y_test.max())
print("pred min/max:", preds.min(), preds.max())

# ----------------------------
# 7) Evaluate on held-out test
# ----------------------------

# MAE: average absolute error in "exam score points".
mae = mean_absolute_error(y_test, preds)

# MSE: average squared error (penalizes larger errors more strongly).
mse = mean_squared_error(y_test, preds)

# RMSE: square root of MSE, puts error back into original score units.
rmse = mse ** 0.5

# R2: proportion of target variance explained by the model (higher is better).
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# --------------------------------
# 8) Cross-validation for stability
# --------------------------------

# cross_validate performs K-fold CV (here K=5):
# - splits data into 5 folds
# - trains on 4 folds, validates on 1 fold
# - repeats so each fold is validation once
# This gives more robust performance estimates than one split alone.
cv_results = cross_validate(
    model,  # full pipeline (preprocessing + model) is CV'd correctly per fold
    X,
    y,
    cv=5,
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
    n_jobs=-1,  # use all CPU cores
)

# scikit-learn stores "loss-style" metrics as negative values during scoring.
# We multiply by -1 to recover positive MAE/RMSE.
cv_mae = -cv_results["test_neg_mean_absolute_error"]
cv_rmse = -cv_results["test_neg_root_mean_squared_error"]
cv_r2 = cv_results["test_r2"]

# mean() gives average across folds; std() gives fold-to-fold variability.
print("\nCV MAE mean/std:", cv_mae.mean(), cv_mae.std())
print("CV RMSE mean/std:", cv_rmse.mean(), cv_rmse.std())
print("CV R2 mean/std:", cv_r2.mean(), cv_r2.std())

# ---------------------------------------------
# 9) Interpret linear model via coefficients
# ---------------------------------------------

# named_steps lets us access individual steps inside the pipeline by name.
# `preprocessor` step can return transformed feature names after one-hot encoding.
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# For linear regression, coef_ contains one coefficient per transformed feature.
# Positive coefficient: feature pushes prediction up.
# Negative coefficient: feature pushes prediction down.
coefs = model.named_steps["regressor"].coef_

# Build a tidy table for interpretation.
coef_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coef": coefs,
    }
).sort_values("coef", ascending=False)

# head(10): largest positive coefficients.
print("\nTop 10 positive coefficients:")
print(coef_df.head(10))

# tail(10): most negative coefficients (since sorted descending).
print("\nTop 10 negative coefficients:")
print(coef_df.tail(10))

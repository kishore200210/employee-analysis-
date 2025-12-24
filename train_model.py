# Employee Attrition – Training Pipeline


import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import custom feature engineering
from feature_engineering import FeatureEngineering

# 1. Load Dataset
DATA_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\employe_project\cleaned_file.csv"

df = pd.read_csv(DATA_PATH)

# Drop unwanted index column if present
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("Dataset shape:", df.shape)


# 2. Target & Features
TARGET_COL = "Attrition_Yes"

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

X = df.drop(TARGET_COL, axis=1)

# NaN in Attrition_Yes means "No Attrition"
y = df[TARGET_COL].fillna(0).astype(int)

print("\nTarget distribution:")
print(y.value_counts())


# 3. Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Build Pipeline
pipeline = Pipeline(steps=[
    ("feature_engineering", FeatureEngineering()),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# 5. Train Model
pipeline.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# 7. Save Pipeline
MODEL_PATH = "attrition_pipeline.pkl"
joblib.dump(pipeline, MODEL_PATH)

print(f"\n✅ Pipeline saved successfully as {MODEL_PATH}")

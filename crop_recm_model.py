import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("crop_recommendation.csv")

# Features and label
X = data.drop("label", axis=1)
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "crop_model.pkl")

print("Crop model trained successfully!")
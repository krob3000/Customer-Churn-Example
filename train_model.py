
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset

df = pd.read_csv("C:\\Users\\User\\CustomerChurnPractice\\data.csv")  # Or use full path if needed
print("✅ Data loaded successfully!")
print("Columns in dataset:", df.columns.tolist())

# Detect target column
possible_targets = ["Churn", "Exited", "Target"]
target_column = None
for col in df.columns:
    if col in possible_targets:
        target_column = col
        break

if not target_column:
    raise ValueError("❌ Target column not found! Please rename your target column to 'Churn' or 'Exited'.")

print(f"✅ Using target column: {target_column}")

# Prepare features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

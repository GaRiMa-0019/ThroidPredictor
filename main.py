import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/raw/thyroid.csv")

print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# -----------------------------
# Quick Dataset Inspection
# -----------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Data Preprocessing
# -----------------------------

# Remove column with excessive missing values
df.drop(columns=["TBG"], inplace=True)

# Fill missing values in categorical column
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

# Fill missing values in numerical columns using median
numeric_columns = ["TSH", "T3", "TT4", "T4U", "FTI"]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nRemaining Missing Values:")
print(df.isnull().sum())

# -----------------------------
# Features (X) and Target (y)
# -----------------------------

# Remove patient ID because it is only an identifier
df.drop(columns=["patient_id"], inplace=True)

# Features
X = df.drop(columns=["target"])

# Target
y = df["target"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Classes:")
print(y.unique())

# -----------------------------
# Encode Target Variable
# -----------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nEncoded Target Classes:")
print(label_encoder.classes_)

# -----------------------------
# Find Categorical Columns
# -----------------------------

categorical_columns = X.select_dtypes(include=["string", "object"]).columns

print("\nCategorical Columns:")
print(categorical_columns)

# -----------------------------
# One-Hot Encode Categorical Features
# -----------------------------

X = pd.get_dummies(X, columns=categorical_columns)

print("\nEncoded Feature Shape:", X.shape)
print(X.head())

# -----------------------------
# Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Feature Shape:", X_train.shape)
print("Testing Feature Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# -----------------------------
# Train Decision Tree
# -----------------------------

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X_train, y_train)

print("\n✅ Decision Tree model trained successfully!")

# -----------------------------
# Make Predictions
# -----------------------------

predictions = decision_tree.predict(X_test)

decoded_predictions = label_encoder.inverse_transform(predictions)

print("\nFirst 10 Decoded Predictions:")
print(decoded_predictions[:10])

# -----------------------------
# Accuracy Score
# -----------------------------
accuracy = accuracy_score(y_test, predictions)  

print(f"\nAccuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# -----------------------------
# Random Forest Model
# -----------------------------

random_forest = RandomForestClassifier(random_state=42)

random_forest.fit(X_train, y_train)

# Predictions using Random Forest
rf_predictions = random_forest.predict(X_test)

rf_decoded_predictions = label_encoder.inverse_transform(rf_predictions)

print("\nFirst 10 Random Forest Predictions:")
print(rf_decoded_predictions[:10])

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:", rf_accuracy)

print("\n========== Model Comparison ==========")
print(f"Decision Tree Accuracy : {accuracy:.2%}")
print(f"Random Forest Accuracy : {rf_accuracy:.2%}")

if accuracy > rf_accuracy:
    print("🏆 Best Model: Decision Tree")
else:
    print("🏆 Best Model: Random Forest")


# -----------------------------
# Confusion Matrix Heatmap
# -----------------------------

labels = label_encoder.classes_

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig("outputs/figures/confusion_matrix_heatmap.png")
plt.show()

# -----------------------------
# Save Trained Models
# -----------------------------

joblib.dump(decision_tree, "outputs/models/decision_tree.pkl")
joblib.dump(random_forest, "outputs/models/random_forest.pkl")

print("\n✅ Models saved successfully!")

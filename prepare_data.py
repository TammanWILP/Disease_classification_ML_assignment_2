import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/blood_samples_data.csv")  # Read data
target_col = df.columns[-1]  # Get label column

le = LabelEncoder()
df["encoded_" + target_col] = le.fit_transform(df[target_col])  # Add encoded labels in a new column

X = df.drop([target_col, "encoded_" + target_col], axis=1)
y_encoded = df["encoded_" + target_col]
y_original = df[target_col]

X_train, X_test, y_train_encoded, y_test_encoded, y_train_orig, y_test_orig = train_test_split(
    X, y_encoded, y_original, test_size=0.2, random_state=42, stratify=y_encoded
)

train_df = X_train.copy()
train_df[target_col] = y_train_encoded  # Use encoded labels

test_df = X_test.copy()
test_df[target_col] = y_test_orig  # Use original labels

# Create test set without label column
test_without_target_df = X_test.copy()
test_without_target_df.to_csv("data/test_without_target.csv", index=False)

train_df.to_csv("data/train.csv", index=False)  # Save train (encoded labels)
test_df.to_csv("data/test.csv", index=False)    # Save test (original labels)

print("train.csv (encoded labels), test.csv (original labels), and test_without_target.csv (no label) created")
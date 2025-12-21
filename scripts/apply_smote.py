# apply_smote.py
import pandas as pd
from imblearn.over_sampling import SMOTE

# Load training data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]

print("Before SMOTE:", y_train.value_counts().to_dict())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_bal.value_counts().to_dict())

# Save balanced data
X_train_bal.to_csv('data/processed/X_train_balanced.csv', index=False)
y_train_bal.to_csv('data/processed/y_train_balanced.csv', index=False)

print("✅ SMOTE completed and saved.")
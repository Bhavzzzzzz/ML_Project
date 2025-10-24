import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# 1️⃣ Load datasets
# ============================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# ============================
# 2️⃣ Handle categorical variable ("Lifestyle Activities")
# ============================
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit on training data and transform both train/test
train_encoded = encoder.fit_transform(train_df[['Lifestyle Activities']])
test_encoded = encoder.transform(test_df[['Lifestyle Activities']])

# Convert to DataFrames
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(['Lifestyle Activities']))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(['Lifestyle Activities']))

# Add back to original data (dropping old column)
train_df = pd.concat([train_df.drop('Lifestyle Activities', axis=1), train_encoded_df], axis=1)
test_df = pd.concat([test_df.drop('Lifestyle Activities', axis=1), test_encoded_df], axis=1)

# ============================
# 3️⃣ Add Derived Features (refined for synergy)
# ============================
lifestyle_yes_col = 'Lifestyle Activities_Yes'
train_df['Initial_Health_Sq'] = train_df['Initial Health Score'] ** 2
test_df['Initial_Health_Sq'] = test_df['Initial Health Score'] ** 2
train_df['Total_Sessions'] = train_df['Therapy Hours'] + train_df['Follow-Up Sessions']
test_df['Total_Sessions'] = test_df['Therapy Hours'] + test_df['Follow-Up Sessions']
train_df['Therapy_Lifestyle'] = train_df['Therapy Hours'] * train_df[lifestyle_yes_col]
test_df['Therapy_Lifestyle'] = test_df['Therapy Hours'] * test_df[lifestyle_yes_col]
train_df['Sleep_Engagement'] = train_df['Average Sleep Hours'] * train_df['Total_Sessions']
test_df['Sleep_Engagement'] = test_df['Average Sleep Hours'] * test_df['Total_Sessions']

# ============================
# 4️⃣ Handle missing values (only numerics, exclude 'Id' and target)
# ============================
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop(['Id', 'Recovery Index'], errors='ignore').tolist()
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
test_df[numeric_cols] = test_df[numeric_cols].fillna(train_df[numeric_cols].mean())  # Use train means to avoid leakage

# ============================
# 5️⃣ Split data (exclude 'Id')
# ============================
feature_cols = [col for col in train_df.columns if col not in ['Id', 'Recovery Index']]
X = train_df[feature_cols]
y = train_df["Recovery Index"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 1684)

# ============================
# 6️⃣ Train Linear Regression
# ============================
model = LinearRegression()
model.fit(X_train, y_train)

# Check model parameters
print("Model Coefficients (by feature):")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
})
print(coef_df.sort_values('Coefficient', key=abs, ascending=False).round(4))  # Sorted by abs value for insight

print(f"\nIntercept: {model.intercept_:.4f}")

# ============================
# 7️⃣ Evaluate
# ============================
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Validation MSE: {mse:.3f}")
print(f"Validation R²: {r2:.3f}")

# ============================
# 8️⃣ Predict on test.csv (exclude 'Id' for prediction; unrounded floats, clipped to [10,100])
# ============================
test_features = test_df[feature_cols]
test_pred = model.predict(test_features)
test_pred = np.clip(test_pred, 10, 100)  # Clip to range [10,100] (unrounded)

# ============================
# 9️⃣ Create submission.csv (use test 'Id', ignore sample length)
# ============================
submission = pd.DataFrame({
    "Id": test_df["Id"],  # Correct column name: 'Id' (not 'ID')
    "Recovery Index": test_pred
})
submission.to_csv("submission_LR.csv", index=False)

print("✅ submission.csv created successfully!")
print("\nFirst few submission rows:")
print(submission.head())
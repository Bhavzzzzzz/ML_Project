import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from io import StringIO
import matplotlib.pyplot as plt

# Suppress ignorable warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print(f"Running on PyMC v{pm.__version__}")

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================
print("\n--- Step 1: Loading and Preparing Data ---")

# --- Load your training dataset ---
file_path = 'train.csv' 

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'.")
    print(f"Data has {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print(f"\n*** ERROR: File not found at '{file_path}' ***")
    exit()

# --- Preprocessing ---
column_to_map = 'Lifestyle Activities'
if column_to_map not in df.columns:
    print(f"\n*** ERROR: Column '{column_to_map}' not found. ***")
    exit()

print(f"\nPreprocessing column: '{column_to_map}'...")
df[column_to_map] = df[column_to_map].map({'Yes': 1, 'No': 0})

if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Define features (X) and target (y)
X = df.drop('Recovery Index', axis=1)
y = df['Recovery Index']

# Get feature names for later
feature_names = X.columns.tolist()
n_features = len(feature_names)
print(f"Total features being used ({n_features}): {feature_names}")

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation and scaling complete.")

# =============================================================================
# STEP 2 & 3: DEFINE AND TRAIN THE PYMC MODEL (WITH STUDENT-T)
# =============================================================================
print("\n--- Step 2 & 3: Defining and Training Model ---")

with pm.Model() as model:
    # --- Priors ---
    b0 = pm.Normal('b0_intercept', mu=y_train.mean(), sigma=y_train.std())
    b = pm.Normal('b_features', mu=0, sigma=1, shape=n_features)
    sigma = pm.HalfNormal('sigma', sigma=y_train.std())
    
    # ---CHANGE 1: Add a prior for 'nu' (degrees of freedom) ---
    # This controls the "robustness". A low 'nu' is robust to outliers.
    nu = pm.Gamma('nu', alpha=2, beta=0.1)

    # --- Likelihood ---
    mu = b0 + pm.math.dot(X_train_scaled, b)
    
    # ---CHANGE 2: Use StudentT likelihood instead of Normal ---
    y_obs = pm.StudentT('y_obs', mu=mu, sigma=sigma, nu=nu, observed=y_train.values)
    # y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train.values) # <-- This was the old way

    # --- Run Sampler (Training) ---
    print("Starting MCMC sampling (using 4 chains)...")
    idata = pm.sample(1000, tune=1000, chains=4, return_inferencedata=True)
    print("Sampling complete!")

# =============================================================================
# STEP 4: ANALYZE MODEL RESULTS
# =============================================================================
print("\n--- Step 4: Model Results Analysis ---")

# ---CHANGE 3: Add 'nu' to the list of variables to check ---
summary = az.summary(idata, var_names=['b0_intercept', 'b_features', 'sigma', 'nu'])
summary.index = ['Intercept'] + feature_names + ['Model Error (sigma)', 'nu (Robustness)']
print("\nModel Parameter Summary:")
print(summary)
# --- Look at the 'nu (Robustness)' row. If 'nu' is small (e.g., < 10), it means 
# --- outliers were present and the StudentT model was a good choice.

print("\n--- Visualizing Posterior Distributions ---")
try:
    az.plot_posterior(
        idata,
        var_names=['b_features'],
        coords={'b_features_dim_0': feature_names}
    )
    plt.savefig('posterior_plot.png')
    print("Saved 'posterior_plot.png' with feature effects.")
except Exception as e:
    print(f"Could not create plot: {e}")

# =============================================================================
# STEP 5: PREDICT ON VALIDATION SET
# =============================================================================
print("\n--- Step 5: Generating Predictions on Validation (Hold-Out) Data ---")

# 1. Extract the posterior samples
posterior = idata.posterior
b0_samples = posterior['b0_intercept'].values.flatten()
b_samples = posterior['b_features'].values.reshape(-1, n_features) 
sigma_samples = posterior['sigma'].values.flatten()
# -CHANGE 4: Extract the 'nu' samples ---
nu_samples = posterior['nu'].values.flatten()

# 2. Manually calculate 'mu' for the *validation test set*
mu_pred = b0_samples[:, None] + b_samples @ X_test_scaled.T

# 3. Generate new 'y' values
# -CHANGE 5: Use StudentT.dist for prediction ---
posterior_predictive_dist = pm.StudentT.dist(
    mu=mu_pred, 
    sigma=sigma_samples[:, None], 
    nu=nu_samples[:, None]
)
# posterior_predictive_dist = pm.Normal.dist(mu=mu_pred, sigma=sigma_samples[:, None]) # <-- Old way
y_posterior_pred = pm.draw(posterior_predictive_dist, draws=1)

print("Validation prediction generation complete.")

# --- Analyze Prediction on first person in validation set ---
test_index = 0
all_preds_for_one = y_posterior_pred[:, test_index]
mean_pred = all_preds_for_one.mean()
pred_hdi = az.hdi(all_preds_for_one, hdi_prob=0.94)
actual_value = y_test.iloc[test_index]

print("\n--- Example Prediction (First Person in Validation Set) ---")
print(f"Features (unscaled): \n{X_test.iloc[test_index].to_string()}")
print(f"\nActual 'Recovery Index': {actual_value}")
print(f"Model Mean Prediction: {mean_pred:.2f}")
print(f"Model 94% Plausible Range (HDI): [{pred_hdi[0]:.2f}, {pred_hdi[1]:.2f}]")

# =============================================================================
# STEP 6: CREATE SUBMISSION FILE FOR KAGGLE
# =============================================================================
print("\n--- Step 6: Generating Kaggle Submission File ---")

# --- 1. Load the official test.csv from Kaggle ---
try:
    test_df = pd.read_csv('test.csv')
    print("Successfully loaded 'test.csv'.")
except FileNotFoundError:
    print("\n*** ERROR: 'test.csv' not found. ***")
    print("Please download 'test.csv' from the Kaggle competition page.")
    exit()

# --- 2. Store the 'Id' column ---
test_ids = test_df['Id']

# --- 3. Preprocess the test_df EXACTLY like the training data ---
print("Preprocessing official test data...")
test_df['Lifestyle Activities'] = test_df['Lifestyle Activities'].map({'Yes': 1, 'No': 0})

# We are back to the simpler model, so no new feature is added
test_df_processed = test_df[feature_names]

# Use the *same* scaler from Step 1
test_scaled = scaler.transform(test_df_processed)
print("Test data processed and scaled.")

# --- 4. Generate Predictions for the official test set ---
print("Generating predictions on test set...")
mu_pred_submission = b0_samples[:, None] + b_samples @ test_scaled.T

# ---CHANGE 6: Use StudentT.dist for submission predictions ---
posterior_predictive_dist_sub = pm.StudentT.dist(
    mu=mu_pred_submission, 
    sigma=sigma_samples[:, None],
    nu=nu_samples[:, None]
)
# posterior_predictive_dist_sub = pm.Normal.dist(...) # <-- Old way
y_pred_submission_samples = pm.draw(posterior_predictive_dist_sub, draws=1)

# We want the mean prediction for each person
mean_predictions = y_pred_submission_samples.mean(axis=0)
print("Predictions generated.")

# --- 5. Create and Save the submission.csv file ---
submission_df = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': mean_predictions
})

submission_df.to_csv('submission.csv', index=False)

print("\n--- 'submission.csv' created successfully! ---")
print("\n--- Bayesian Model Workflow Finished ---")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- Load Data ---
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: train.csv or test.csv not found")
    exit()

test_ids = test_df['Id']

X = train_df.drop(['Id', 'Recovery Index'], axis=1)
y = train_df['Recovery Index']
X_test = test_df.drop('Id', axis=1)

# --- Feature Engineering ---
def preprocess_and_engineer_features(df):
    df['Lifestyle Activities'] = df['Lifestyle Activities'].fillna('No')
    df['Lifestyle_Active'] = df['Lifestyle Activities'].apply(lambda x: 1 if x == 'Yes' else 0)
    df = df.drop('Lifestyle Activities', axis=1)

    T = df['Therapy Hours']
    H = df['Initial Health Score']
    S = df['Average Sleep Hours']
    F = df['Follow-Up Sessions']
    L = df['Lifestyle_Active']
    epsilon = 1e-6

    df['Initial_Health_Inverse'] = 1 / (H + epsilon)
    df['Sleep_Hours_Inverse'] = 1 / (S + epsilon)
    df['FollowUp_Inverse_Squared'] = 1 / (F**2 + epsilon)
    df['Therapy_Health_Product'] = T * H
    df['Sleep_FollowUp_Product'] = S * F
    df['Total_Effort_Scaled_Health'] = (T + F) * H
    df['Total_Baseline_Capacity'] = H * S
    df['Therapy_Per_Sleep_Ratio'] = T / (S + epsilon)
    df['Health_Per_Therapy_Ratio'] = H / (T + epsilon)
    df['Sleep_Per_FollowUp_Ratio'] = S / (F + epsilon)
    df['FollowUp_Per_Therapy_Ratio'] = F / (T + epsilon)
    df['Total_Commitment_Per_Sleep'] = (T + F) / (S + epsilon)
    df['Active_Health_Boost'] = H * L
    df['Treatment_Synergy_Active'] = T * F * L
    df['Health_Sleep_Active_Triple'] = H * S * L
    df['Sleep_Active_Interaction'] = S * L
    return df

# --- Split into Train & Validation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocess ---
X_train_processed = preprocess_and_engineer_features(X_train.copy())
X_val_processed = preprocess_and_engineer_features(X_val.copy())
X_test_processed = preprocess_and_engineer_features(X_test.copy())

X_val_processed = X_val_processed[X_train_processed.columns]
X_test_processed = X_test_processed[X_train_processed.columns]

# --- Scale ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_val_scaled = scaler.transform(X_val_processed)
X_test_scaled = scaler.transform(X_test_processed)

# --- Find Training & Validation Errors for Different Alphas ---
alphas = np.logspace(-3, 2, 20)  # from 0.001 to 100
train_errors = []
val_errors = []

for alpha in alphas:
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    train_errors.append(train_rmse)
    val_errors.append(val_rmse)

# --- Plot Training vs Validation Error Curve ---
plt.figure(figsize=(8,6))
plt.plot(alphas, train_errors, marker='o', label='Training RMSE', color='blue')
plt.plot(alphas, val_errors, marker='o', label='Validation RMSE', color='orange')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('RMSE')
plt.title('Training vs Validation RMSE for Ridge Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Train Final Model (best alpha) ---
best_alpha = alphas[np.argmin(val_errors)]
print(f"Best alpha: {best_alpha:.4f}")

model = Ridge(alpha=best_alpha, random_state=42)
model.fit(X_train_scaled, y_train)

# --- Predict on Test Set ---
test_predictions = model.predict(X_test_scaled)
submission_df = pd.DataFrame({'Id': test_ids, 'Recovery Index': test_predictions})
submission_df.to_csv('ridge_regression.csv', index=False)
print("Saved submission file as ridge_regression.csv")

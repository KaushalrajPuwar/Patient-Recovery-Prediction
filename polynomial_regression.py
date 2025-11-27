import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Load data ---
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

# --- Split into Train/Validation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Preprocess ---
X_train_processed = preprocess_and_engineer_features(X_train.copy())
X_val_processed = preprocess_and_engineer_features(X_val.copy())
X_test_processed = preprocess_and_engineer_features(X_test.copy())

X_val_processed = X_val_processed[X_train_processed.columns]
X_test_processed = X_test_processed[X_train_processed.columns]

# --- Try multiple polynomial degrees (1–3) ---
degrees = [1, 2, 3]
train_errors, val_errors = [], []

for degree in degrees:
    print(f"Processing polynomial degree {degree}...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_processed)
    X_val_poly = poly.transform(X_val_processed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_val_scaled = scaler.transform(X_val_poly)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    train_errors.append(train_rmse)
    val_errors.append(val_rmse)

# --- Plot Training vs Validation Error Curve ---
plt.figure(figsize=(8,6))
plt.plot(degrees, train_errors, marker='o', color='blue', label='Training RMSE')
plt.plot(degrees, val_errors, marker='o', color='orange', label='Validation RMSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('RMSE')
plt.title('Training vs Validation RMSE for Polynomial Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Train final model with best degree ---
best_degree = degrees[np.argmin(val_errors)]
print(f"Best polynomial degree: {best_degree}")

poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly_full = poly.fit_transform(preprocess_and_engineer_features(X.copy()))
X_test_poly = poly.transform(preprocess_and_engineer_features(X_test.copy()))

scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X_train_poly_full)
X_test_scaled = scaler.transform(X_test_poly)

model = LinearRegression()
model.fit(X_train_scaled_full, y)

test_predictions = model.predict(X_test_scaled)
submission_df = pd.DataFrame({'Id': test_ids, 'Recovery Index': test_predictions})
submission_df.to_csv('polynomial_regression.csv', index=False)
print("Saved submission file as polynomial_regression.csv")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 


try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')


except FileNotFoundError:
    print("Error")
    exit()
test_ids = test_df['Id']
X_train = train_df.drop(['Id', 'Recovery Index'], axis=1)
y_train = train_df['Recovery Index']
X_test = test_df.drop('Id', axis=1)

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

X_train_processed = preprocess_and_engineer_features(X_train.copy())
X_test_processed = preprocess_and_engineer_features(X_test.copy())
X_test_processed = X_test_processed[X_train_processed.columns]

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train_processed)
X_test_poly = poly.transform(X_test_processed)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

train_predictions = model.predict(X_train_scaled)

train_mean = np.mean(train_predictions)
train_std = np.std(train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)

print(f"Training Prediction Mean: {train_mean:.4f}")
print(f"Training Prediction Std Dev: {train_std:.4f}")
print(f"Training RMSE Score: {train_rmse:.4f}")

test_predictions = model.predict(X_test_scaled)

submission_df = pd.DataFrame({
    'Id': test_ids,
    'Recovery Index': test_predictions
})

output_filename = 'polynomial_regression.csv'
submission_df.to_csv(output_filename, index=False)

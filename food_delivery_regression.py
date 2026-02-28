# ======================================================
# 1. Import Required Libraries
# ======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ======================================================
# 2. Load Dataset and Initial Exploration
# ======================================================
df = pd.read_csv("food_delivery_time_estimation.csv")
pd.set_option('display.max_columns', None)

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)
print(df.isnull().sum())


# ======================================================
# 3. Scatter Plot — Raw Relationship
# ======================================================
plt.figure(figsize=(8, 5))
plt.scatter(df['distance_km'], df['delivery_time'], color='blue')
plt.title('Raw Relationship: Distance vs. Delivery Time')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (min)')
plt.grid(True)
plt.savefig('01_scatter_distance_vs_time.png', dpi=300)
plt.show()


# ======================================================
# 4. Box Plots — Outlier Inspection
# ======================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=df["distance_km"])
plt.title("Distance (km) — Box Plot")

plt.subplot(1, 2, 2)
sns.boxplot(x=df["delivery_time"])
plt.title("Delivery Time (min) — Box Plot")

plt.tight_layout()
plt.savefig('02_boxplots_raw.png', dpi=300)
plt.show()


# ======================================================
# 5. Correlation Matrix
# ======================================================
corr_matrix = df.select_dtypes(include='number').corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig('03_correlation_matrix.png', dpi=300)
plt.show()


# ======================================================
# 6. Shared Preprocessing (applied to both models)
# ======================================================
def preprocess(data):
    data = data.copy()
    data.drop("order_id", axis=1, inplace=True)
    data = pd.get_dummies(data, columns=["weather"], drop_first=True)
    return data

df_processed = preprocess(df)


# ======================================================
# 7. Outlier Treatment Methods
# Three approaches applied to delivery_time:
#   1. IQR Removal   — drops rows outside 1.5*IQR bounds
#   2. Winsorization — caps values at 5th and 95th percentile
#   3. Z-Score       — drops rows where |z| > 3
# ======================================================

# --- 7a. IQR Removal ---
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df_iqr = remove_outliers_iqr(df_processed, "delivery_time")

# --- 7b. Winsorization (Capping) ---
def winsorize(data, column, lower_pct=0.05, upper_pct=0.95):
    data = data.copy()
    lower = data[column].quantile(lower_pct)
    upper = data[column].quantile(upper_pct)
    data[column] = data[column].clip(lower, upper)
    return data

df_wins = winsorize(df_processed, "delivery_time")

# --- 7c. Z-Score Removal ---
def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    return data[z_scores <= threshold]

df_zscore = remove_outliers_zscore(df_processed, "delivery_time")

print(f"Dataset size — Original     : {df_processed.shape}")
print(f"Dataset size — IQR Removal  : {df_iqr.shape}")
print(f"Dataset size — Winsorization: {df_wins.shape}")
print(f"Dataset size — Z-Score      : {df_zscore.shape}")

# Box plots after each treatment
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, data, title in zip(axes,
    [df_processed, df_iqr, df_wins, df_zscore],
    ["Original", "IQR Removal", "Winsorization", "Z-Score"]):
    sns.boxplot(x=data["delivery_time"], ax=ax)
    ax.set_title(f"{title}\nn={len(data)}")
plt.suptitle("Delivery Time — Outlier Treatment Comparison", y=1.02)
plt.tight_layout()
plt.savefig('04_boxplots_outlier_comparison.png', dpi=300)
plt.show()


# ======================================================
# 8. Model Training Function
# ======================================================
cols_to_scale = ["distance_km", "rider_speed"]

def train_model(data, label):
    Y = data["delivery_time"]
    X = data.drop("delivery_time", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    print(f"\n--- {label} ---")
    print(f"  Intercept (β₀) : {model.intercept_:.2f}")
    coeff_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_})
    print(coeff_df.to_string(index=False))
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f} min")

    return y_test, y_pred, r2, rmse, label


# ======================================================
# 9. Train All Four Models
# ======================================================
y_test_base,   y_pred_base,   r2_base,   rmse_base,   label_base   = train_model(df_processed, "Base Model (With Outliers)")
y_test_iqr,    y_pred_iqr,    r2_iqr,    rmse_iqr,    label_iqr    = train_model(df_iqr,       "IQR Removal")
y_test_wins,   y_pred_wins,   r2_wins,   rmse_wins,   label_wins   = train_model(df_wins,      "Winsorization (Capping)")
y_test_zscore, y_pred_zscore, r2_zscore, rmse_zscore, label_zscore = train_model(df_zscore,    "Z-Score Removal")


# ======================================================
# 10. Individual Fit Plots
# ======================================================
def plot_fit(y_test, y_pred, r2, rmse, label, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='darkgreen', alpha=0.7)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit (Y=X)')
    plt.title(f'{label}\nR²: {r2:.2f} | RMSE: {rmse:.2f} min')
    plt.xlabel('Actual Delivery Time (min)')
    plt.ylabel('Predicted Delivery Time (min)')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.show()

plot_fit(y_test_base,   y_pred_base,   r2_base,   rmse_base,   "Base Model (With Outliers)",    "05_fit_base_model.png")
plot_fit(y_test_iqr,    y_pred_iqr,    r2_iqr,    rmse_iqr,    "IQR Removal",                   "06_fit_iqr_model.png")
plot_fit(y_test_wins,   y_pred_wins,   r2_wins,   rmse_wins,   "Winsorization (Capping)",       "07_fit_wins_model.png")
plot_fit(y_test_zscore, y_pred_zscore, r2_zscore, rmse_zscore, "Z-Score Removal",               "08_fit_zscore_model.png")


# ======================================================
# 10b. Residual Analysis
# Checks linear regression assumptions:
# - Residuals should be randomly scattered around 0 (no pattern)
# - Residuals should be approximately normally distributed
# ======================================================
def plot_residuals(y_test, y_pred, label, filename):
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, color='steelblue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title(f'Residuals vs Fitted — {label}')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, color='steelblue')
    plt.title(f'Residual Distribution — {label}')
    plt.xlabel('Residuals')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"\nResidual Summary — {label}")
    print(f"  Mean   : {residuals.mean():.4f}  (should be ~0)")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Min    : {residuals.min():.4f}")
    print(f"  Max    : {residuals.max():.4f}")

plot_residuals(y_test_base,   y_pred_base,   "Base Model",             "09_residuals_base.png")
plot_residuals(y_test_iqr,    y_pred_iqr,    "IQR Removal",            "10_residuals_iqr.png")
plot_residuals(y_test_wins,   y_pred_wins,   "Winsorization",          "11_residuals_wins.png")
plot_residuals(y_test_zscore, y_pred_zscore, "Z-Score Removal",        "12_residuals_zscore.png")


# ======================================================
# 11. Model Comparison Summary
# ======================================================
comparison = pd.DataFrame({
    "Model":      [label_base, label_iqr, label_wins, label_zscore],
    "R²":         [round(r2_base, 4),    round(r2_iqr, 4),    round(r2_wins, 4),    round(r2_zscore, 4)],
    "RMSE (min)": [round(rmse_base, 4),  round(rmse_iqr, 4),  round(rmse_wins, 4),  round(rmse_zscore, 4)],
    "Improvement": [
        "—",
        f"RMSE ↓ {((rmse_base - rmse_iqr)    / rmse_base * 100):.1f}%",
        f"RMSE ↓ {((rmse_base - rmse_wins)   / rmse_base * 100):.1f}%",
        f"RMSE ↓ {((rmse_base - rmse_zscore) / rmse_base * 100):.1f}%",
    ]
})

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(comparison.to_string(index=False))
print("=" * 70)
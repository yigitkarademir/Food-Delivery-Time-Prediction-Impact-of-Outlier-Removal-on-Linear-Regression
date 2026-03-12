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
# 6. Shared Preprocessing
# ======================================================
def preprocess(data):
    data = data.copy()
    data.drop("order_id", axis=1, inplace=True)
    data = pd.get_dummies(data, columns=["weather"], drop_first=True)
    return data


df_processed = preprocess(df)

# ======================================================
# 7. GLOBAL TRAIN/TEST SPLIT & SCALING
# ======================================================
X = df_processed.drop("delivery_time", axis=1)
Y = df_processed["delivery_time"]

X_train_base, X_test, y_train_base, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

cols_to_scale = ["distance_km", "rider_speed"]
scaler = StandardScaler()
X_train_base[cols_to_scale] = scaler.fit_transform(X_train_base[cols_to_scale])
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])


# ======================================================
# 8. Outlier Treatment Methods (SADECE EĞİTİM SETİNE UYGULANIR)
# ======================================================
def apply_iqr_train(X_tr, y_tr):
    Q1 = y_tr.quantile(0.25)
    Q3 = y_tr.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (y_tr >= lower) & (y_tr <= upper)
    return X_tr[mask], y_tr[mask]


X_train_iqr, y_train_iqr = apply_iqr_train(X_train_base, y_train_base)


def apply_winsorize_train(X_tr, y_tr, lower_pct=0.05, upper_pct=0.95):
    y_tr_wins = y_tr.copy()
    lower = y_tr_wins.quantile(lower_pct)
    upper = y_tr_wins.quantile(upper_pct)
    y_tr_wins = y_tr_wins.clip(lower, upper)
    return X_tr, y_tr_wins


X_train_wins, y_train_wins = apply_winsorize_train(X_train_base, y_train_base)


def apply_zscore_train(X_tr, y_tr, threshold=3):
    z_scores = np.abs((y_tr - y_tr.mean()) / y_tr.std())
    mask = z_scores <= threshold
    return X_tr[mask], y_tr[mask]


X_train_zscore, y_train_zscore = apply_zscore_train(X_train_base, y_train_base)

# Görselleştirme
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, data, title in zip(axes,
                           [y_train_base, y_train_iqr, y_train_wins, y_train_zscore],
                           ["Original (Train)", "IQR Removal (Train)", "Winsorization (Train)", "Z-Score (Train)"]):
    sns.boxplot(x=data, ax=ax)
    ax.set_title(f"{title}\nn={len(data)}")
plt.suptitle("Delivery Time (Train Set) — Outlier Treatment Comparison", y=1.02)
plt.tight_layout()
plt.savefig('04_boxplots_outlier_comparison.png', dpi=300)
plt.show()


# ======================================================
# 9. Model Training & Evaluation Function
# ======================================================
def train_and_evaluate(X_train, y_train, label):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {label} ---")
    print(f"  Intercept (β₀) : {model.intercept_:.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f} min")

    return y_test, y_pred, r2, rmse, label


# ======================================================
# 10. Train All Models
# ======================================================
y_test_base, y_pred_base, r2_base, rmse_base, label_base = train_and_evaluate(X_train_base, y_train_base, "Base Model")
y_test_iqr, y_pred_iqr, r2_iqr, rmse_iqr, label_iqr = train_and_evaluate(X_train_iqr, y_train_iqr, "IQR Removal")
y_test_wins, y_pred_wins, r2_wins, rmse_wins, label_wins = train_and_evaluate(X_train_wins, y_train_wins,
                                                                              "Winsorization")
y_test_zscore, y_pred_zscore, r2_zscore, rmse_zscore, label_zscore = train_and_evaluate(X_train_zscore, y_train_zscore,
                                                                                        "Z-Score Removal")


# ======================================================
# 11. Individual Fit Plots & Residual Analysis
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


models_data = [
    (y_test_base, y_pred_base, r2_base, rmse_base, label_base, "base"),
    (y_test_iqr, y_pred_iqr, r2_iqr, rmse_iqr, label_iqr, "iqr"),
    (y_test_wins, y_pred_wins, r2_wins, rmse_wins, label_wins, "wins"),
    (y_test_zscore, y_pred_zscore, r2_zscore, rmse_zscore, label_zscore, "zscore")
]

for i, (yt, yp, r2, rmse, lbl, suffix) in enumerate(models_data):
    fit_filename = f"{5 + i:02d}_fit_{suffix}_model.png"
    res_filename = f"{9 + i:02d}_residuals_{suffix}.png"
    plot_fit(yt, yp, r2, rmse, lbl, fit_filename)
    plot_residuals(yt, yp, lbl, res_filename)

# ======================================================
# 12. Final Model Comparison Summary
# ======================================================
comparison = pd.DataFrame({
    "Model": [label_base, label_iqr, label_wins, label_zscore],
    "R²": [round(r2_base, 4), round(r2_iqr, 4), round(r2_wins, 4), round(r2_zscore, 4)],
    "RMSE (min)": [round(rmse_base, 4), round(rmse_iqr, 4), round(rmse_wins, 4), round(rmse_zscore, 4)]
})

comparison["Improvement"] = comparison["RMSE (min)"].apply(
    lambda
        x: f"RMSE ↓ {((rmse_base - x) / rmse_base * 100):.1f}%" if x < rmse_base else f"RMSE ↑ {((x - rmse_base) / rmse_base * 100):.1f}%"
)
comparison.loc[0, "Improvement"] = "—"

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON (TEST SET PERFORMANCE)")
print("=" * 75)
print(comparison.to_string(index=False))
print("=" * 75)
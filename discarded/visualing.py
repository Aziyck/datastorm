import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Date
# ==============================
nim = pd.read_csv("csv/NIM_PD_LGD_Baza.csv")

cols_to_use = [
    "Rata_politica_BNM_%", 
    "PIB_real_crestere_%", 
    "IPC_medie_anuala_%", 
    "Somaj_%",
    "Curs_MDL_pe_USD", 
    "Remitente_USD_mld", 
    "FX_YoY_%", 
    "Remitente_YoY_%",
    "Randament_active_%", 
    "Cost_depozite_%", 
    "NIM_%", 
    "PD_%", 
    "LGD_%", 
    "Cost_risc_%", 
    "EA_index", 
    "NII_index", 
    "LLP_index"
]
all_cols = cols_to_use + ["PnL_net_index"]

data = nim.dropna(subset=all_cols)
X = data[['PIB_real_crestere_%', "IPC_medie_anuala_%", 'NIM_%']]
y = data['PnL_net_index']

# ==============================
# VIF
# ==============================
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# ==============================
# LOO-CV function
# ==============================
def loo_cv(model, X, y):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        y_true.append(y_test.values[0])
        y_pred.append(pred[0])
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse, y_true, y_pred

# ==============================
# Linear Regression
# ==============================
lin_reg = LinearRegression()
r2_lin, rmse_lin, y_true_lin, y_pred_lin = loo_cv(lin_reg, X, y)

X2 = sm.add_constant(X)
model_sm = sm.OLS(y, X2).fit()

print("\n=== Regressie Liniară (LOO-CV) ===")
print(f"R²: {r2_lin:.3f} | RMSE: {rmse_lin:.3f}")
print(model_sm.summary())

# ==============================
# Ridge Regression
# ==============================
ridge = Ridge(alpha=1.0)
r2_ridge, rmse_ridge, y_true_ridge, y_pred_ridge = loo_cv(ridge, X, y)

ridge.fit(X, y)
print("\n=== Ridge Regression ===")
print(f"R²: {r2_ridge:.3f} | RMSE: {rmse_ridge:.3f}")
print("Coeficienți:", ridge.coef_)
print("Intercept:", ridge.intercept_)

# ==============================
# Lasso Regression
# ==============================
lasso = Lasso(alpha=0.1)
r2_lasso, rmse_lasso, y_true_lasso, y_pred_lasso = loo_cv(lasso, X, y)

lasso.fit(X, y)
print("\n=== Lasso Regression ===")
print(f"R²: {r2_lasso:.3f} | RMSE: {rmse_lasso:.3f}")
print("Coeficienți:", lasso.coef_)
print("Intercept:", lasso.intercept_)

# ==============================
# Comparatie modele
# ==============================
results = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Lasso'],
    'R²': [r2_lin, r2_ridge, r2_lasso],
    'RMSE': [rmse_lin, rmse_ridge, rmse_lasso]
})
print("\n=== Comparatie modele (LOO-CV) ===")
print(results)

# ==============================
# Vizualizare rezultate
# ==============================
# 1️⃣ Bar + linie R² vs RMSE
fig, ax1 = plt.subplots(figsize=(8,5))

# Bara albastră pentru R²
ax1.bar(results['Model'], results['R²'], color='skyblue', label='R²')
ax1.set_ylabel('R²', color='blue')
ax1.set_ylim(0, 1.1)
ax1.tick_params(axis='y', labelcolor='blue')

# Linie roșie pentru RMSE (axa secundară)
ax2 = ax1.twinx()
ax2.plot(results['Model'], results['RMSE'], color='red', marker='o', linewidth=2, label='RMSE')
ax2.set_ylabel('RMSE', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, max(results['RMSE'])*1.2)  # limite corecte

# Titlu + grid ușor
plt.title('Comparatie modele: R² vs RMSE')
ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

# Legende combinate
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.show()

# 2️⃣ Scatter Linear: predicții vs reale
plt.figure(figsize=(6,6))
plt.scatter(y_true_lin, y_pred_lin, color='blue', label='Linear Pred')
plt.plot([min(y_true_lin), max(y_true_lin)], [min(y_true_lin), max(y_true_lin)], 'r--', label='Ideal')
plt.xlabel('Valori reale')
plt.ylabel('Predicții')
plt.title('Linear Regression: Valori reale vs Predicții')
plt.legend()
plt.show()

# 3️⃣ Heatmap coeficienți
features = X.columns
coef_df = pd.DataFrame({
    'Feature': features,
    'Linear': lin_reg.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
})
coef_df = coef_df.set_index('Feature')

plt.figure(figsize=(6,4))
sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0)
plt.title('Coeficienti modele')
plt.show()

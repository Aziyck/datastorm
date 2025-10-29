import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =============================
# Date
# =============================

# --- Vizualizeaza ---
visualise = False

# --- Alegere dataset ---
# 1 - Baza
# 2 - Criza
# 3 - Optimist
# 4 - Pesimist
scenario = 4


scenario_map = {
    1: "Baza",
    2: "Criza",
    3: "Optimist",
    4: "Pesimist"
}

chosen_scenario = scenario_map[scenario]

nim = pd.read_csv(f"csv/NIM_PD_LGD_{chosen_scenario}.csv")

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
    "EA_index"
]
all_cols = cols_to_use + ["PnL_net_index"]

data = nim.dropna(subset=all_cols)
# X = data[["Rata_politica_BNM_%", "PIB_real_crestere_%", "Somaj_%", "IPC_medie_anuala_%" ]]
X = data[[
    # "Rata_politica_BNM_%", 
    "PIB_real_crestere_%", 
    # "IPC_medie_anuala_%", 
    # "Somaj_%",
    # "Curs_MDL_pe_USD", 
    # "Remitente_USD_mld", 
    # "FX_YoY_%", 
    # "Remitente_YoY_%",
    # "Randament_active_%", 
    # "Cost_depozite_%", 
    "NIM_%", 
    # "PD_%", 
    # "LGD_%", 
    # "Cost_risc_%", 
    # "EA_index"
    ]]
# X = data[cols_to_use]
y = data['PnL_net_index']

# =============================
# VIF
# =============================
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# =============================
# LOO-CV function
# =============================
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

# =============================
# Linear Regression
# =============================
lin_reg = LinearRegression()
r2_lin, rmse_lin, _, _ = loo_cv(lin_reg, X, y)

X2 = sm.add_constant(X)
model_sm = sm.OLS(y, X2).fit()

print("\n=== Regressie Liniară (LOO-CV) ===")
print(f"R²: {r2_lin:.3f} | RMSE: {rmse_lin:.3f}")
print(model_sm.summary())

# =============================
# Ridge Regression
# =============================
X_ridge = data[cols_to_use]
ridge = Ridge(alpha=1.0)
r2_ridge, rmse_ridge, _, _ = loo_cv(ridge, X_ridge, y)

ridge.fit(X_ridge, y)
print("\n=== Ridge Regression ===")
print(f"R²: {r2_ridge:.3f} | RMSE: {rmse_ridge:.3f}")
print("Coeficienți:", ridge.coef_)
print("Intercept:", ridge.intercept_)

# =============================
# Lasso Regression
# =============================
X_lasso = data[cols_to_use]
lasso = Lasso(alpha=0.1)
r2_lasso, rmse_lasso, _, _ = loo_cv(lasso, X_lasso, y)

lasso.fit(X_lasso, y)
print("\n=== Lasso Regression ===")
print(f"R²: {r2_lasso:.3f} | RMSE: {rmse_lasso:.3f}")
print("Coeficienți:", lasso.coef_)
print("Intercept:", lasso.intercept_)

# =============================
# Comparatie modele
# =============================
results = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Lasso'],
    'R²': [r2_lin, r2_ridge, r2_lasso],
    'RMSE': [rmse_lin, rmse_ridge, rmse_lasso]
})
print("\n=== Comparatie modele (LOO-CV) ===")
print(results)

# --- Salvam modelele fiecare în fișier ---
to_save_linear = {'model': lin_reg, 'X_columns': list(X.columns) }
joblib.dump(to_save_linear, f"models/{chosen_scenario}_Linear.pkl")

to_save_ridge = {'model': ridge, 'X_columns': list(X_ridge.columns)}
joblib.dump(to_save_ridge, f"models/{chosen_scenario}_Ridge.pkl")

to_save_lasso = {'model': lasso, 'X_columns': list(X_lasso.columns)}
joblib.dump(to_save_lasso, f"models/{chosen_scenario}_Lasso.pkl")

# ==============================
# Vizualizare rezultate
# ==============================

if (visualise):

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
    # Predicții pentru Linear folosind modelul antrenat pe tot datasetul
    # ==============================
    # Scatter plot: predicții vs valori reale pentru toate modelele
    # ==============================

    # Predicții pe tot datasetul
    y_pred_lin = lin_reg.predict(X)
    y_pred_ridge = ridge.predict(X_ridge)
    y_pred_lasso = lasso.predict(X_lasso)

    plt.figure(figsize=(7,7))

    # Linear
    plt.scatter(y, y_pred_lin, color='blue', label='Linear', alpha=0.7)

    # Ridge
    plt.scatter(y, y_pred_ridge, color='green', label='Ridge', alpha=0.7)

    # Lasso
    plt.scatter(y, y_pred_lasso, color='orange', label='Lasso', alpha=0.7)

    # Linia ideală
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal', linewidth=2)

    plt.xlabel('Valori reale')
    plt.ylabel('Predicții')
    plt.title('Comparatie predicții vs valori reale: Linear, Ridge, Lasso')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # 3️⃣ Heatmap coeficienți modele
    # Folosim toate coloanele relevante
    features = cols_to_use  # sau X.columns dacă ești pe subset

    # Construim DataFrame cu coeficienți, completând cu 0 acolo unde modelul nu are coeficient
    coef_df = pd.DataFrame({
        'Feature': features,
        'Linear': [lin_reg.coef_[list(X.columns).index(f)] if f in X.columns else np.nan for f in features],
        'Ridge': [ridge.coef_[list(X_ridge.columns).index(f)] if f in X_ridge.columns else 0 for f in features],
        'Lasso': [lasso.coef_[list(X_lasso.columns).index(f)] if f in X_lasso.columns else 0 for f in features]
    })

    coef_df = coef_df.set_index('Feature')

    plt.figure(figsize=(8,6))
    sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Coeficienti modele: Linear, Ridge, Lasso')
    plt.show()
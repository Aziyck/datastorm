import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split


nim = pd.read_csv("csv/NIM_PD_LGD_Pesimist.csv")

#Ce coloane vom folosi
# cols_to_use = [
#     "Rata_politica_BNM_%", 
#     "PIB_real_crestere_%", 
#     "IPC_medie_anuala_%", 
#     "Somaj_%",
#     "Curs_MDL_pe_USD", 
#     "Remitente_USD_mld", 
#     "FX_YoY_%", 
#     "Remitente_YoY_%",
#     "Randament_active_%", 
#     "Cost_depozite_%", 
#     "NIM_%", 
#     "PD_%", 
#     "LGD_%", 
#     "Cost_risc_%", 
#     "EA_index", 
#     "NII_index", 
#     "LLP_index"]
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
    "EA_index"]
all_cols = cols_to_use + ["PnL_net_index"]

#Pregatim modelul
data = nim.dropna(subset=all_cols)

# =============================
# 2. Explorare vizuala
# =============================
# for col in ['Rata_politica_BNM_%', 'PIB_real_crestere_%', 'IPC_medie_anuala_%', 'NIM_%', 'PD_%', "Cost_depozite_%", "Randament_active_%"]:
#     sns.scatterplot(data=data, x=col, y='PnL_net_index')
#     plt.title(f'Relatia dintre {col} si PnL_net_index')
#     plt.show()


# =============================
# 3. Împărțim datele în train/test
# =============================
X = data[["Rata_politica_BNM_%", "PIB_real_crestere_%", "NIM_%"]]
y = data['PnL_net_index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 3. VIF
# VIF arată de câte ori varianța unui coeficient de regresie este mărită din cauza corelării cu alte variabile independente.
# Mai simplu spus:
# 👉 dacă două sau mai multe variabile din model sunt foarte corelate între ele, VIF va fi mare → modelul devine instabil, iar coeficienții pot fi distorsionați.
#
# Interval VIF	Interpretare	     Recomandare
# 1 – 2	        Corelație mică	        OK
# 2 – 5	        Corelație moderată	    Poate fi acceptabilă
# > 5	        Corelație mare	        Posibilă multicoliniaritate
# > 10	        Foarte mare	            Trebuie acționat (elimină variabila sau transformă modelul)
# =============================
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# =============================
# 4. Regressie liniară simplă
# =============================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

r2_lin = r2_score(y_test, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))

X2 = sm.add_constant(X)
model = sm.OLS(y, X2).fit()

print("Regresie Liniară:")
print(f"R²: {r2_lin:.3f} | RMSE: {rmse_lin:.3f}")
print(model.summary())



# =============================
# 5. Regressie Polinomială (grad 2)
# =============================
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly_train = poly.fit_transform(X_train)
# X_poly_test = poly.transform(X_test)

# poly_reg = LinearRegression()
# poly_reg.fit(X_poly_train, y_train)
# y_pred_poly = poly_reg.predict(X_poly_test)

# r2_poly = r2_score(y_test, y_pred_poly)
# rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))


# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X)
# X_poly2 = sm.add_constant(X_poly)

# model_poly = sm.OLS(y, X_poly2).fit()

# print("\nRegresie Polinomială (grad 2):")
# print(f"R²: {r2_poly:.3f} | RMSE: {rmse_poly:.3f}")
# print(model_poly.summary())


# =============================
# 6. Ridge Regression
# =============================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print("\nRidge Regression:")
print(f"R²: {r2_ridge:.3f} | RMSE: {rmse_ridge:.3f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("Ridge coefficients:", ridge.coef_)
print("Intercept:", ridge.intercept_)


# =============================
# 7. Lasso Regression
# =============================
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print("\nLasso Regression:")
print(f"R²: {r2_lasso:.3f} | RMSE: {rmse_lasso:.3f}")

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print("Lasso coefficients:", lasso.coef_)
print("Intercept:", lasso.intercept_)


# =============================
# 8. Comparație între modele
#
# R² (R-pătrat, Coeficient de determinare)
# Ce măsoară:
# Cât de bine explică modelul variația variabilei dependente (y) folosind variabilele explicative (X).
# Cu alte cuvinte: "Ce procent din variația lui PnL_net_index e explicat de model?"
#
# RMSE (Root Mean Squared Error)
# Ce măsoară:
# Cât de mare este eroarea medie dintre predicțiile modelului și valorile reale.
# Este o măsură a deviației absolute în unitățile variabilei țintă.
# =============================
results = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Lasso'],
    'R²': [r2_lin, r2_ridge, r2_lasso],
    'RMSE': [rmse_lin, rmse_ridge, rmse_lasso]
})

print("\n=== Comparatie modele ===")
print(results)


# # =============================
# # 9. Vizualizare rezultate
# # =============================

# # 1️⃣ Bar + linie R² vs RMSE
# fig, ax1 = plt.subplots(figsize=(8,5))

# # Bara albastră pentru R²
# ax1.bar(results['Model'], results['R²'], color='skyblue', label='R²')
# ax1.set_ylabel('R²', color='blue')
# ax1.set_ylim(0, 1.1)
# ax1.tick_params(axis='y', labelcolor='blue')

# # Linie roșie pentru RMSE (axa secundară)
# ax2 = ax1.twinx()
# ax2.plot(results['Model'], results['RMSE'], color='red', marker='o', linewidth=2, label='RMSE')
# ax2.set_ylabel('RMSE', color='red')
# ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_ylim(0, max(results['RMSE'])*1.2)  # limite corecte

# # Titlu + grid ușor
# plt.title('Comparatie modele: R² vs RMSE')
# ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

# # Legende combinate
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# plt.show()



# # 3️⃣ Heatmap coeficienți
# features = X.columns
# coef_df = pd.DataFrame({
#     'Feature': features,
#     'Linear': lin_reg.coef_,
#     'Ridge': ridge.coef_,
#     'Lasso': lasso.coef_
# })
# coef_df = coef_df.set_index('Feature')

# plt.figure(figsize=(6,4))
# sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0)
# plt.title('Coeficienti modele')
# plt.show()

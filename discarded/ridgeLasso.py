from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Pregatim datele (folosim aceleasi variabile numerice)
cols_to_use = [
    "Rata_politica_BNM_%", "PIB_real_crestere_%", "IPC_medie_anuala_%",
    "Somaj_%", "Curs_MDL_pe_USD", "Remitente_USD_mld", "FX_YoY_%",
    "Remitente_YoY_%", "Randament_active_%", "Cost_depozite_%", "NIM_%",
    "PD_%", "LGD_%", "Cost_risc_%", "EA_index", "NII_index", "LLP_index"
]

nim = pd.read_csv("csv/NIM_PD_LGD.csv")

data = nim.dropna(subset=cols_to_use + ["PnL_net_index"])
X = data[cols_to_use].values
y = data["PnL_net_index"].values

# Standardizam variabilele (important pentru Ridge/Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Impartim datele in train/test (optional)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ridge
ridge = Ridge(alpha=1.0)  # alpha = penalizare
ridge.fit(X_train, y_train)
print("Ridge coef:", ridge.coef_)
print("Ridge R^2:", ridge.score(X_test, y_test))

# Lasso
lasso = Lasso(alpha=0.1)  # alpha = penalizare (mare = mai multe coef = 0)
lasso.fit(X_train, y_train)
print("Lasso coef:", lasso.coef_)
print("Lasso R^2:", lasso.score(X_test, y_test))


plt.figure(figsize=(15,6))
plt.bar(np.arange(len(cols_to_use)) - 0.2, ridge.coef_, width=0.4, label='Ridge')
plt.bar(np.arange(len(cols_to_use)) + 0.2, lasso.coef_, width=0.4, label='Lasso')
plt.xticks(np.arange(len(cols_to_use)), cols_to_use, rotation=90)
plt.ylabel("Coeficienti")
plt.title("Comparatie coeficienti Ridge vs Lasso")
plt.legend()
plt.show()
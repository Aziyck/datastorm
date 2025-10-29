# Create a small dataset
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

date_baza = pd.read_csv("csv/Date_Baza.csv")
param_scen = pd.read_csv("csv/Parametri.csv")
rez_scen = pd.read_csv("csv/Rezultate.csv")
param_model = pd.read_csv("csv/Parametri_Model.csv")
nim = pd.read_csv("csv/NIM_PD_LGD_Baza.csv")
pnl = pd.read_csv("csv/PNL_Punti.csv")

#Ce coloane vom folosi
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
    "LLP_index"]
all_cols = cols_to_use + ["PnL_net_index"]

#Pregatim modelul
data = nim.dropna(subset=all_cols)
# X = data[cols_to_use]
# X = data[["Rata_politica_BNM_%","PIB_real_crestere_%", "IPC_medie_anuala_%", "NII_index", "PD_%"]]
X = data[cols_to_use]
y = data["PnL_net_index"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

#Pregatim graficul de corelatii
numeric_data = data[all_cols]
corr = numeric_data.corr().abs()


    #Aratam graficul de corelatii la consola
# print("=== Matricea de corelații ===")
# print(corr.round(2).to_string())  

    #Aratam graficul de corelatii la ecran
sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest")
plt.show()





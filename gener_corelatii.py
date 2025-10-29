import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lista scenariilor
scenarios = ["Baza", "Optimist", "Pesimist", "Criza"]

# Coloanele folosite
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

# Stil global
sns.set(style="white", font_scale=0.8)

# Parcurgem toate scenariile
for scen in scenarios:
    file_path = f"csv/NIM_PD_LGD_{scen}.csv"
    
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=all_cols)
        corr = df[all_cols].corr().abs()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", square=True)
        plt.title(f"Heatmap Corelații - Scenariul {scen}")
        plt.tight_layout()
        plt.show()

        # Arată toate coloanele și toate rândurile
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 200)  # lățime mai mare, să nu mai dea wrap
        pd.set_option('display.float_format', '{:.2f}'.format)  # pentru 2 zecimale

        # Afișare tabel corelații în consolă
        print(f"Corelatii - Scenariul {scen}")
        print(corr)

    except FileNotFoundError:
        print(f"Fișierul {file_path} nu a fost găsit.")
    except Exception as e:
        print(f"Eroare la procesarea {file_path}: {e}")

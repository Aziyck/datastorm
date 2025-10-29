import pandas as pd
import numpy as np

# --- Seed pentru reproductibilitate ---
np.random.seed(69)

# --- Alegere dataset ---
# 1 - Baza
# 2 - Criza
# 3 - Optimist
# 4 - Pesimist
dataset_choice = 2  # 🔹 Schimbă aici setul de date


# --- Date istorice ---
if dataset_choice == 1:  # Baza
    data = pd.DataFrame({
        'An': [2019,2020,2021,2022,2023,2024],
        'Rata_politica_BNM_%': [6.5,5.5,2.6,20,7,4.25],
        'PIB_real_crestere_%': [3.6,-8.3,13.9,-4.6,1.2,0.1],
        'IPC_medie_anuala_%': [4.8,3.8,5.1,28.7,13.4,4.7],
        'Somaj_%': [1.5,1.2,0.8,0.9,1.6,1.4],
        'Curs_MDL_pe_USD': [17.6,17.3,17.7,18.9,18.2,17.8],
        'Remitente_USD_mld': [1.2,1.4,1.6,2.1,2.3,2.0],
        'FX_YoY_%': [0,-1.7,2.31,6.78,-3.7,-2.2],
        'EA_index': [102.9,96,106.7,102.8,103.8,103.9],
        'Randament_active_%': [7.9,7.3,5.56,16,8.2,6.55],
        'Cost_depozite_%': [3.05,2.02,0.03,11.88,3.21,1.74],
        'NIM_%': [4.85,5.28,5.53,4.12,4.99,4.81],
        'PD_%': [3,4.58,1.35,5.52,3.67,3.4],
        'LGD_%': [35,34.97,34.92,35.51,35.1,34.95]
    })
elif dataset_choice == 2:  # Criza
    data = pd.DataFrame({
        'An': [2019, 2020, 2021, 2022, 2023, 2024],
        'Rata_politica_BNM_%': [10.5, 9.5, 6.6, 24, 11, 8.25],
        'PIB_real_crestere_%': [-1.4, -13.3, 8.9, -9.6, -3.8, -4.9],
        'IPC_medie_anuala_%': [10.8, 9.8, 11.1, 34.7, 19.4, 10.7],
        'Somaj_%': [4.5, 4.2, 3.8, 3.9, 4.6, 4.4],
        'Curs_MDL_pe_USD': [20.24, 19.89, 20.35, 21.73, 20.93, 20.47],
        'Remitente_USD_mld': [0.96, 1.12, 1.28, 1.68, 1.84, 1.6],
        'FX_YoY_%': [0, -1.7, 2.31, 6.78, -3.7, -2.2],
        'Remitente_YoY_%': [0, 16.67, 14.29, 31.25, 9.52, -13.04],
        'Randament_active_%': [10.3, 9.7, 7.96, 18.4, 10.6, 8.95],
        'Cost_depozite_%': [5.85, 4.82, 2.83, 14.67, 6.01, 4.54],
        'NIM_%': [4.45, 4.88, 5.13, 3.73, 4.59, 4.41],
        'PD_%': [3, 4.58, 1.35, 5.52, 3.67, 3.4],
        'LGD_%': [35, 34.97, 34.92, 35.51, 35.1, 34.95],
        'Cost_risc_%': [1.05, 1.6, 0.47, 1.96, 1.29, 1.19],
        'EA_index': [98.9, 88.4, 94.7, 87.4, 84.7, 81.4],
        'NII_index': [44, 43.1, 48.5, 32.5, 38.9, 35.9],
        'LLP_index': [10.4, 14.1, 4.5, 17.1, 10.9, 9.7],
        'PnL_net_index': [33.6, 29, 44, 15.4, 28, 26.3]
    })
elif dataset_choice == 3:  # Optimist
    data = pd.DataFrame({
        'An': [2019, 2020, 2021, 2022, 2023, 2024],
        'Rata_politica_BNM_%': [5.5, 4.5, 1.6, 19, 6, 3.25],
        'PIB_real_crestere_%': [5.6, -6.3, 15.9, -2.6, 3.2, 2.1],
        'IPC_medie_anuala_%': [2.8, 1.8, 3.1, 26.7, 11.4, 2.7],
        'Somaj_%': [1, 0.7, 0.3, 0.4, 1.1, 0.9],
        'Curs_MDL_pe_USD': [17.07, 16.78, 17.17, 18.33, 17.65, 17.27],
        'Remitente_USD_mld': [1.32, 1.54, 1.76, 2.31, 2.53, 2.2],
        'FX_YoY_%': [0, -1.7, 2.31, 6.78, -3.7, -2.2],
        'Remitente_YoY_%': [0, 16.67, 14.29, 31.25, 9.52, -13.04],
        'Randament_active_%': [7.3, 6.7, 4.96, 15.4, 7.6, 5.95],
        'Cost_depozite_%': [2.35, 1.32, 0, 11.18, 2.51, 1.04],
        'NIM_%': [4.95, 5.38, 4.96, 4.23, 5.09, 4.91],
        'PD_%': [3, 4.58, 1.35, 5.52, 3.67, 3.4],
        'LGD_%': [35, 34.97, 34.92, 35.51, 35.1, 34.95],
        'Cost_risc_%': [1.05, 1.6, 0.47, 1.96, 1.29, 1.19],
        'EA_index': [104.5, 99.2, 111.8, 109.5, 112.3, 114.2],
        'NII_index': [51.7, 53.4, 55.5, 46.3, 57.2, 56.1],
        'LLP_index': [11, 15.9, 5.3, 21.4, 14.5, 13.6],
        'PnL_net_index': [40.7, 37.5, 50.2, 24.8, 42.7, 42.5]
    })
elif dataset_choice == 4:  # Pesimist
    data = pd.DataFrame({
        'An': [2019, 2020, 2021, 2022, 2023, 2024],
        'Rata_politica_BNM_%': [8, 7, 4.1, 21.5, 8.5, 5.75],
        'PIB_real_crestere_%': [1.6, -10.3, 11.9, -6.6, -0.8, -1.9],
        'IPC_medie_anuala_%': [7.3, 6.3, 7.6, 31.2, 15.9, 7.2],
        'Somaj_%': [2.5, 2.2, 1.8, 1.9, 2.6, 2.4],
        'Curs_MDL_pe_USD': [18.83, 18.51, 18.94, 20.22, 19.47, 19.05],
        'Remitente_USD_mld': [1.12, 1.3, 1.49, 1.95, 2.14, 1.86],
        'FX_YoY_%': [0, -1.7, 2.31, 6.78, -3.7, -2.2],
        'Remitente_YoY_%': [0, 16.67, 14.29, 31.25, 9.52, -13.04],
        'Randament_active_%': [8.8, 8.2, 6.46, 16.9, 9.1, 7.45],
        'Cost_depozite_%': [4.1, 3.07, 1.08, 12.92, 4.26, 2.79],
        'NIM_%': [4.7, 5.13, 5.38, 3.97, 4.84, 4.66],
        'PD_%': [3, 4.58, 1.35, 5.52, 3.67, 3.4],
        'LGD_%': [35, 34.97, 34.92, 35.51, 35.1, 34.95],
        'Cost_risc_%': [1.05, 1.6, 0.47, 1.96, 1.29, 1.19],
        'EA_index': [101.3, 92.9, 101.8, 96.4, 95.8, 94.3],
        'NII_index': [47.6, 47.7, 54.7, 38.3, 46.4, 44],
        'LLP_index': [10.6, 14.9, 4.8, 18.9, 12.3, 11.2],
        'PnL_net_index': [37, 32.8, 49.9, 19.4, 34, 32.8]
    })
else:
    raise ValueError("dataset_choice trebuie sa fie 1-4")

# --- Mapare dataset_choice -> nume scenariu ---
scenario_map = {
    1: "Baza",
    2: "Criza",
    3: "Optimist",
    4: "Pesimist"
}

chosen_scenario = scenario_map[dataset_choice]

# --- Parametri ---
params = {
    'alpha_passthrough_activ': 0.6,
    'beta_passthrough_depozite': 0.7,
    'spread_structural_activ_pp': 4,
    'spread_structural_depozite_pp': -1.5,
    'gamma_dep_remit_pp_perc': 0.02,
    'nivel_PD_baza_%': 3,
    'sens_PD_PIB_pp_perc': -0.15,
    'sens_PD_IPC_pp_perc': 0.05,
    'sens_PD_SOMAJ_pp_perc': 0.3,
    'sens_PD_FX_pp_percY': 0.04,
    'lim_PD_min_%': 0.2,
    'lim_PD_max_%': 12,
    'nivel_LGD_baza_%': 35,
    'sens_LGD_rate_pp_perc': 0.02,
    'sens_LGD_IPC_pp_perc': 0.01,
    'lim_LGD_min_%': 20,
    'lim_LGD_max_%': 70,
    'EA_index_start_2019': 100,
    'gamma_cres_EA_perc_of_PIB': 0.8
}

# --- Functie AR(1) generala ---
def ar1(value, phi=0.7, sigma=1.0, drift=0.0, min_val=None, max_val=None):
    """AR(1) cu drift si zgomot, optional limitare min/max."""
    new_val = phi * value + (1 - phi) * value + drift + np.random.normal(0, sigma)
    if min_val is not None:
        new_val = max(min_val, new_val)
    if max_val is not None:
        new_val = min(max_val, new_val)
    return new_val

# --- Configurare perioada ---
years_future = list(range(2025, 2038))
last_row = data.iloc[-1]

# --- Parametrii sigma pentru volatilitate ---
sigma_map = {
    'Rata_politica_BNM_%': 1.5,
    'PIB_real_crestere_%': 3.0,
    'IPC_medie_anuala_%': 2.0,
    'Somaj_%': 0.3,
    'Curs_MDL_pe_USD': 0.15,
    'Remitente_USD_mld': 0.2,
    'FX_YoY_%': 1.0
}

# --- Generare valori viitoare ---
future_data = []
EA_index_prev = last_row['EA_index']
Remitente_prev = last_row['Remitente_USD_mld']
Somaj_prev = last_row['Somaj_%']

for year in years_future:
    # --- Macro autoregresive ---
    Rata_BNM = ar1(last_row['Rata_politica_BNM_%'], phi=0.7, sigma=sigma_map['Rata_politica_BNM_%'], drift=0)
    PIB = ar1(last_row['PIB_real_crestere_%'], phi=0.6, sigma=sigma_map['PIB_real_crestere_%'], drift=0)
    IPC = ar1(last_row['IPC_medie_anuala_%'], phi=0.6, sigma=sigma_map['IPC_medie_anuala_%'], drift=0)
    Somaj = ar1(Somaj_prev, phi=0.85, sigma=sigma_map['Somaj_%'], drift=0.05, min_val=0.5, max_val=10)
    Curs = ar1(last_row['Curs_MDL_pe_USD'], phi=0.8, sigma=sigma_map['Curs_MDL_pe_USD'])
    Remitente = ar1(last_row['Remitente_USD_mld'], phi=0.8, sigma=sigma_map['Remitente_USD_mld'], drift=0)
    FX_YoY = ar1(last_row['FX_YoY_%'], phi=0.5, sigma=sigma_map['FX_YoY_%'])

    # --- Calcul Remitente YoY ---
    Remitente_YoY = (Remitente - Remitente_prev) / Remitente_prev * 100
    Remitente_prev = Remitente

    # --- Randament active si Cost depozite ---
    Randament_active = Rata_BNM * params['alpha_passthrough_activ'] + params['spread_structural_activ_pp']
    Cost_depozite = Rata_BNM * params['beta_passthrough_depozite'] + params['spread_structural_depozite_pp'] + params['gamma_dep_remit_pp_perc'] * Remitente_YoY
    NIM = Randament_active - Cost_depozite

    # --- PD ---
    PD = params['nivel_PD_baza_%'] + params['sens_PD_PIB_pp_perc'] * PIB + params['sens_PD_IPC_pp_perc'] * IPC + params['sens_PD_SOMAJ_pp_perc'] * Somaj + params['sens_PD_FX_pp_percY'] * FX_YoY
    PD = np.clip(PD, params['lim_PD_min_%'], params['lim_PD_max_%'])

    # --- LGD ---
    LGD = params['nivel_LGD_baza_%'] + params['sens_LGD_rate_pp_perc'] * Rata_BNM + params['sens_LGD_IPC_pp_perc'] * IPC
    LGD = np.clip(LGD, params['lim_LGD_min_%'], params['lim_LGD_max_%'])

    # --- EA_index cu AR(1) ---
    EA_min = 50    # prag minim realist pentru EA_index
    EA_max = 150   # prag maxim realist pentru EA_index
    drift_EA = params['gamma_cres_EA_perc_of_PIB'] * PIB / 100
    EA_index = ar1(EA_index_prev, phi=0.75, sigma=3.0, drift=drift_EA, min_val=EA_min, max_val=EA_max)
    EA_index_prev = EA_index

    # --- Indici financiari ---
    NII_index = NIM * EA_index / 10
    LLP_index = PD / 100 * LGD * EA_index / 10
    PnL_net_index = NII_index - LLP_index

    # --- Salvare valori ---
    future_data.append({
        'An': year,
        'Scenariu': chosen_scenario,
        'Rata_politica_BNM_%': round(Rata_BNM, 2),
        'PIB_real_crestere_%': round(PIB, 2),
        'IPC_medie_anuala_%': round(IPC, 2),
        'Somaj_%': round(Somaj, 2),
        'Curs_MDL_pe_USD': round(Curs, 2),
        'Remitente_USD_mld': round(Remitente, 2),
        'FX_YoY_%': round(FX_YoY, 2),
        'Remitente_YoY_%': round(Remitente_YoY, 2),
        'Randament_active_%': round(Randament_active, 2),
        'Cost_depozite_%': round(Cost_depozite, 2),
        'NIM_%': round(NIM, 2),
        'PD_%': round(PD, 2),
        'LGD_%': round(LGD, 2),
        'Cost_risc_%': round(PD*LGD/100, 2),
        'EA_index': round(EA_index, 2),
        'NII_index': round(NII_index, 2),
        'LLP_index': round(LLP_index, 2),
        'PnL_net_index': round(PnL_net_index, 2)
    })

    # --- Actualizare last_row ---
    last_row = pd.Series(future_data[-1])
    Somaj_prev = Somaj

# --- Creare DataFrame ---
future_df = pd.DataFrame(future_data)
future_df.to_csv("csv/gen/Date_Generate_2025_2037.csv", index=False)
print(future_df.head(10))
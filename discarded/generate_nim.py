import pandas as pd
import numpy as np

# === Citim datele ===
macro = pd.read_csv("Rezultate_Macro.csv")
params = pd.read_csv("csv/Parametri_Model.csv", index_col=0)

# === Parametrii ===
p = params['Valoare'].to_dict()

# === Asigurare nume coloane coerente ===
# macro.columns = macro.columns.str.replace('̦', 's')
# macro.columns = macro.columns.str.replace('̆', '')
# macro.columns = macro.columns.str.replace('́', '')
# macro.columns = macro.columns.str.replace('̀', '')
# macro.columns = [c.strip().replace(" ", "_") for c in macro.columns]

# === Inițializare coloane derivate ===
for col in ['FX_YoY_%', 'Remitente_YoY_%', 'Randament_active_%', 'Cost_depozite_%',
            'NIM_%', 'PD_%', 'LGD_%', 'Cost_risc_%', 'EA_index',
            'NII_index', 'LLP_index', 'PnL_net_index']:
    macro[col] = np.nan

# === Sortăm și calculăm YoY ===
macro = macro.sort_values(by=['Scenariu', 'An'])
for scen in macro['Scenariu'].unique():
    df = macro[macro['Scenariu'] == scen]
    macro.loc[df.index, 'FX_YoY_%'] = df['Curs_MDL_pe_USD'].pct_change() * 100
    macro.loc[df.index, 'Remitente_YoY_%'] = df['Remitente_USD_mld'].pct_change() * 100

# === Calcul Randament active (yield on assets) ===
macro['Randament_active_%'] = (
    macro['Rata_politica_BNM_%'] * p['alpha_passthrough_activ']
    + p['spread_structural_activ_pp']
    + 0.1 * macro['PIB_real_crestere_%']  # creșterea economică stimulează randamentele
)
macro['Randament_active_%'] = macro['Randament_active_%'].clip(lower=4, upper=18)

# === Cost depozite ===
macro['Cost_depozite_%'] = (
    macro['Rata_politica_BNM_%'] * p['beta_passthrough_depozite']
    + p['spread_structural_depozite_pp']
    + p['gamma_dep_remit_pp_perc'] * macro['Remitente_YoY_%'] / 100
)
macro['Cost_depozite_%'] = macro['Cost_depozite_%'].clip(lower=0.5, upper=12)

# === NIM ===
macro['NIM_%'] = (macro['Randament_active_%'] - macro['Cost_depozite_%']).clip(lower=3, upper=6)

# === Probabilitate de default (PD) ===
macro['PD_%'] = (
    p['nivel_PD_baza_%']
    + p['sens_PD_PIB_pp_perc'] * macro['PIB_real_crestere_%']
    + p['sens_PD_IPC_pp_perc'] * macro['IPC_medie_anuala_%']
    + p['sens_PD_SOMAJ_pp_perc'] * macro['Somaj_%']
    + p['sens_PD_FX_pp_percY'] * macro['FX_YoY_%']
)
macro['PD_%'] = macro['PD_%'].clip(lower=p['lim_PD_min_%'], upper=p['lim_PD_max_%'])

# === LGD ===
macro['LGD_%'] = (
    p['nivel_LGD_baza_%']
    + p['sens_LGD_rate_pp_perc'] * macro['Rata_politica_BNM_%']
    + p['sens_LGD_IPC_pp_perc'] * macro['IPC_medie_anuala_%']
)
macro['LGD_%'] = macro['LGD_%'].clip(lower=p['lim_LGD_min_%'], upper=p['lim_LGD_max_%'])

# === Cost risc ===
macro['Cost_risc_%'] = (macro['PD_%'] * macro['LGD_%'] / 100).clip(0.5, 3)

# === EA index (active totale) ===
macro['EA_index'] = (
    p['EA_index_start_2019'] *
    (1 + p['gamma_cres_EA_perc_of_PIB'] * macro['PIB_real_crestere_%'] / 100)
)
macro['EA_index'] = macro['EA_index'].clip(lower=90, upper=115)

# === NII / LLP / PnL ===
macro['NII_index'] = (macro['NIM_%'] * macro['EA_index'])
macro['LLP_index'] = (macro['Cost_risc_%'] * macro['EA_index'] / 10).clip(5, 25)
macro['PnL_net_index'] = (macro['NII_index'] - macro['LLP_index']).clip(20, 55)

# === Salvare finală ===
macro.to_csv("csv/NIM_PD_LGD_generated.csv", index=False)
print("✅ Tabel generat cu succes!")
print(macro[['An', 'Scenariu', 'Rata_politica_BNM_%', 'Randament_active_%',
             'Cost_depozite_%', 'NIM_%', 'PD_%', 'LGD_%', 'Cost_risc_%',
             'NII_index', 'LLP_index', 'PnL_net_index']].round(2))

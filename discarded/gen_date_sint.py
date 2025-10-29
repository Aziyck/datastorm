# -*- coding: utf-8 -*-
"""
Generator AR(1) pe scenarii (optimist, pesimist, criză)
- Citește istoricul din "Tabel BAZA.xlsx" (foaia NIM_PD_LGD)
- Citește parametrii din "PARAMETRI MODEL.xlsx" (foaia Parametri_Model)
- Estimează automat phi (coef. AR1) din istoric, cu opțiune de override din parametri
- Permite drift și sigma (volatilitate a șocului) pe scenariu, cu override din parametri
- Generează variabilele macro cu AR(1) și apoi derivă variabile bancare (NIM, PD, LGD, etc.)
- Salvează într-un Excel: Date_sintetice_AR_scenarii.xlsx
"""
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------- CONFIG CĂI --------------------

BAZA_XLSX =  "new/xlsx/TABEL_BAZA.xlsx"
PARAM_XLSX = "new/xlsx/PARAMETRI_MODEL.xlsx"
OUT_XLSX = "new/xlsx/Date_sintetice_AR_scenarii.xlsx"

# -------------------- LOAD INPUT --------------------
baza = pd.read_excel(BAZA_XLSX, sheet_name="NIM_PD_LGD")
param = pd.read_excel(PARAM_XLSX, sheet_name="Parametri_Model")

# dict parametri: {cheie: valoare}
pdict = {}
for _, row in param.iterrows():
    k = str(row.get("Parametru", "")).strip()
    v = row.get("Valoare", np.nan)
    if k:
        pdict[k] = v

# -------------------- VARIABILE --------------------
SCEN_NAMES = ["optimist", "pesimist", "criză"]
start_year = int(pdict.get("ar_start_year", 2026))
horizon_years = int(pdict.get("ar_horizon_years", 12))

macro_cols = [
    "Rata_politica_BNM_%",
    "PIB_real_creștere_%",
    "IPC_medie_anuala_%",
    "Șomaj_%",
    "FX_YoY_%",
    "Remitente_YoY_%",
]

# -------------------- HELPER: nume "safe" pentru chei override --------------------
def safe_key(s: str) -> str:
    s = s.lower()
    s = (s
         .replace("%","perc")
         .replace("̦","")
         .replace(" ", "_")
         .replace("ă","a").replace("â","a").replace("î","i").replace("ș","s").replace("ş","s").replace("ț","t").replace("ţ","t")
         .replace("−","-").replace("–","-").replace("—","-"))
    while "__" in s:
        s = s.replace("__","_")
    return s

safe_names = {col: safe_key(col) for col in macro_cols}

# -------------------- ESTIMARE PHI AR(1) --------------------
def estimate_phi(series: pd.Series) -> float:
    x = series.dropna().values
    if len(x) < 3:
        return 0.5
    x_t = x[1:]
    x_lag = x[:-1]
    var = np.var(x_lag)
    if var == 0:
        return 0.5
    phi = np.cov(x_t, x_lag, bias=True)[0, 1] / var
    return float(np.clip(phi, -0.95, 0.95))

phi = {}
mu_hist = {}
sigma_hist = {}
for col in macro_cols:
    s = baza[col]
    phi[col] = estimate_phi(s)
    mu_hist[col] = float(s.dropna().mean()) if s.dropna().size else 0.0
    sigma_hist[col] = float(s.dropna().std()) if s.dropna().size else 1.0

# override phi din parametri dacă există: ex. "phi_pib_real_crestere_perc"
for col in macro_cols:
    sk = safe_names[col]
    key = f"phi_{sk}"
    if key in pdict and pd.notna(pdict[key]):
        try:
            phi[col] = float(pdict[key])
        except Exception:
            pass

# -------------------- DRIFT & SIGMA pe scenarii --------------------
default_drift = {
    "optimist": {
        "PIB_real_creștere_%": +2.0,      # conform pp_add optimist
        "IPC_medie_anuala_%": -2.0,        # optimist -> scade inflația
        "Șomaj_%": -0.5,                   # optimist -> șomaj scade
        "FX_YoY_%": -0.03,                  # pct_mult optimist
        "Remitente_YoY_%": +0.10,           # pct_mult optimist
        "Rata_politica_BNM_%": -1.0,        # pp_add optimist
    },
    "pesimist": {
        "PIB_real_creștere_%": -2.0,      # scenariu pesimist
        "IPC_medie_anuala_%": +2.5,
        "Șomaj_%": +1.0,
        "FX_YoY_%": +0.07,
        "Remitente_YoY_%": -0.07,
        "Rata_politica_BNM_%": +1.5,
    },
    "criză": {
        "PIB_real_creștere_%": -5.0,      # scenariu criză
        "IPC_medie_anuala_%": +6.0,
        "Șomaj_%": +3.0,
        "FX_YoY_%": +0.15,
        "Remitente_YoY_%": -0.20,
        "Rata_politica_BNM_%": +4.0,
    }
}
default_sigma_mult = {"optimist": 0.7, "pesimist": 1.0, "criză": 1.5}

# Permitem override din Excel:
# drift_<safe>_<scen>  (ex: drift_pib_real_crestere_pesimist = -0.6)
# sigma_<safe>_<scen>  (ex: sigma_ipc_medie_anuala_perc_criză = 1.8) -> multiplicator vs std istoric
scenario_drift = {s: {**default_drift.get(s, {})} for s in SCEN_NAMES}
scenario_sigma_mult = {s: default_sigma_mult.get(s, 1.0) for s in SCEN_NAMES}

for col in macro_cols:
    sk = safe_names[col]
    for scen in SCEN_NAMES:
        k_drift = f"drift_{sk}_{scen}"
        if k_drift in pdict and pd.notna(pdict[k_drift]):
            try:
                scenario_drift[scen][col] = float(pdict[k_drift])
            except Exception:
                pass
        k_sigma = f"sigma_{sk}_{scen}"
        if k_sigma in pdict and pd.notna(pdict[k_sigma]):
            try:
                val = float(pdict[k_sigma])
                scenario_sigma_mult[scen] = val if val > 0 else scenario_sigma_mult[scen]
            except Exception:
                pass

# -------------------- LIMITE (opțional) --------------------
def get_bounds(name: str, default_min=-np.inf, default_max=np.inf):
    key_min = None
    key_max = None
    for k in pdict.keys():
        lk = k.lower()
        if name.lower() in lk and ("min" in lk):
            key_min = k
        if name.lower() in lk and ("max" in lk):
            key_max = k
    vmin = pdict.get(key_min, default_min)
    vmax = pdict.get(key_max, default_max)
    try:
        vmin = float(vmin)
    except Exception:
        vmin = default_min
    try:
        vmax = float(vmax)
    except Exception:
        vmax = default_max
    return vmin, vmax


# -------------------- SIMULARE AR(1) MACRO --------------------
def simulate_macro_AR1(scen_name: str, years=horizon_years, start=start_year, seed=123):
    rng = np.random.default_rng(seed + hash(scen_name) % 10_000)
    
    # Start din media istorică (mai robust)
    last = {}
    for col in macro_cols:
        s = pd.to_numeric(baza[col], errors='coerce').dropna()
        last[col] = mu_hist[col] if not s.empty else 0.0
    
    rows = []
    for t in range(years):
        y = start + t
        row = {"An": y, "Scenariu": scen_name}
        for col in macro_cols:
            drift = scenario_drift.get(scen_name, {}).get(col, 0.0)
            sigma = scenario_sigma_mult.get(scen_name, 1.0) * (sigma_hist[col] if sigma_hist[col] > 0 else 1.0)
            eps = rng.normal(0.0, sigma)
            
            # AR(1)
            x_next = drift + phi[col] * last[col] + eps
            
            # Aplicare limite dacă există
            vmin, vmax = get_bounds(col, -1e9, 1e9)
            x_next = float(np.clip(x_next, vmin, vmax))
            
            row[col] = x_next
            last[col] = x_next
        rows.append(row)
    
    return pd.DataFrame(rows)


# -------------------- DERIVARE VARIABILE BANCARE --------------------
def derive_bank_vars(df: pd.DataFrame) -> pd.DataFrame:
    alpha = float(pdict.get("alpha_passthrough_activ", 0.7))
    beta  = float(pdict.get("beta_passthrough_depozite", 0.5))
    spread_act = float(pdict.get("spread_structural_activ_pp", 3.0))
    spread_dep = float(pdict.get("spread_structural_depozite_pp", 0.5))
    gamma_dep_remit = float(pdict.get("gamma_dep_remit_pp_perc", 0.02))

    pd_base = float(pdict.get("nivel_PD_baza_%", 4.0))
    pd_pib  = float(pdict.get("sens_PD_PIB_pp_perc", -0.10))
    pd_ipc  = float(pdict.get("sens_PD_IPC_pp_perc", 0.05))
    pd_som  = float(pdict.get("sens_PD_SOMAJ_pp_perc", 0.30))
    pd_fx   = float(pdict.get("sens_PD_FX_pp_percY", 0.03))
    pd_min, pd_max = float(pdict.get("lim_PD_min_%", 1.0)), float(pdict.get("lim_PD_max_%", 20.0))

    lgd_base = float(pdict.get("nivel_LGD_baza_%", 35.0))
    lgd_rate = float(pdict.get("sens_LGD_rate_pp_perc", 0.05))
    lgd_ipc  = float(pdict.get("sens_LGD_IPC_pp_perc", 0.08))
    lgd_min, lgd_max = float(pdict.get("lim_LGD_min_%", 20.0)), float(pdict.get("lim_LGD_max_%", 70.0))

    EA0 = float(pdict.get("EA_index_start_2019", 100.0))
    gamma_EA_PIB = float(pdict.get("gamma_cres_EA_perc_of_PIB", 0.2))

    out = df.copy()
    out["Randament_active_%"] = alpha * out["Rata_politica_BNM_%"] + spread_act
    out["Cost_depozite_%"] = beta * out["Rata_politica_BNM_%"] + spread_dep + gamma_dep_remit * out.get("Remitente_YoY_%", 0.0)
    out["NIM_%"] = out["Randament_active_%"] - out["Cost_depozite_%"]

    out["PD_%"] = (
        pd_base
        + pd_pib * out["PIB_real_creștere_%"]
        + pd_ipc * out["IPC_medie_anuala_%"]
        + pd_som * out["Șomaj_%"]
        + pd_fx  * out["FX_YoY_%"]
    ).clip(pd_min, pd_max)

    out["LGD_%"] = (
        lgd_base
        + lgd_rate * out["Rata_politica_BNM_%"]
        + lgd_ipc  * out["IPC_medie_anuala_%"]
    ).clip(lgd_min, lgd_max)

    # EA_index pornește din ultima valoare istorică (fallback la EA0)
    if "EA_index" in baza.columns and not baza["EA_index"].dropna().empty:
        EA_start = float(baza["EA_index"].dropna().iloc[-1])
    else:
        EA_start = EA0

    ea_vals = []
    prev = EA_start
    for _, r in out.sort_values(["An"]).iterrows():
        growth = 1.0 + (gamma_EA_PIB * r["PIB_real_creștere_%"]) / 100.0
        prev = max(0.0, prev * growth)
        ea_vals.append(prev)
    out["EA_index"] = ea_vals

    out["NII_index"] = out["EA_index"] * (out["NIM_%"] / 10.0)
    out["LLP_index"] = out["EA_index"] * (out["PD_%"]/100.0) * (out["LGD_%"]/100.0) * 2.0
    out["Cost_risc_%"] = (out["LLP_index"] / out["EA_index"]).replace([np.inf, -np.inf], np.nan) * 100.0
    out["PnL_net_index"] = out["NII_index"] - out["LLP_index"]
    return out

# -------------------- RUN & SAVE --------------------
def main():
    synth_list = []
    for scen in SCEN_NAMES:
        sim_mac = simulate_macro_AR1(scen)
        sim_full = derive_bank_vars(sim_mac)
        synth_list.append(sim_full)
    synth_all = pd.concat(synth_list, ignore_index=True)

    # Ordine coloane
    bank_cols = [
        "Randament_active_%","Cost_depozite_%","NIM_%","PD_%","LGD_%",
        "Cost_risc_%","EA_index","NII_index","LLP_index","PnL_net_index",
    ]
    ordered = ["An","Scenariu"] + macro_cols + bank_cols
    ordered = [c for c in ordered if c in synth_all.columns]
    synth_all = synth_all[ordered]

    # 🔹 Rotunjim toate valorile numerice la 2 zecimale
    synth_all = synth_all.apply(lambda x: np.round(x, 2) if np.issubdtype(x.dtype, np.number) else x)

    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        synth_all.to_excel(writer, sheet_name="TOATE", index=False)
        for scen in SCEN_NAMES:
            synth_all[synth_all["Scenariu"] == scen].to_excel(writer, sheet_name=scen.capitalize(), index=False)

    print(f"Scris: {OUT_XLSX}")

if __name__ == "__main__":
    main()

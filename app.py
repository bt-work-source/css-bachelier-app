import numpy as np
import pandas as pd
import streamlit as st
from math import sqrt
from scipy.stats import norm
from collections import deque
import re

from arch import arch_model

# ======= Globális konstansok =======
BUS_DAYS_PER_YEAR = 252
ROLL_WIN_STD      = 20         # dCSS rolling std ablak (nap)
LEVEL_MED_WIN     = 30         # L: |CSS| 30-napos mediánja
VOL_FLOOR         = 1e-6
GAMMA_FALLBACK    = 0.6        # ha γ prefix-becslés nem áll elő
VC_QUANTILE       = 95         # vol-cap kvantilis (évesített dCSS-vol alapján)

st.set_page_config(page_title="Forward CSS – Bachelier (GARCH + state-dependent σ)", layout="wide")
st.title("Forward CSS – Bachelier opcióárazó")
st.caption("Volatilitás: GARCH(1,1), t-eloszlás + state-dependent skálázás (RMED30 + γ) + vol-cap (q=95).")

# ---------- Segédek: beolvasás ----------
def parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=None, engine="python")
    date_col, css_col = None, None
    for c in df.columns:
        cl = str(c).lower()
        if date_col is None and ("date" in cl or "datum" in cl or cl == "dátum"):
            date_col = c
        if css_col is None and ("css" in cl or "spark" in cl or "spread" in cl):
            css_col = c
    if date_col is None or css_col is None:
        raise ValueError(f"Nem találom a dátum/CSS oszlopot. Oszlopok: {list(df.columns)}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "css": pd.to_numeric(
            pd.Series(df[css_col].astype(str))
              .str.replace("\xa0", "", regex=False)
              .str.replace(" ", "", regex=False)
              .str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)  # ezer elválasztó pont
              .str.replace(",", ".", regex=False),
            errors="coerce"
        )
    }).dropna().sort_values("date").reset_index(drop=True)
    out["dcss"] = out["css"].diff()
    return out

# ---------- GARCH illesztés ----------
def fit_garch_t(dcss: pd.Series):
    y = dcss.dropna().values
    if len(y) < 40:
        raise ValueError("Túl rövid idősor a GARCH-hoz (≥ 100 megfigyelés szükséges).")
    am = arch_model(y, mean="zero", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    return res

def build_F_and_sigma(df: pd.DataFrame, garch_res):
    # dCSS-hez tartozó feltételes vol (napos)
    sigma_series = pd.Series(
        garch_res.conditional_volatility,
        index=pd.to_datetime(df.loc[df["dcss"].notna(), "date"])
    ).astype(float)

    # CSS szint
    css_series = pd.Series(df["css"].values, index=pd.to_datetime(df["date"])).astype(float).sort_index()

    common_idx = css_series.index.intersection(sigma_series.index)
    F_t  = css_series.loc[common_idx]
    sig_t = sigma_series.loc[common_idx].clip(lower=VOL_FLOOR)
    return F_t, sig_t

# ---------- Vol-cap (napos σ plafon a dCSS évesített kvantilise alapján) ----------
def sigma_cap_daily_from_ann_quantile(df_dt: pd.DataFrame, q_pct: int = VC_QUANTILE) -> float:
    df_dt = df_dt.copy()
    df_dt["date"] = pd.to_datetime(df_dt["date"])
    df_dt = df_dt.set_index("date").sort_index()
    ann_vol = df_dt["dcss"].astype(float).rolling(ROLL_WIN_STD).std(ddof=1) * np.sqrt(BUS_DAYS_PER_YEAR)
    ann_vol = ann_vol.dropna()
    if ann_vol.empty:
        return np.inf
    ceil_ann = float(np.nanpercentile(ann_vol.values, int(q_pct)))
    return ceil_ann / np.sqrt(BUS_DAYS_PER_YEAR)

# ---------- γ prefix-becslés (expanding, cutoff-ig) ----------
def _winsor(s: pd.Series, p: float):
    if p <= 0 or s.dropna().empty:
        return s
    lo, hi = np.nanpercentile(s.dropna(), [p, 100-p])
    return s.clip(lower=lo, upper=hi)

def estimate_gamma_prefix(F_series: pd.Series,
                          sig_base_daily: pd.Series,
                          real_vol_daily: pd.Series,
                          L_series_full: pd.Series,
                          cutoff_date: pd.Timestamp,
                          winsor_pct: float = 1.0,
                          clip_gamma=(0.0, 2.0),
                          min_obs: int = 60) -> float:
    if cutoff_date is None:
        return np.nan
    idx_all = F_series.index.intersection(sig_base_daily.index).intersection(real_vol_daily.index).intersection(L_series_full.index)
    idx = idx_all[idx_all <= pd.to_datetime(cutoff_date)]
    if len(idx) < min_obs:
        return np.nan
    F = F_series.reindex(idx).astype(float)
    S = sig_base_daily.reindex(idx).astype(float).clip(lower=1e-10)
    R = real_vol_daily.reindex(idx).astype(float).clip(lower=1e-10)
    L = L_series_full.reindex(idx).astype(float).clip(lower=1e-8)

    base = (np.abs(F) / (np.abs(F) + L)).clip(lower=1e-8, upper=1-1e-8)
    x = np.log(base)
    y = np.log((R / S).clip(lower=1e-8))
    xw, yw = _winsor(x, winsor_pct), _winsor(y, winsor_pct)

    X = np.c_[np.ones_like(xw), xw]
    yv = yw.values
    msk = np.isfinite(X).all(axis=1) & np.isfinite(yv)
    Xf, yf = X[msk], yv[msk]
    if Xf.shape[0] < min_obs or np.std(Xf[:, 1]) < 1e-8:
        return np.nan
    try:
        beta = np.linalg.lstsq(Xf, yf, rcond=None)[0]
        gamma_hat = float(beta[1])
    except np.linalg.LinAlgError:
        return np.nan
    if clip_gamma is not None:
        gamma_hat = float(np.clip(gamma_hat, clip_gamma[0], clip_gamma[1]))
    return gamma_hat

# ---------- State-dependent évesített vol (RMED30 + γ + vol-cap) ----------
def build_state_dep_ann_vol(F_t: pd.Series, sig_t: pd.Series, CSS_all: pd.Series,
                            cutoff: pd.Timestamp, vc_quantile: int,
                            df_for_cap_two_cols: pd.DataFrame,
                            med_win: int = LEVEL_MED_WIN) -> pd.Series:
    if F_t.empty or sig_t.empty:
        return pd.Series(dtype=float)
    idx_all = F_t.index.intersection(sig_t.index)
    idx_all = idx_all[idx_all <= cutoff]
    if len(idx_all) == 0:
        return pd.Series(dtype=float)

    F = F_t.reindex(idx_all).astype(float)
    S = sig_t.reindex(idx_all).astype(float).clip(lower=VOL_FLOOR)

    L_ser_full = CSS_all.abs().rolling(int(med_win)).median().bfill().fillna(CSS_all.abs().median())
    L_ser_full = L_ser_full.reindex(idx_all, method="ffill").astype(float)

    dcss = CSS_all.diff()
    real_vol_daily = dcss.rolling(ROLL_WIN_STD).std(ddof=1) / np.sqrt(BUS_DAYS_PER_YEAR)
    real_vol_daily = real_vol_daily.reindex(idx_all)

    gamma_hat = estimate_gamma_prefix(
        F_series=F, sig_base_daily=S, real_vol_daily=real_vol_daily,
        L_series_full=L_ser_full, cutoff_date=cutoff,
        winsor_pct=1.0, clip_gamma=(0.0, 2.0), min_obs=60
    )
    if not np.isfinite(gamma_hat):
        gamma_hat = GAMMA_FALLBACK

    base = (np.abs(F) / (np.abs(F) + L_ser_full)).clip(lower=1e-8, upper=1-1e-8)
    factor = base.pow(float(gamma_hat))

    cap_d = sigma_cap_daily_from_ann_quantile(df_for_cap_two_cols[["date", "dcss"]].copy(), q_pct=int(vc_quantile))
    S_cap = np.minimum(S, cap_d) if np.isfinite(cap_d) else S
    sigma_eff_daily = np.maximum(VOL_FLOOR, S_cap * factor)
    sigma_eff_ann = sigma_eff_daily * np.sqrt(BUS_DAYS_PER_YEAR)
    return sigma_eff_ann.rename("state_dep")

# ---------- Bachelier ár ----------
def bachelier_price(F, K, sigma_ann, T_years, call_put="call"):
    if sigma_ann <= 0 or T_years <= 0:
        intrinsic = max(F - K, 0.0) if call_put == "call" else max(K - F, 0.0)
        return float(intrinsic)
    srt = sigma_ann * sqrt(T_years)
    d = (F - K) / srt
    if call_put == "call":
        return float((F - K) * norm.cdf(d) + srt * norm.pdf(d))
    else:
        return float((K - F) * norm.cdf(-d) + srt * norm.pdf(d))

# ---------- Oldalsáv / UI ----------
with st.sidebar:
    st.header("Adatfeltöltés")
    up = st.file_uploader("CSS idősor (CSV: date, css)", type=["csv"])
    st.caption("Oszlopok: 'date', 'css'. Vessző/pontosvessző/tab és dec. vessző is támogatott.")

    st.header("Paraméterek")
    K = st.number_input("Strike / küszöb K (EUR/MWh)", value=4.0, step=0.1, format="%.4f")
    call_put = st.radio("Opció típusa", ["call", "put"], index=0, horizontal=True)

    st.markdown("---")
    val_date = st.date_input("Értékelési dátum", value=pd.Timestamp.today().date())
    maturity = st.date_input("Lejárat", value=(pd.Timestamp.today() + pd.Timedelta(days=90)).date())

if up is None:
    st.info("Tölts fel egy CSV-t a bal oldali panelen.")
else:
    try:
        # --- Beolvasás + cut ---
        df = parse_csv(up)
        series_min = pd.to_datetime(df["date"]).min().date()
        series_max = pd.to_datetime(df["date"]).max().date()
        val_date = min(max(val_date, series_min), series_max)

        if maturity <= val_date:
            st.error("A lejáratnak az értékelési dátum után kell lennie.")
            st.stop()
        T_years = max(1.0 / BUS_DAYS_PER_YEAR, (pd.Timestamp(maturity) - pd.Timestamp(val_date)).days / 365.25)

        df_cut = df.loc[df["date"] <= pd.Timestamp(val_date)].copy()
        if df_cut["dcss"].dropna().shape[0] < 100:
            st.error("Nincs elég adat az értékelési napig a GARCH illesztéshez (≥ 100 differencia szükséges).")
            st.stop()

        # --- GARCH(1,1)-t + F_t, σ_t (napos) ---
        res = fit_garch_t(df_cut["dcss"])
        F_t_full, sig_t_full = build_F_and_sigma(df_cut, res)
        if F_t_full.empty or sig_t_full.empty:
            st.error("Nem áll elő F_t / σ_t az értékelési napig.")
            st.stop()

        F_t = F_t_full
        sig_t = sig_t_full
        CSS_all = pd.Series(df_cut["css"].values, index=pd.to_datetime(df_cut["date"])).astype(float).sort_index()

        # --- State-dependent σ (annualizált) az értékelési napig ---
        sd_series = build_state_dep_ann_vol(
            F_t=F_t, sig_t=sig_t, CSS_all=CSS_all,
            cutoff=pd.Timestamp(val_date), vc_quantile=VC_QUANTILE,
            df_for_cap_two_cols=df.loc[df["date"] <= pd.Timestamp(val_date), ["date", "dcss"]].copy(),
            med_win=LEVEL_MED_WIN
        )
        if sd_series.dropna().empty:
            st.error("A state-dependent σ nem számolható az adott adaton.")
            st.stop()

        sigma_ann = float(sd_series.dropna().iloc[-1])
        F_now = float(F_t.iloc[-1])

        # --- Opcióár ---
        price = bachelier_price(F_now, float(K), sigma_ann, float(T_years), call_put)
        intrinsic = max(F_now - float(K), 0.0) if call_put == "call" else max(float(K) - F_now, 0.0)
        time_value = max(0.0, price - intrinsic)

        st.subheader("Eredmény (Bachelier, σ = GARCH + state-dependent)")
        st.markdown(
            f"**CSS (F):** `{F_now:.4f}` EUR/MWh  |  "
            f"**σ (annualizált, state-dep):** `{sigma_ann:.6f}`  |  "
            f"**T (év):** `{T_years:.4f}`  |  "
            f"**K:** `{float(K):.4f}`  \n\n"
            f"**Opció prémium:** `{price:.6f}`  |  "
            f"**Belső érték:** `{intrinsic:.6f}`  |  "
            f"**Időérték:** `{time_value:.6f}`"
        )

        # --- Gyors vizualizációk ---
        import matplotlib.pyplot as plt

        st.markdown("---")
        st.subheader("Vizualizációk")

        # 1) State-dependent σ vs. hagyományos évesített vol (csak interpretációhoz)
        ann_vol_series = (df_cut["dcss"].rolling(ROLL_WIN_STD).std(ddof=1) * np.sqrt(BUS_DAYS_PER_YEAR)).dropna()
        ann_vol_series.index = pd.to_datetime(df_cut.loc[ann_vol_series.index, "date"])
        sd_series = sd_series.dropna()

        figv, axv = plt.subplots(figsize=(11, 3.6))
        if not sd_series.empty:
            axv.plot(sd_series.index, sd_series.values, lw=1.8, label="State-dep σ (évesített)")
        if not ann_vol_series.empty:
            axv.plot(ann_vol_series.index, ann_vol_series.values, lw=1.2, ls="--", label=f"Hagyományos σ (dCSS, {ROLL_WIN_STD}n)")
        axv.set_title("Évesített volatilitás – értékelési napig")
        axv.set_ylabel("σ (EUR/MWh)"); axv.set_xlabel("Dátum"); axv.grid(True, alpha=0.35); axv.legend(loc="upper left")
        st.pyplot(figv, clear_figure=True)

        # 2) Időérték T függvényében (a kapott σ mellett)
        T_grid = np.linspace(1e-4, max(T_years, 1e-3), 60)
        tv_curve = [max(0.0, bachelier_price(F_now, float(K), sigma_ann, t, call_put) - intrinsic) for t in T_grid]
        fig_tv, ax_tv = plt.subplots(figsize=(11, 3.0))
        ax_tv.plot(T_grid, tv_curve, lw=1.8)
        ax_tv.set_title("Opció időértéke a futamidő függvényében (Bachelier)")
        ax_tv.set_xlabel("T (év)"); ax_tv.set_ylabel("EUR/MWh"); ax_tv.grid(True, alpha=0.35)
        st.pyplot(fig_tv, clear_figure=True)

        # 3) Belső érték F függvényében (aktuális K mellett)
        span = max(5.0, sigma_ann * sqrt(max(T_years, 1e-4)) * 6.0)
        F_grid = np.linspace(float(K) - span, float(K) + span, 200)
        intrinsic_curve = np.maximum(F_grid - float(K), 0.0) if call_put == "call" else np.maximum(float(K) - F_grid, 0.0)
        fig_intr, ax_intr = plt.subplots(figsize=(11, 3.0))
        ax_intr.plot(F_grid, intrinsic_curve, lw=1.8, label="Belső érték")
        ax_intr.axvline(float(K), color="gray", ls="--", lw=1.0, label="Strike K")
        ax_intr.axvline(F_now, color="tab:orange", ls=":", lw=1.2, label="Aktuális CSS (F)")
        ax_intr.set_title("Belső érték a CSS (F) függvényében")
        ax_intr.set_xlabel("CSS (F) – EUR/MWh"); ax_intr.set_ylabel("Belső érték (EUR/MWh)")
        ax_intr.grid(True, alpha=0.35); ax_intr.legend(loc="upper left")
        st.pyplot(fig_intr, clear_figure=True)

        # Mintatábla
        st.markdown("---")
        st.subheader("Bemeneti adatok (minta)")
        st.dataframe(df.tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"Hiba: {e}")

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
GAMMA_FALLBACK    = 0.6        # ha γ rolling becslés nem áll elő
VC_QUANTILE       = 95         # vol-cap kvantilis (évesített dCSS-vol alapján)

st.set_page_config(page_title="Forward CSS – Bachelier (GARCH + state-dependent σ)", layout="wide")
st.title("Forward CSS – Bachelier opcióárazó")
st.caption("Volatilitás: GARCH(1,1) t-eloszlás + state-dependent skálázás (RMED30 + γ) + vol-cap (q=95).")

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
    if len(y) < 30:  # 30 megfigyeléstől fusson
        raise ValueError("Túl rövid idősor a GARCH-hoz (≥ 30 megfigyelés szükséges).")
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

# ---------- γ becslés (rolling OLS + EMA) ----------
def _winsor(s: pd.Series, p: float):
    if p <= 0 or s.dropna().empty:
        return s
    lo, hi = np.nanpercentile(s.dropna(), [p, 100-p])
    return s.clip(lower=lo, upper=hi)

def estimate_gamma_series(F_series: pd.Series,
                          sig_base_daily: pd.Series,
                          real_vol_daily: pd.Series,
                          L_series_full: pd.Series,
                          winsor_pct: float = 1.0,
                          clip_gamma=(0.0, 2.0),
                          win: int = 120,
                          min_obs: int = 40,
                          ema_halflife: int = 30) -> pd.Series:
    idx = F_series.index.intersection(sig_base_daily.index).intersection(real_vol_daily.index).intersection(L_series_full.index)
    if len(idx) == 0:
        return pd.Series(dtype=float, name="gamma_t")
    F = F_series.reindex(idx).astype(float)
    S = sig_base_daily.reindex(idx).astype(float).clip(lower=1e-10)
    R = real_vol_daily.reindex(idx).astype(float).clip(lower=1e-10)
    L = L_series_full.reindex(idx).astype(float).clip(lower=1e-8)
    base = (np.abs(F) / (np.abs(F) + L)).clip(lower=1e-8, upper=1-1e-8)
    x = np.log(base); y = np.log((R / S).clip(lower=1e-8))
    xw, yw = _winsor(x, winsor_pct), _winsor(y, winsor_pct)

    gam = pd.Series(index=idx, dtype=float, name="gamma_t")
    bx, by = deque(), deque()
    for t, (xi, yi) in enumerate(zip(xw.values, yw.values)):
        bx.append(xi); by.append(yi)
        if len(bx) > win:
            bx.popleft(); by.popleft()
        bxx = np.array([v for v in bx if np.isfinite(v)])
        byy = np.array([v for v in by if np.isfinite(v)])
        if bxx.size >= min_obs and np.nanstd(bxx) > 1e-8:
            X = np.c_[np.ones_like(bxx), bxx]
            beta = np.linalg.lstsq(X, byy, rcond=None)[0]
            gam.iloc[t] = float(beta[1])
        else:
            gam.iloc[t] = np.nan

    if ema_halflife and ema_halflife > 0:
        gam = gam.ewm(halflife=ema_halflife, min_periods=min_obs).mean()
    if clip_gamma is not None:
        gam = gam.clip(lower=clip_gamma[0], upper=clip_gamma[1])
    return gam

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

    gamma_series = estimate_gamma_series(
        F_series=F, sig_base_daily=S, real_vol_daily=real_vol_daily,
        L_series_full=L_ser_full,
        winsor_pct=1.0, clip_gamma=(0.0, 2.0),
        win=120, min_obs=40, ema_halflife=30
    )
    gamma_hat = float(gamma_series.dropna().iloc[-1]) if not gamma_series.dropna().empty else np.nan
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
        if df_cut["dcss"].dropna().shape[0] < 30:
            st.error("Nincs elég adat az értékelési napig a GARCH illesztéshez (≥ 30 differencia szükséges).")
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

        # --- Hagyományos évesített vol a vizuális összehasonlításhoz: DÁTUM INDEXRE számolva ---
        s_dcss = df_cut.set_index(pd.to_datetime(df_cut["date"]))["dcss"].astype(float)
        ann_vol_series = (s_dcss.rolling(ROLL_WIN_STD).std(ddof=1) * np.sqrt(BUS_DAYS_PER_YEAR)).dropna()
        sigma_ann_trad = float(ann_vol_series.iloc[-1]) if not ann_vol_series.empty else sigma_ann  # fallback

        # --- Opcióár ---
        price = bachelier_price(F_now, float(K), sigma_ann, float(T_years), call_put)
        intrinsic = max(F_now - float(K), 0.0) if call_put == "call" else max(float(K) - F_now, 0.0)
        time_value = max(0.0, price - intrinsic)

        st.subheader("Eredmény (Bachelier, σ = GARCH + state-dependent)")
        st.markdown(
            f"**CSS (F):** `{F_now:.4f}` EUR/MWh  |  "
            f"**σ (annualizált, state-dep):** `{sigma_ann:.6f}`  |  "
            f"**σ (annualizált, hagyományos):** `{sigma_ann_trad:.6f}`  |  "
            f"**T (év):** `{T_years:.4f}`  |  "
            f"**K:** `{float(K):.4f}`  \n\n"
            f"**Opció prémium:** `{price:.6f}`  |  "
            f"**Belső érték:** `{intrinsic:.6f}`  |  "
            f"**Időérték:** `{time_value:.6f}`"
        )

        # --- Vizualizációk ---
        import matplotlib.pyplot as plt

        st.markdown("---")
        st.subheader("Vizualizációk")

        # (1) Időérték T függvényében
        T_grid = np.linspace(1e-4, max(T_years, 1e-3), 60)
        tv_curve = [max(0.0, bachelier_price(F_now, float(K), sigma_ann, t, call_put) - intrinsic) for t in T_grid]
        T_grid_np = np.asarray(T_grid, dtype=float)
        tv_curve_np = np.asarray(tv_curve, dtype=float)
        fig_tv, ax_tv = plt.subplots(figsize=(11, 3.0))
        ax_tv.plot(T_grid_np, tv_curve_np, lw=1.8)
        ax_tv.set_title("1) Opció időértéke a futamidő függvényében (Bachelier)")
        ax_tv.set_xlabel("T (év)"); ax_tv.set_ylabel("EUR/MWh"); ax_tv.grid(True, alpha=0.35)
        st.pyplot(fig_tv, clear_figure=True)

        # (2) Belső érték F függvényében
        span = max(5.0, sigma_ann * sqrt(max(T_years, 1e-4)) * 6.0)
        F_grid = np.linspace(float(K) - span, float(K) + span, 200)
        intrinsic_curve = np.maximum(F_grid - float(K), 0.0) if call_put == "call" else np.maximum(float(K) - F_grid, 0.0)
        fig_intr, ax_intr = plt.subplots(figsize=(11, 3.0))
        ax_intr.plot(np.asarray(F_grid), np.asarray(intrinsic_curve), lw=1.8, label="Belső érték")
        ax_intr.axvline(float(K), color="gray", ls="--", lw=1.0, label="Strike K")
        ax_intr.axvline(F_now, color="tab:orange", ls=":", lw=1.2, label="Aktuális CSS (F)")
        ax_intr.set_title("2) Belső érték a CSS (F) függvényében")
        ax_intr.set_xlabel("CSS (F) – EUR/MWh"); ax_intr.set_ylabel("Belső érték (EUR/MWh)")
        ax_intr.grid(True, alpha=0.35); ax_intr.legend(loc="upper left")
        st.pyplot(fig_intr, clear_figure=True)

        # (3) Volatilitás idősor – state-dependent vs. hagyományos
        sd_series_plot = sd_series.dropna()
        figv, axv = plt.subplots(figsize=(11, 3.6))
        if not sd_series_plot.empty:
            axv.plot(sd_series_plot.index, sd_series_plot.values, lw=1.8, label="State-dependent σ (évesített)")
        if not ann_vol_series.empty:
            axv.plot(ann_vol_series.index, ann_vol_series.values, lw=1.2, ls="--", label=f"Hagyományos σ (dCSS, {ROLL_WIN_STD}n)")
        axv.set_title("3) Évesített volatilitás – értékelési napig")
        axv.set_ylabel("σ (EUR/MWh)"); axv.set_xlabel("Dátum"); axv.grid(True, alpha=0.35); axv.legend(loc="upper left")
        st.pyplot(figv, clear_figure=True)

        # (4) Fáklyadiagram: 95% CI jövőre (state-dep vs. hagyományos σ) – egyező hosszú numpy vektorokkal
        n_steps = 60 if T_years > 0 else 2
        t_years_grid = np.linspace(0.0, float(T_years), n_steps)
        z = norm.ppf(0.975)  # 95% kétoldali
        mean_path = np.full_like(t_years_grid, F_now, dtype=float)

        sd_state = sigma_ann * np.sqrt(np.maximum(t_years_grid, 1e-12))
        sd_trad  = sigma_ann_trad * np.sqrt(np.maximum(t_years_grid, 1e-12))

        lower_state = mean_path - z * sd_state
        upper_state = mean_path + z * sd_state
        lower_trad  = mean_path - z * sd_trad
        upper_trad  = mean_path + z * sd_trad

        days_grid = t_years_grid * 365.25
        # minden vektor tisztán 1D és azonos hossz
        days_grid = np.asarray(days_grid, dtype=float)
        lower_trad = np.asarray(lower_trad, dtype=float); upper_trad = np.asarray(upper_trad, dtype=float)
        lower_state = np.asarray(lower_state, dtype=float); upper_state = np.asarray(upper_state, dtype=float)

        fig_fan, ax_fan = plt.subplots(figsize=(11, 3.6))
        ax_fan.plot(days_grid, mean_path, lw=1.6, label="Várható pálya (F_now)")
        ax_fan.fill_between(days_grid, lower_trad, upper_trad, alpha=0.25, label="95% CI – hagyományos σ", color="gray")
        ax_fan.fill_between(days_grid, lower_state, upper_state, alpha=0.25, label="95% CI – state-dependent σ", color="tab:blue")
        ax_fan.set_title("4) CSS fáklyadiagram – 95% konfidenciasáv (hagyományos vs. state-dependent σ)")
        ax_fan.set_xlabel("Hátralévő napok a lejáratig"); ax_fan.set_ylabel("CSS (EUR/MWh)")
        ax_fan.grid(True, alpha=0.35); ax_fan.legend(loc="upper left")
        st.pyplot(fig_fan, clear_figure=True)

        # Mintatábla
        st.markdown("---")
        st.subheader("Bemeneti adatok (minta)")
        st.dataframe(df.tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"Hiba: {e}")

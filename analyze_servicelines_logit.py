import os
import sys
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime


# Write all regression artifacts here
ANALYSIS_DIR = os.path.join("data", "regression")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_master() -> gpd.GeoDataFrame:
    """Load the house-level master file produced by master_file_creation.py.

    Searches common locations regardless of where regression outputs are written.
    """
    candidates = [
        os.path.join("data", "analysis", "servicelines_house_with_attributes"),
        os.path.join("data", "regression", "servicelines_house_with_attributes"),
        os.path.join("data", "servicelines_house_with_attributes"),
    ]
    exts = (".geoparquet", ".parquet", ".geojson", ".csv")
    for base in candidates:
        for ext in exts:
            p = base + ext
            if os.path.exists(p):
                try:
                    if ext in (".geoparquet", ".parquet"):
                        return gpd.read_parquet(p)
                    if ext == ".geojson":
                        return gpd.read_file(p)
                    if ext == ".csv":
                        df = pd.read_csv(p, low_memory=False)
                        if {"lon", "lat"}.issubset(df.columns):
                            return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
                        return gpd.GeoDataFrame(df, geometry=None)
                except Exception as e:
                    log(f"[WARN] Failed to read {p}: {e}")
    raise SystemExit("Master file servicelines_house_with_attributes not found. Run master_file_creation.py first.")


def normalize_status(val: Optional[str]) -> str:
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().lower()
    if "non" in s and "lead" in s:
        return "Non-Lead"
    if "lead" in s:
        return "Lead"
    if s in {"yes", "y"}:
        return "Lead"
    if s in {"no", "n"}:
        return "Non-Lead"
    return "Unknown"


def build_logit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(df.copy())
    # Target from bothsidesstatus
    if "bothsidesstatus" not in out.columns:
        raise SystemExit("bothsidesstatus column not found in master file")
    out["bothsidesstatus_cat"] = out["bothsidesstatus"].map(normalize_status)
    out = out[out["bothsidesstatus_cat"].isin(["Lead", "Non-Lead"])].copy()
    out["target_lead"] = (out["bothsidesstatus_cat"] == "Lead").astype(int)

    # Age features
    year_col = None
    for c in ["yearstructbuilt", "YearBuilt", "year_built", "YEARBUILT"]:
        if c in out.columns:
            year_col = c
            break
    if year_col is None:
        raise SystemExit("No year built column (yearstructbuilt/YearBuilt/etc.) found in master file")

    y = pd.to_numeric(out[year_col], errors="coerce")
    current_year = pd.Timestamp.now().year
    # Clean zeros and future years; keep true pre-1800 structures
    valid_year = (y > 0) & (y <= current_year)
    out.loc[~valid_year, year_col] = np.nan
    out["age_years"] = current_year - pd.to_numeric(out[year_col], errors="coerce")

    # Post-1988 indicator (>= 1988 considered post-1988)
    out["post_1988"] = (pd.to_numeric(out[year_col], errors="coerce") >= 1988).astype(int)

    # Interaction term: age x post_1988
    out["age_x_post_1988"] = out["age_years"] * out["post_1988"]

    # Income (MHI) as numeric covariate if available
    if "MHI" in out.columns:
        out["MHI"] = pd.to_numeric(out["MHI"], errors="coerce")
        out["MHI_10k"] = out["MHI"] / 10000.0

    # Additional covariates
    if "Pblack" in out.columns:
        out["Pblack"] = pd.to_numeric(out["Pblack"], errors="coerce")
    if "J40_TotCatExceeded" in out.columns:
        out["J40_TotCatExceeded_num"] = pd.to_numeric(out["J40_TotCatExceeded"], errors="coerce").fillna(0).astype(int)
    # Disadvantaged as binary (Yes/True/1 → 1)
    if "disadvantaged" in out.columns:
        def _bin(x):
            if pd.isna(x):
                return 0
            s = str(x).strip().lower()
            return 1 if s in {"yes", "y", "true", "t", "1"} else 0
        out["disadvantaged_num"] = out["disadvantaged"].apply(_bin)

    # Keep only records with valid age
    out = out[out["age_years"].notna()].copy()
    keep_cols = ["target_lead", "age_years", "post_1988", "age_x_post_1988", year_col]
    if "MHI_10k" in out.columns:
        keep_cols.append("MHI_10k")
    if "Pblack" in out.columns:
        keep_cols.append("Pblack")
    if "J40_TotCatExceeded_num" in out.columns:
        keep_cols.append("J40_TotCatExceeded_num")
    if "disadvantaged_num" in out.columns:
        keep_cols.append("disadvantaged_num")
    if "neighborhood_name" in out.columns:
        keep_cols.append("neighborhood_name")
    return out[keep_cols].rename(columns={year_col: "year_built"})


def fit_logit_models(ds: pd.DataFrame):
    """Fit both statsmodels and scikit-learn logit models and return rich outputs.

    Returns a dict with keys (present when available):
      - sm_terms: per-term statsmodels table (coef, p, CI, odds ratios)
      - sm_model: model-level stats (N, llf, llnull, pseudo_r2_mcfadden, AIC, BIC)
      - sk_coefs: sklearn coefficients (intercept + betas)
      - sk_metrics: sklearn metrics (AUC, accuracy, precision, recall, f1, log_loss, brier)
      - sk_confusion: confusion matrix counts
    """
    feature_cols = ["age_years", "post_1988", "age_x_post_1988"]
    if "MHI_10k" in ds.columns:
        feature_cols.append("MHI_10k")
    if "Pblack" in ds.columns:
        feature_cols.append("Pblack")
    if "J40_TotCatExceeded_num" in ds.columns:
        feature_cols.append("J40_TotCatExceeded_num")
    if "disadvantaged_num" in ds.columns:
        feature_cols.append("disadvantaged_num")
    X = ds[feature_cols].copy()
    # Neighborhood fixed effects (dummies, drop first to avoid multicollinearity)
    if "neighborhood_name" in ds.columns:
        neigh = pd.get_dummies(ds["neighborhood_name"].astype("category"), prefix="neigh", drop_first=True)
        X = pd.concat([X, neigh], axis=1)

    # Coerce to numeric and align y and X for statsmodels
    X = X.apply(pd.to_numeric, errors="coerce")
    y_series = ds["target_lead"].astype(int)
    dfm = pd.concat([y_series.rename("target_lead"), X], axis=1)
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna()
    y = ds["target_lead"].astype(int).values

    out = {}

    # Statsmodels with p-values and pseudo R^2
    try:
        import statsmodels.api as sm

        X_sm = sm.add_constant(dfm.drop(columns=["target_lead"]).astype(float))
        y_sm = dfm["target_lead"].astype(int).values
        model = sm.Logit(y_sm, X_sm)
        res = model.fit(disp=False)
        params = res.params
        bse = res.bse
        pvals = res.pvalues
        zvals = res.tvalues
        conf = res.conf_int()
        terms = pd.DataFrame({
            "term": params.index,
            "coef": params.values,
            "std_err": bse.values,
            "z_value": zvals.values,
            "p_value": pvals.values,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
            "odds_ratio": np.exp(params.values),
            "or_ci_lower": np.exp(conf[0].values),
            "or_ci_upper": np.exp(conf[1].values),
        })
        # Model-level stats
        llf = res.llf
        llnull = getattr(res, "llnull", np.nan)
        pseudo_r2 = np.nan
        if llnull and np.isfinite(llf) and np.isfinite(llnull) and llnull != 0:
            pseudo_r2 = 1 - (llf / llnull)
        sm_model = pd.DataFrame([
            {
                "n_obs": int(res.nobs),
                "llf": float(llf),
                "llnull": float(llnull) if np.isfinite(llnull) else np.nan,
                "pseudo_r2_mcfadden": float(pseudo_r2) if np.isfinite(pseudo_r2) else np.nan,
                "aic": float(res.aic) if hasattr(res, "aic") else np.nan,
                "bic": float(res.bic) if hasattr(res, "bic") else np.nan,
            }
        ])
        out["sm_terms"] = terms
        out["sm_model"] = sm_model
    except Exception as e:
        log(f"[INFO] statsmodels not available or failed ({e}); continuing with scikit-learn only.")

    # scikit-learn metrics
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            log_loss,
            brier_score_loss,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split

        X2 = dfm.drop(columns=["target_lead"]).astype(float)
        y2 = dfm["target_lead"].astype(int).values
        X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.25, random_state=42, stratify=y2)
        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(X_train, y_train)
        coefs = pd.DataFrame({
            "term": ["intercept"] + X2.columns.tolist(),
            "coef": np.concatenate([clf.intercept_, clf.coef_.ravel()])
        })
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sk_metrics = pd.DataFrame([
            {"metric": "auc", "value": roc_auc_score(y_test, y_prob)},
            {"metric": "accuracy", "value": accuracy_score(y_test, y_pred)},
            {"metric": "precision", "value": precision_score(y_test, y_pred, zero_division=0)},
            {"metric": "recall", "value": recall_score(y_test, y_pred, zero_division=0)},
            {"metric": "f1", "value": f1_score(y_test, y_pred, zero_division=0)},
            {"metric": "log_loss", "value": log_loss(y_test, y_prob)},
            {"metric": "brier", "value": brier_score_loss(y_test, y_prob)},
        ])
        sk_conf = pd.DataFrame(cm, columns=["pred_0", "pred_1"], index=["true_0", "true_1"]).reset_index().rename(columns={"index": ""})
        out["sk_coefs"] = coefs
        out["sk_metrics"] = sk_metrics
        out["sk_confusion"] = sk_conf
    except Exception as e:
        log(f"[WARN] scikit-learn not available or failed: {e}")

    return out


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    gdf = load_master()
    ds = build_logit_dataset(gdf)

    # Save modeling dataset
    def safe_write_df(df: pd.DataFrame, path: str):
        try:
            df.to_csv(path, index=False)
            log(f"[OK] Wrote {path}")
        except PermissionError:
            alt = os.path.splitext(path)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_path = f"{alt[0]}_{ts}{alt[1]}"
            df.to_csv(new_path, index=False)
            log(f"[WARN] Permission denied on {path}; wrote to {new_path}")

    ds_csv = os.path.join(ANALYSIS_DIR, "servicelines_logit_age_dataset.csv")
    safe_write_df(ds, ds_csv)

    # Fit models
    res = fit_logit_models(ds)
    # statsmodels outputs
    if "sm_terms" in res and not res["sm_terms"].empty:
        sm_terms_csv = os.path.join(ANALYSIS_DIR, "servicelines_logit_age_terms_statsmodels.csv")
        safe_write_df(res["sm_terms"], sm_terms_csv)
    if "sm_model" in res and not res["sm_model"].empty:
        sm_model_csv = os.path.join(ANALYSIS_DIR, "servicelines_logit_age_model_statsmodels.csv")
        safe_write_df(res["sm_model"], sm_model_csv)
    # sklearn outputs
    if "sk_coefs" in res and not res["sk_coefs"].empty:
        sk_coefs_csv = os.path.join(ANALYSIS_DIR, "servicelines_logit_age_coefs_sklearn.csv")
        safe_write_df(res["sk_coefs"], sk_coefs_csv)
    if "sk_metrics" in res and not res["sk_metrics"].empty:
        sk_metrics_csv = os.path.join(ANALYSIS_DIR, "servicelines_logit_age_metrics_sklearn.csv")
        safe_write_df(res["sk_metrics"], sk_metrics_csv)
    if "sk_confusion" in res and not res["sk_confusion"].empty:
        sk_conf_csv = os.path.join(ANALYSIS_DIR, "servicelines_logit_age_confusion_sklearn.csv")
        safe_write_df(res["sk_confusion"], sk_conf_csv)


if __name__ == "__main__":
    main()

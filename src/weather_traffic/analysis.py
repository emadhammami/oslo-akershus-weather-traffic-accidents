"""Portable refactor of the weather/traffic-accident coursework workflow.

Derived from the sanitized historical Colab experiment. Source data are not
included and the refactor has not independently reproduced the report metrics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

FROST_ENDPOINT = "https://frost.met.no/observations/v0.jsonld"
START = "2014-01-01"
END = "2024-12-31"
STATIONS = ("SN18700", "SN4780", "SN17850")
TEMP = "mean(air_temperature P1D)"
PRECIP = "sum(precipitation_amount P1D)"
MONTHS = {n: i for i, n in enumerate((
    "Januar", "Februar", "Mars", "April", "Mai", "Juni", "Juli",
    "August", "September", "Oktober", "November", "Desember"), 1)}
PREDICTORS = ["mean_air_temperature", "total_precipitation", "month_sin", "month_cos", "year_index"]


def fetch_frost(*, client_id: str | None = None, client_secret: str | None = None,
                start: str = START, end: str = END, stations: tuple[str, ...] = STATIONS) -> pd.DataFrame:
    """Fetch daily observations with explicit authentication; no secrets in logs."""
    client_id = client_id or os.getenv("FROST_CLIENT_ID")
    if not client_id:
        raise ValueError("Set FROST_CLIENT_ID in your environment before fetching Frost data")
    secret = client_secret if client_secret is not None else os.getenv("FROST_CLIENT_SECRET", "")
    records = []
    for station in stations:
        response = requests.get(
            FROST_ENDPOINT,
            params={"sources": station, "elements": f"{TEMP},{PRECIP}",
                    "referencetime": f"{start}/{end}"},
            auth=(client_id, secret),
            headers={"User-Agent": "USN-MIS404-Weather-Analytics/0.1"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json().get("data")
        if not isinstance(data, list):
            raise ValueError(f"Missing Frost response data for station {station}")
        for entry in data:
            row = {"station": station, "date": pd.to_datetime(entry["referenceTime"]).normalize()}
            for observation in entry.get("observations", []):
                row[observation["elementId"]] = observation["value"]
            records.append(row)
    if not records:
        raise ValueError("Frost returned no weather observations")
    return pd.DataFrame(records)


def monthly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean across regional stations per day, then mean/sum over each month."""
    missing = {"date", TEMP, PRECIP}.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing weather columns: {sorted(missing)}")
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.tz_localize(None).dt.normalize()
    for name in (TEMP, PRECIP):
        data[name] = pd.to_numeric(data[name], errors="coerce")
    region = (data.groupby("date", as_index=False)[[TEMP, PRECIP]].mean()
              .rename(columns={TEMP: "daily_mean_temperature", PRECIP: "daily_mean_precipitation"}))
    region["year"] = region["date"].dt.year
    region["month"] = region["date"].dt.month
    monthly = (region.groupby(["year", "month"], as_index=False)
               .agg(mean_air_temperature=("daily_mean_temperature", "mean"),
                    total_precipitation=("daily_mean_precipitation", lambda s: s.sum(min_count=1))))
    return monthly


def load_accidents(csv_path: str | Path, *, start_year: int = 2014,
                   end_year: int = 2024) -> pd.DataFrame:
    """Read the original monthly, semicolon-delimited TRINE export."""
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    df.columns = [str(s).strip().replace("\n", "") for s in df.columns]
    names = {"År": "year", "Måned": "month_name", "Dødsulykke": "fatal_accidents",
             "Ulykke med hardt skadde": "serious_injury_accidents",
             "Ulykke med lettere skadde": "minor_injury_accidents"}
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing accident columns: {sorted(missing)}")
    df = df.rename(columns=names)
    df["month"] = df["month_name"].astype(str).str.strip().map(MONTHS)
    if df["month"].isna().any():
        raise ValueError("Unrecognized Norwegian month name(s) in TRINE data")
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df = df.loc[df["year"].between(start_year, end_year)].copy()
    categories = ["fatal_accidents", "serious_injury_accidents", "minor_injury_accidents"]
    for name in categories:
        df[name] = pd.to_numeric(df[name], errors="raise").fillna(0)
        if (df[name] < 0).any():
            raise ValueError(f"Negative accident count: {name}")
    if df.duplicated(subset=["year", "month"]).any():
        raise ValueError("Multiple TRINE rows for a year/month; aggregate intentionally before running")
    df["month"] = df["month"].astype(int)
    df["injury_accidents"] = df[categories].sum(axis=1)
    return df[["year", "month", *categories, "injury_accidents"]]


def join_and_features(accidents: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(accidents, weather, on=["year", "month"], how="inner", validate="one_to_one")
    if df.empty:
        raise ValueError("No matching months between accident and weather data")
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df = df.sort_values("date").reset_index(drop=True)
    df["year_index"] = df["year"] - df["year"].min()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    required = ["injury_accidents", *PREDICTORS]
    if df[required].isna().any().any():
        raise ValueError("Missing weather or accident values: inspect coverage before modeling")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate joined year-month rows")
    return df


def run_models(df: pd.DataFrame, *, outdir: str | Path, seed: int = 42) -> pd.DataFrame:
    """Train/evaluate the three report model classes on a chronological holdout."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if len(df) < 24:
        raise ValueError("Insufficient monthly observations; at least 24 required")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Data must be sorted chronologically before evaluation")
    split = int(len(df) * 0.8)
    X, y = df[PREDICTORS], df["injury_accidents"]
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")
    poisson = sm.GLM(y_train, X_train_sm, family=sm.families.Poisson()).fit()
    negbin = NegativeBinomial(y_train, X_train_sm).fit(disp=False)
    forest = RandomForestRegressor(n_estimators=100, random_state=seed).fit(X_train, y_train)
    predictions = {
        "Poisson regression": poisson.predict(X_test_sm),
        "Negative binomial regression": negbin.predict(X_test_sm),
        "Random forest regression": forest.predict(X_test),
    }
    metrics = pd.DataFrame([{
        "Model": name,
        "MAE": float(mean_absolute_error(y_test, np.asarray(values))),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, np.asarray(values)))),
    } for name, values in predictions.items()])
    metrics.to_csv(outdir / "model_comparison.csv", index=False)
    coef = pd.DataFrame({"Variable": poisson.params.index,
                         "Coefficient": poisson.params.values,
                         "IRR": np.exp(poisson.params.values),
                         "p-value": poisson.pvalues.values})
    coef.to_csv(outdir / "poisson_coefficients_irr.csv", index=False)
    pd.DataFrame({"Feature": PREDICTORS, "Importance": forest.feature_importances_}).to_csv(
        outdir / "random_forest_feature_importance.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"].iloc[split:], y_test.to_numpy(), marker="o", label="Observed")
    for name, pred in predictions.items():
        ax.plot(df["date"].iloc[split:], np.asarray(pred), label=name)
    ax.set(title="Chronological holdout: injury accidents by month", xlabel="Month", ylabel="Accidents")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outdir / "actual_vs_predicted_holdout.png", dpi=160)
    plt.close(fig)
    fitted = np.asarray(poisson.predict(X_train_sm))
    dispersion = float(np.sum((y_train.to_numpy() - fitted) ** 2 / np.maximum(fitted, 1e-12)) / poisson.df_resid)
    summary = {"records": len(df), "train_months": split, "test_months": len(df) - split,
               "first_month": str(df["date"].iloc[0].date()), "last_month": str(df["date"].iloc[-1].date()),
               "poisson_pearson_overdispersion": dispersion,
               "note": "Exploratory refactor; historical course-report metrics are not automatically reproduced."}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return metrics

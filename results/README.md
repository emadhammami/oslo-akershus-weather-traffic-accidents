# Results from the executed Colab study

These tables and figures were extracted from the **original executed Google Colab notebook** for the MIS404 group case study. The notebook execution is dated **13 October 2026**; the underlying study covers January 2014–December 2024 in Oslo and Akershus.

## Evaluation setup

- 132 monthly joined observations, following aggregation from 12,051 station-day weather rows.
- Chronological split: 105 training months and 27 holdout months.
- Outcome: monthly count of personal-injury traffic accidents.
- Models: Poisson regression, negative binomial regression, and random forest regression.
- Test MAE and RMSE were copied by a parser from the **printed six-decimal notebook output**, not rerun or manually invented.

## Holdout results

| Model | MAE (lower better) | RMSE (lower better) |
|---|---:|---:|
| Poisson Regression | 10.544880 | 14.122437 |
| Negative Binomial Regression | 10.421884 | 13.907886 |
| **Random Forest Regressor** | **9.498889** | **11.434560** |

The Pearson overdispersion statistic for the fitted Poisson model was **2.78**, suggesting overdispersion. The random forest had the lowest error on this particular 27-month holdout; this is not a production forecast validation.

## Primary figures

- [Observed vs predicted accident counts on chronological holdout](../figures/11_holdout_actual_vs_predicted.png)
- [Random forest feature importances](../figures/12_random_forest_feature_importance.png)
- [Poisson regression coefficients](../figures/10_poisson_coefficients.png)
- [Monthly accident counts](../figures/01_monthly_personal_injury_accidents.png)
- [Relationship with temperature](../figures/04_temperature_vs_accidents.png)
- [Relationship with precipitation](../figures/05_precipitation_vs_accidents.png)

All 12 figures were recovered as actual `image/png` outputs in the notebook. `experiment_metadata.json` records which notebook output each image came from and a SHA-256 for the extracted image bytes.

## Provenance and constraints

This is a **group coursework case study**, not a peer-reviewed research publication or a deployed safety system. Results are observational associations, not causal estimates. The test period contains only 27 months, and monthly weather aggregates cannot establish crash-level effects. Traffic exposure and other important confounders are not fully modeled.

The source TRINE CSV and daily Frost observations are **not included**. The original notebook was **not included**, because its historical source contained embedded API credentials. The repository's public notebook separately excludes credentials and saved outputs. The portable refactor in `src/` has not been independently rerun here; these archived outputs come from the original Colab execution, not the refactor.

# ML Plan

## Goal

Classify thermal events detected from NASA FIRMS data into categories such
as industrial fire, wildfire, agricultural burning, gas flare, and other
persistent thermal sources — and flag abnormal thermal activity relative to
a location's historical baseline.

## Planned feature groups

- **Thermal features** — brightness temperature, fire radiative power (FRP),
  detection confidence, and how these trend across repeated detections.
- **Spatial features** — proximity to OSM-tagged industrial sites, roads,
  settlements; land-cover/land-use classification at and around the
  detection point.
- **Land-cover features** — vegetation vs. bare land vs. built-up
  classification from satellite/land-cover data, used to help separate
  agricultural/wildfire signals from industrial ones.
- **Temporal features** — detection frequency and persistence over time,
  time-of-day/seasonality patterns, and deviation from historical baseline
  activity at the same location.

## Candidate models

Initial candidates under consideration for the classification task:

- **Random Forest** — strong baseline, robust to mixed feature types, easy
  to interpret via feature importance.
- **XGBoost** — typically strong performance on tabular data with
  engineered features; supports class imbalance handling.

Neither is selected as the final model — the choice will be driven by
evaluation results once labeled data and features are available.

## Evaluation approach

Standard classification metrics will be used: precision, recall, F1-score
(per class and macro/weighted averages), and confusion matrix. See
`aiml/src/evaluation/metrics.py` for the (currently placeholder) evaluation
interface.

## Anomaly detection & risk scoring

Beyond classification, the system will compare each thermal source against
its own historical baseline to flag abnormal activity, and combine
classification confidence, persistence, and anomaly signals into a risk
score surfaced to the frontend.

## Current status

No model has been trained. No datasets have been downloaded. All modules in
`aiml/src/` are placeholders with `TODO` comments describing the intended
implementation. This document will be updated as the approach is validated
against real data.

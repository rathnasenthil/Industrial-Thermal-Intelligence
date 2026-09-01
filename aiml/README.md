# AIML — Industrial Fire Intelligence Platform

Python package for data ingestion, preprocessing, feature engineering,
model training, evaluation and inference for classifying industrial fires
and persistent thermal sources.

## Tech stack

- Python 3.12+
- pandas, numpy, scipy
- scikit-learn, xgboost
- geopandas, shapely, pyproj, rasterio (geospatial)
- requests, httpx (data ingestion)
- matplotlib, jupyter (exploration)

## Folder structure

```
aiml/
├── data/
│   ├── raw/         # untouched source data (gitignored)
│   ├── processed/   # cleaned/merged datasets (gitignored)
│   └── external/    # third-party reference data (gitignored)
├── notebooks/        # exploratory analysis
├── src/
│   ├── data_ingestion/       # firms.py, osm.py
│   ├── preprocessing/        # cleaning, event formation
│   ├── feature_engineering/  # spatial/temporal/thermal features
│   ├── models/                # classifier
│   ├── evaluation/            # metrics
│   └── inference/             # prediction entrypoint for the backend
└── tests/             # environment/import sanity tests
```

## Local development

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Tests

```bash
pytest
```

These tests only verify the Python environment and that modules import
correctly — no data ingestion, training, or inference logic is implemented
yet.

## Notes

- No real datasets, trained models, or prediction outputs are included.
- All ingestion/feature/model/inference functions are placeholders with
  `TODO` comments and raise `NotImplementedError`.
- The backend will eventually call into `src/inference/predict.py` through
  a clean interface — avoid duplicating this logic elsewhere.

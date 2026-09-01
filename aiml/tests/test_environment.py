"""
Basic environment sanity checks for the AIML package.

These tests intentionally do not exercise any real data-processing or ML
logic (none is implemented yet) — they only confirm that the Python
environment is set up correctly and that core dependencies import cleanly.
"""

import sys


def test_python_version() -> None:
    assert sys.version_info >= (3, 12), "AIML requires Python 3.12+"


def test_core_dependencies_import() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401
    import geopandas  # noqa: F401
    import shapely  # noqa: F401


def test_src_package_imports() -> None:
    from src.data_ingestion import firms, osm  # noqa: F401
    from src.preprocessing import preprocessing  # noqa: F401
    from src.feature_engineering import features  # noqa: F401
    from src.models import classifier  # noqa: F401
    from src.evaluation import metrics  # noqa: F401
    from src.inference import predict  # noqa: F401

"""
Confidence and day/night formatting normalization for FIRMS detections.

This module deliberately does the *minimum* possible: it strips stray
whitespace and normalizes letter case so that e.g. ``" n"`` and ``"N"``
are treated the same as ``"n"``. It does NOT:

* map FIRMS' categorical confidence codes (``n``/``l``/``h``) to numbers,
* invent a new confidence scale,
* reclassify day/night values.

The native confidence field may later be used as an evidence feature by
the Trust Scoring stage, but its meaning must not be redefined here.
"""

from __future__ import annotations

import pandas as pd


def normalize_confidence(confidence: pd.Series) -> pd.Series:
    """Strip whitespace and lower-case alphabetic confidence codes.

    VIIRS confidence values are the letter codes ``n`` (nominal),
    ``l`` (low) and ``h`` (high); some FIRMS products instead use a
    numeric 0-100 confidence. Only whitespace is stripped for numeric-look
    values; alphabetic codes are additionally lower-cased. The underlying
    values and their meaning are otherwise preserved exactly.

    Args:
        confidence: The raw ``confidence`` column.

    Returns:
        A cleaned copy of the column. Missing values remain missing.
    """
    cleaned = confidence.astype("string").str.strip()
    is_alpha = cleaned.str.isalpha().fillna(False)
    cleaned = cleaned.mask(is_alpha, cleaned.str.lower())
    return cleaned


def normalize_daynight(daynight: pd.Series) -> pd.Series:
    """Strip whitespace and upper-case the day/night flag.

    FIRMS uses single-letter codes ``D`` (day) / ``N`` (night). This only
    normalizes formatting (whitespace/case); it does not invent new
    categories or reinterpret the flag.

    Args:
        daynight: The raw ``daynight`` column.

    Returns:
        A cleaned copy of the column. Missing values remain missing.
    """
    cleaned = daynight.astype("string").str.strip()
    is_alpha = cleaned.str.isalpha().fillna(False)
    cleaned = cleaned.mask(is_alpha, cleaned.str.upper())
    return cleaned

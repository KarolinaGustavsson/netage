"""Column definitions and type specifications for the AMORIS dataset.

The 15 biomarker features span six pre-specified groups (lipid, inflammation,
renal, hepatic, glycemic, hematologic); group membership lives in
configs/variable_groups.yaml and must not be derived from this file.
"""
from __future__ import annotations

from typing import Final

META_COLS: Final[list[str]] = ["id", "sex"]

# Attained-age time scale: individuals enter at age_at_baseline (left truncation)
# and exit at age_at_exit (event or censoring).
SURVIVAL_COLS: Final[list[str]] = ["age_at_baseline", "age_at_exit", "event"]

FEATURE_COLS: Final[list[str]] = [
    # lipid
    "cholesterol",    # total cholesterol (mmol/L)
    "triglycerides",  # serum triglycerides (mmol/L)
    "hdl",            # HDL cholesterol (mmol/L)
    "ldl",            # LDL cholesterol (mmol/L)
    # inflammation
    "crp",            # C-reactive protein (mg/L)
    "wbc",            # white blood cell count (10^9/L)
    "albumin",        # serum albumin (g/L)
    # renal
    "creatinine",     # serum creatinine (µmol/L)
    "uric_acid",      # serum uric acid (µmol/L)
    # hepatic
    "alp",            # alkaline phosphatase (U/L)
    "ggt",            # gamma-glutamyl transferase (U/L)
    "alt",            # alanine aminotransferase (U/L)
    # glycemic
    "glucose",        # fasting plasma glucose (mmol/L)
    # hematologic
    "hemoglobin",     # hemoglobin (g/dL)
    "iron",           # serum iron (µmol/L)
]

ALL_COLS: Final[list[str]] = META_COLS + SURVIVAL_COLS + FEATURE_COLS

EXPECTED_DTYPES: Final[dict[str, str]] = {
    "id": "object",
    "sex": "int64",
    "age_at_baseline": "float64",
    "age_at_exit": "float64",
    "event": "int64",
    **{col: "float64" for col in FEATURE_COLS},
}

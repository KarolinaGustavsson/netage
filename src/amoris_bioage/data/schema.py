"""Column definitions and type specifications for the AMORIS dataset.

The 17 biomarker features span seven pre-specified groups (lipid, inflammation,
renal, hepatic, glycemic, hematologic_iron, mineral_electrolyte); group membership lives in
configs/variable_groups.yaml and must not be derived from this file.

Cox model training uses all-cause mortality (status/event = 1 indicates death).
Cause-specific outcomes (dementia, etc.) are derived separately from the Event column
for validation and downstream analysis.

Event column codes (from scrambled_b.csv):
  -10: Dementia diagnosis before sampling (prevalent)
   10: Dementia diagnosis after sampling (incident)
   20: Date of death (mortality)
   30: Emigration (censored)
   40: Lost to follow-up (censored)
   50: Alive at 2020-12-31 (censored)
"""
from __future__ import annotations

from typing import Final

META_COLS: Final[list[str]] = ["id", "sex"]

# Attained-age time scale: individuals enter at age_at_baseline (left truncation)
# and exit at age_at_exit (event or censoring).
SURVIVAL_COLS: Final[list[str]] = ["age_at_baseline", "age_at_exit", "event"]

FEATURE_COLS: Final[list[str]] = [
    # lipid
    "TC",             # serum cholesterol (mmol/L)
    "TG",             # fasting serum triglycerides (mmol/L)
    # inflammation
    "S_Hapt",         # serum haptoglobin (g/L)
    "S_Alb",          # serum albumin (g/L)
    # renal
    "S_Krea",         # serum creatinine (µmol/L)
    "S_Urat",         # serum urate / uric acid (µmol/L)
    "S_Urea",         # serum urea / blood urea nitrogen (mmol/L)
    # hepatic
    "S_Alp",          # serum alkaline phosphatase (U/L)
    "S_LD",           # serum lactate dehydrogenase (U/L)
    # glycemic
    "fS_Gluk",        # fasting serum glucose (mmol/L)
    "S_FAMN",         # serum fructosamine (µmol/L)
    # hematologic / iron metabolism
    "fS_Jaern",       # fasting serum iron (µmol/L)
    "fS_TIBC",        # total iron-binding capacity (µmol/L)
    "Fe_maet",        # transferrin saturation / iron saturation (%)
    # minerals / electrolytes
    "S_Ca",           # serum calcium (mmol/L)
    "S_P",            # serum phosphate (mmol/L)
    "S_K",            # serum potassium (mmol/L)
]

ALL_COLS: Final[list[str]] = META_COLS + SURVIVAL_COLS + FEATURE_COLS

# Column name mapping from scrambled_b.csv to schema
CSV_COL_MAPPING: Final[dict[str, str]] = {
    "sampleID": "id",
    "Kon": "sex",
    "age": "age_at_baseline",
    "lastAge": "age_at_exit",
    "status": "event",
}

EXPECTED_DTYPES: Final[dict[str, str]] = {
    "id": "object",
    "sex": "int64",
    "age_at_baseline": "float64",
    "age_at_exit": "float64",
    "event": "int64",
    **{col: "float64" for col in FEATURE_COLS},
}


def derive_dementia_outcome(event_code: int) -> int:
    """Derive binary dementia outcome from Event column code.
    
    Args:
        event_code: Event column code from scrambled_b.csv
        
    Returns:
        1 if dementia diagnosis (before or after sampling), 0 otherwise
    """
    return int(event_code in [-10, 10])

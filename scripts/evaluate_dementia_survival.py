"""Evaluate BA scores for dementia with CoxPH and Fine-Gray.

This script supports two analyses:
1) Left-truncated CoxPH on attained-age time scale:
   Surv(age_at_baseline, age_at_exit, dementia_event) ~ score + sex
2) Fine-Gray subdistribution hazards for dementia with death as competing risk:
   event code: 1=dementia, 2=death, 0=censoring

Notes:
- Dementia event uses Event == 10 (incident dementia after baseline).
- Death competing event uses Event == 20.
- All other Event codes are treated as censored.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test

from amoris_bioage.config import load_config
from amoris_bioage.data.loader import load_raw

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _load_and_merge_scores(raw_df: pd.DataFrame, score_files: list[Path]) -> pd.DataFrame:
    """Merge score columns from one or more files by id."""
    df = raw_df.copy()
    for score_file in score_files:
        score_df = pd.read_csv(score_file)
        if "id" not in score_df.columns:
            raise ValueError(f"Score file lacks 'id' column: {score_file}")
        score_cols = [c for c in score_df.columns if c != "id"]
        overlap = set(score_cols) & set(df.columns)
        if overlap:
            # Keep raw columns unchanged; suffix incoming overlaps.
            score_df = score_df.rename(columns={c: f"{c}_from_{score_file.stem}" for c in overlap})
            score_cols = [c if c not in overlap else f"{c}_from_{score_file.stem}" for c in score_cols]
        df = df.merge(score_df[["id"] + score_cols], on="id", how="left")
        logger.info("Merged scores from %s (%d columns)", score_file, len(score_cols))
    return df


def _subset_to_ids(df: pd.DataFrame, ids_file: Path) -> pd.DataFrame:
    """Restrict analysis dataframe to IDs listed in a CSV file."""
    ids_df = pd.read_csv(ids_file)
    if "id" not in ids_df.columns:
        raise ValueError(f"IDs file lacks 'id' column: {ids_file}")
    n_before = len(df)
    keep_ids = set(ids_df["id"].astype(str))
    out = df[df["id"].astype(str).isin(keep_ids)].copy()
    logger.info("Subset to IDs from %s: %d -> %d rows", ids_file, n_before, len(out))
    return out


def _cox_age_timescale(df: pd.DataFrame, score_col: str) -> dict[str, float | int | str]:
    """Fit left-truncated Cox model with attained age time scale."""
    model_df = df[
        ["age_at_baseline", "age_at_exit", "sex", "dementia_event", score_col]
    ].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    if model_df.empty:
        raise ValueError(f"No valid rows after filtering for score '{score_col}'")
    n_events = int(model_df["dementia_event"].sum())
    logger.info(
        "%s Cox input: N=%d dementia_events=%d",
        score_col,
        len(model_df),
        n_events,
    )
    if n_events == 0:
        raise ValueError(f"No dementia events available for score '{score_col}'")

    # If sex has no variation in this subset, drop it from the model.
    include_sex = model_df["sex"].nunique() > 1
    formula = f"{score_col} + sex" if include_sex else f"{score_col}"
    if not include_sex:
        logger.warning("%s: sex has no variation after filtering, fitting without sex", score_col)

    # Robust fitting: try unpenalized first, then mild ridge penalties.
    fit_ok = False
    last_exc: Exception | None = None
    for penalizer in (0.0, 1e-6, 1e-4, 1e-2):
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            cph.fit(
                model_df,
                duration_col="age_at_exit",
                event_col="dementia_event",
                entry_col="age_at_baseline",
                formula=formula,
            )
            fit_ok = True
            if penalizer > 0:
                logger.warning(
                    "%s: Cox converged with penalizer=%g", score_col, penalizer
                )
            break
        except ConvergenceError as exc:
            last_exc = exc
            logger.warning(
                "%s: Cox failed with penalizer=%g (%s)", score_col, penalizer, exc
            )
    if not fit_ok:
        raise RuntimeError(
            f"CoxPH failed to converge for '{score_col}' even with penalization."
        ) from last_exc

    row = cph.summary.loc[score_col]
    beta = float(row["coef"])
    se = float(row["se(coef)"])
    hr = float(np.exp(beta))
    lb = float(np.exp(beta - 1.96 * se))
    ub = float(np.exp(beta + 1.96 * se))
    p = float(row["p"])
    std_beta = float(beta * model_df[score_col].std())
    concordance = float(cph.concordance_index_)

    ph_p = np.nan
    ph_global_p = np.nan
    try:
        zph = proportional_hazard_test(cph, model_df, time_transform="rank")
        if score_col in zph.summary.index:
            ph_p = float(zph.summary.loc[score_col, "p"])
        if "global" in zph.summary.index:
            ph_global_p = float(zph.summary.loc["global", "p"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("PH test failed for %s: %s", score_col, exc)

    return {
        "score": score_col,
        "model": "CoxPH age-timescale sex-adjusted",
        "beta": beta,
        "se": se,
        "HR": hr,
        "lb": lb,
        "ub": ub,
        "p": p,
        "p_PH_score": ph_p,
        "global_p_PH": ph_global_p,
        "concordance": concordance,
        "N": int(len(model_df)),
        "std_beta": std_beta,
    }


def _finegray(df: pd.DataFrame, score_col: str) -> dict[str, float | int | str]:
    """Fit Fine-Gray model; death is competing event.

    Uses follow-up time (age_at_exit - age_at_baseline) as duration because
    common Python Fine-Gray implementations do not support delayed entry.
    """
    model_df = df[["age_at_baseline", "age_at_exit", "sex", "fg_event", score_col]].copy()
    model_df["followup"] = model_df["age_at_exit"] - model_df["age_at_baseline"]
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    model_df = model_df[model_df["followup"] > 0].copy()
    if model_df.empty:
        raise ValueError(f"No valid rows for Fine-Gray with score '{score_col}'")
    n_dementia = int((model_df["fg_event"] == 1).sum())
    n_death = int((model_df["fg_event"] == 2).sum())
    logger.info(
        "%s Fine-Gray input: N=%d dementia=%d death=%d",
        score_col,
        len(model_df),
        n_dementia,
        n_death,
    )
    if n_dementia == 0:
        raise ValueError(f"No dementia events available for Fine-Gray score '{score_col}'")

    # Try lifelines FineAndGray first.
    try:
        from lifelines import FineAndGrayFitter  # type: ignore

        fg = FineAndGrayFitter()
        include_sex = model_df["sex"].nunique() > 1
        try:
            if include_sex:
                fg.fit(
                    model_df,
                    duration_col="followup",
                    event_col="fg_event",
                    event_of_interest=1,
                    formula=f"{score_col} + sex",
                )
            else:
                fg.fit(
                    model_df,
                    duration_col="followup",
                    event_col="fg_event",
                    event_of_interest=1,
                    formula=f"{score_col}",
                )
        except TypeError:
            fg.fit(
                model_df,
                duration_col="followup",
                event_col="fg_event",
                event_of_interest=1,
            )

        # If formula was unsupported in this lifelines version, refit using only
        # the required columns and rely on default covariate inclusion.
        if score_col not in fg.summary.index:
            base_cols = ["followup", "fg_event", score_col]
            if include_sex:
                base_cols.append("sex")
            fg_df = model_df[base_cols].copy()
            fg.fit(
                fg_df,
                duration_col="followup",
                event_col="fg_event",
                event_of_interest=1,
            )

        row = fg.summary.loc[score_col]
        beta = float(row["coef"])
        se = float(row["se(coef)"])
        hr = float(np.exp(beta))
        p = float(row["p"])
        lb = float(np.exp(beta - 1.96 * se))
        ub = float(np.exp(beta + 1.96 * se))
        std_beta = float(beta * model_df[score_col].std())

        return {
            "score": score_col,
            "model": "FineGray subdistribution sex-adjusted",
            "beta": beta,
            "se": se,
            "HR": hr,
            "lb": lb,
            "ub": ub,
            "p": p,
            "p_PH_score": np.nan,
            "global_p_PH": np.nan,
            "concordance": np.nan,
            "N": int(len(model_df)),
            "std_beta": std_beta,
        }
    except ImportError as exc:
        raise RuntimeError(
            "FineAndGrayFitter is unavailable in lifelines. "
            "Please update lifelines or run Fine-Gray in R."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--score-files",
        nargs="*",
        default=[],
        help="Optional CSV files with id + score columns to merge before modeling",
    )
    parser.add_argument(
        "--score-cols",
        nargs="+",
        required=True,
        help="Score columns to evaluate (must exist after merging)",
    )
    parser.add_argument(
        "--ids-file",
        default=None,
        help="Optional CSV with id column to restrict analysis set (e.g., outputs/derived/test.csv)",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/results/dementia_survival_metrics.csv",
        help="Output CSV path for model metrics",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/results/dementia_survival_metrics.json",
        help="Output JSON path for model metrics",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_df = load_raw(cfg.data.raw_path)
    logger.info("Loaded raw cohort for dementia analysis: N=%d", len(raw_df))

    # Dementia/competing-event coding from Event:
    # 10 = dementia, 20 = death, else censored.
    if "Event" not in raw_df.columns:
        raise ValueError("Expected raw 'Event' column for dementia/death coding.")
    raw_df["dementia_event"] = (raw_df["Event"] == 10).astype(int)
    raw_df["fg_event"] = np.select(
        [raw_df["Event"] == 10, raw_df["Event"] == 20],
        [1, 2],
        default=0,
    ).astype(int)
    logger.info(
        "Raw dementia/death coding counts: dementia=%d death=%d censored=%d",
        int((raw_df["fg_event"] == 1).sum()),
        int((raw_df["fg_event"] == 2).sum()),
        int((raw_df["fg_event"] == 0).sum()),
    )

    score_files = [Path(p) for p in args.score_files]
    df = _load_and_merge_scores(raw_df, score_files) if score_files else raw_df.copy()
    if args.ids_file:
        df = _subset_to_ids(df, Path(args.ids_file))
    logger.info(
        "Analysis-set dementia/death counts: dementia=%d death=%d censored=%d",
        int((df["fg_event"] == 1).sum()),
        int((df["fg_event"] == 2).sum()),
        int((df["fg_event"] == 0).sum()),
    )

    missing_scores = [c for c in args.score_cols if c not in df.columns]
    if missing_scores:
        raise ValueError(f"Missing score column(s): {missing_scores}")

    rows: list[dict[str, float | int | str]] = []
    for score_col in args.score_cols:
        logger.info("Running dementia models for score: %s", score_col)
        try:
            rows.append(_cox_age_timescale(df, score_col))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cox failed for %s: %s", score_col, exc)
            rows.append(
                {
                    "score": score_col,
                    "model": "CoxPH age-timescale sex-adjusted",
                    "beta": np.nan,
                    "se": np.nan,
                    "HR": np.nan,
                    "lb": np.nan,
                    "ub": np.nan,
                    "p": np.nan,
                    "p_PH_score": np.nan,
                    "global_p_PH": np.nan,
                    "concordance": np.nan,
                    "N": 0,
                    "std_beta": np.nan,
                    "error": str(exc),
                }
            )
        try:
            rows.append(_finegray(df, score_col))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fine-Gray failed for %s: %s", score_col, exc)
            rows.append(
                {
                    "score": score_col,
                    "model": "FineGray subdistribution sex-adjusted",
                    "beta": np.nan,
                    "se": np.nan,
                    "HR": np.nan,
                    "lb": np.nan,
                    "ub": np.nan,
                    "p": np.nan,
                    "p_PH_score": np.nan,
                    "global_p_PH": np.nan,
                    "concordance": np.nan,
                    "N": 0,
                    "std_beta": np.nan,
                    "error": str(exc),
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Saved dementia survival metrics to %s and %s", out_csv, out_json)


if __name__ == "__main__":
    main()

"""SHAP and integrated-gradients attribution for biological age and age gap."""
from amoris_bioage.attribution.background import make_age_stratified_background
from amoris_bioage.attribution.ig_explainer import CoxIGExplainer
from amoris_bioage.attribution.shap_explainer import BioageShapExplainer

__all__ = [
    "make_age_stratified_background",
    "BioageShapExplainer",
    "CoxIGExplainer",
]

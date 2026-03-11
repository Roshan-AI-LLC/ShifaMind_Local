from benchmark.models.caml import CAML
from benchmark.models.laat import LAAT
from benchmark.models.plm_icd import PLMICD
from benchmark.models.msmn import MSMN
from benchmark.models.vanilla_cbm import VanillaCBM

MODEL_REGISTRY = {
    "CAML":      CAML,
    "LAAT":      LAAT,
    "PLMICD":    PLMICD,
    "MSMN":      MSMN,
    "VanillaCBM": VanillaCBM,
}

__all__ = ["CAML", "LAAT", "PLMICD", "MSMN", "VanillaCBM", "MODEL_REGISTRY"]

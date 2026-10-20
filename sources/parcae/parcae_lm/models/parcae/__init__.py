from parcae_lm.models.parcae.config import ParcaeConfig
from parcae_lm.models.parcae.init import ParcaeInit


def __getattr__(name):
    if name == "Parcae":
        from parcae_lm.models.parcae.parcae import Parcae
        return Parcae
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ParcaeConfig", "ParcaeInit", "Parcae"]

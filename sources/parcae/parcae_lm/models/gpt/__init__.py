from parcae_lm.models.gpt.config import GPTConfig
from parcae_lm.models.gpt.init import GPTInit


def __getattr__(name):
    if name == "GPT":
        from parcae_lm.models.gpt.gpt import GPT
        return GPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GPTConfig", "GPT", "GPTInit"]

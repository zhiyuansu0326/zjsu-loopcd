from dataclasses import dataclass
import torch
from parcae_lm.models.config import Config
from parcae_lm.models.gpt.init import GPTInit

@dataclass
class GPTConfig(Config):
    n_layer: int = 16
    tie_embeddings: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.init = GPTInit(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            self.n_layer,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from parcae_lm.models.gpt import GPT
        return GPT(self, **kwargs)
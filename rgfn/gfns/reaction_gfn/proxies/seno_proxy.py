import warnings
from typing import List

import gin
import torch
from wurlitzer import pipes

from rgfn.api.env_base import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase

try:
    from gneprop.rewards import load_model, predict
except ImportError:
    warnings.warn("GNEprop not found, senolytic proxy will be disabled.")


@gin.configurable()
class SenoProxy(CachedProxyBase[ReactionState]):
    def __init__(self, checkpoint_path: str, batch_size: int = 128):
        super().__init__()
        self.device = "cpu"
        self.batch_size = batch_size
        self.model = load_model(checkpoint_path)
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    @torch.no_grad()
    def _compute_proxy_output(self, states: List[ReactionState]) -> List[float]:
        smiles = [state.molecule.smiles for state in states]

        with pipes():
            scores = (
                predict(
                    self.model,
                    smiles,
                    batch_size=self.batch_size,
                    gpus=(0 if self.device == torch.device("cpu") else 1),
                )
                * 100
            ).tolist()

        return scores

    def set_device(self, device: str):
        self.device = device

import abc
from typing import List

import gin
from tdc import Oracle

from rgfn.api.env_base import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class DRD2Proxy(CachedProxyBase[ReactionState], abc.ABC):
    def __init__(self):
        super().__init__()

        self.oracle = Oracle(name="DRD2")
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        return self.oracle([state.molecule.smiles for state in states])

import abc
from typing import List

import gin
from rdkit.Chem.QED import qed

from rgfn.api.env_base import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class QEDProxy(CachedProxyBase[ReactionState], abc.ABC):
    def __init__(self):
        super().__init__()

        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        return [qed(state.molecule.rdkit_mol) for state in states]

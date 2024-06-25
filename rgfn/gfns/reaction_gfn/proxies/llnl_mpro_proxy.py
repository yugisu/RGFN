from typing import List

import gin

from rgfn.api.env_base import TState
from rgfn.gfns.reaction_gfn.proxies.gneprop_proxy import GNEpropProxy


@gin.configurable()
class LLNLMproProxy(GNEpropProxy):
    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        return (-1.0 * self._compute_gneprop_output(states) / 100.0).clip(1e-6, 1.0).tolist()

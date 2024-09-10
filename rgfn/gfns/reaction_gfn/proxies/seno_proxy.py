from typing import List

import gin

from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.proxies.gneprop_proxy import GNEpropProxy


@gin.configurable()
class SenoProxy(GNEpropProxy):
    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        return self._compute_gneprop_output(states).clip(1e-6, 1.0).tolist()

import abc
import sys
from typing import List

import numpy.typing as npt
import torch
from wurlitzer import pipes

from rgfn import ROOT_DIR
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase

GNEPROP_PATH = ROOT_DIR / "external" / "gneprop"


class GNEpropProxy(CachedProxyBase[ReactionState], abc.ABC):
    def __init__(self, checkpoint_path: str, batch_size: int = 128):
        super().__init__()

        sys.path.append(str(GNEPROP_PATH))

        from gneprop.rewards import load_model

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
    def _compute_gneprop_output(self, states: List[TState]) -> npt.NDArray:
        from gneprop.rewards import predict

        smiles = [state.molecule.smiles for state in states]

        with pipes():
            scores = predict(
                self.model,
                smiles,
                batch_size=self.batch_size,
                gpus=(0 if self.device == torch.device("cpu") else 1),
            )

        return scores

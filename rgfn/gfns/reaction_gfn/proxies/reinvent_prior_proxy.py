import sys
from typing import Dict, List

import gin
import torch

from rgfn import ROOT_DIR
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase


@gin.configurable()
class ReinventPriorProxy(CachedProxyBase[ReactionState]):
    def __init__(self, prior_name: str = "reinvent", temperature: float = 0.01):
        super().__init__()

        sys.path.append(str(ROOT_DIR / "external/reinvent"))

        from reinvent.chemistry.standardization.rdkit_standardizer import (
            RDKitStandardizer,
            logger,
        )
        from reinvent.models.reinvent.utils import collate_fn
        from reinvent.runmodes import create_adapter

        logger.disabled = True

        prior_path = ROOT_DIR / f"external/reinvent/priors/{prior_name}.prior"
        adapter, _, model_type = create_adapter(prior_path, "inference", "cpu")
        self.prior = adapter.model
        self.standardizer = RDKitStandardizer([])
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}
        self.collate_fn = collate_fn
        self.temperature = temperature

    def set_device(self, device: str):
        self.device = device
        self.prior.device = device
        self.prior.network.to(device)

    def compute_nll(self, smiles_list: List[str]) -> torch.Tensor:
        tokens_list = [self.prior.tokenizer.tokenize(smiles) for smiles in smiles_list]
        encoded_list = [self.prior.vocabulary.encode(tokens) for tokens in tokens_list]

        sequences = [torch.tensor(encoded, dtype=torch.long) for encoded in encoded_list]
        padded_sequences = self.collate_fn(sequences).to(self.device)

        size = (
            self.prior.network._num_layers,
            padded_sequences.size(0),
            self.prior.network._layer_size,
        )
        hidden_state = [torch.zeros(*size).to(self.device), torch.zeros(*size).to(self.device)]
        logits, _ = self.prior.network(
            padded_sequences[:, :-1], hidden_state
        )  # all steps done at once
        log_probs = logits.log_softmax(dim=2)

        return self.prior._nll_loss(log_probs.transpose(1, 2), padded_sequences[:, 1:]).sum(dim=1)

    def _compute_proxy_output(
        self, states: List[ReactionStateTerminal]
    ) -> List[Dict[str, float]] | List[float]:
        smiles_list = [self.standardizer.apply_filter(state.molecule.smiles) for state in states]
        valid_smiles_indices = [i for i, smiles in enumerate(smiles_list) if smiles is not None]
        valid_smiles = [smiles_list[i] for i in valid_smiles_indices]
        probs = [0.0] * len(states)
        if len(valid_smiles) > 0:
            valid_nll = self.compute_nll(valid_smiles)
            valid_prob = torch.exp(-valid_nll * self.temperature).tolist()
            for i, prob in zip(valid_smiles_indices, valid_prob):
                probs[i] = prob
        return probs

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

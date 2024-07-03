from typing import Dict, List

import gin
import torch
from torch import Tensor

from rgfn.api.proxy_base import ProxyBase, ProxyOutput
from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionState, ReactionStateTerminal
from rgfn.gfns.reaction_gfn.policies.graph_transformer import (
    GraphTransformer,
    mol2graph,
    mols2batch,
)


@gin.configurable()
class RNDNoveltyProxy(ProxyBase[ReactionState]):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        lr: float = 0.001,
        optimizer_cls: str = "Adam",
    ):
        super().__init__()

        self.conditional_dim = 0

        def _make_network():
            return GraphTransformer(
                x_dim=71,
                e_dim=4,
                g_dim=self.conditional_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                num_emb=hidden_dim,
            )

        def _make_mlp():
            return torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        self.random_target_network = _make_network()
        self.random_target_mlp = _make_mlp()
        self.predictor_network = _make_network()
        self.predictor_mlp = _make_mlp()

        for param in self.random_target_network.parameters():
            param.requires_grad = False

        for param in self.random_target_mlp.parameters():
            param.requires_grad = False

        if optimizer_cls == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_cls == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer class: {optimizer_cls}")

        self.device = "cpu"
        self.last_update_idx = -1

    def parameters(self):
        yield from self.predictor_network.parameters()
        yield from self.predictor_mlp.parameters()

    def compute_novelty(self, states: List[ReactionStateTerminal]) -> Tensor:
        graphs = [mol2graph(state.molecule.rdkit_mol) for state in states]

        graph_batch = mols2batch(graphs).to(self.device)
        cond_batch = torch.zeros(
            size=(len(states), self.conditional_dim), device=self.device, dtype=torch.float
        )

        target_x = self.random_target_network.forward(graph_batch, cond_batch)
        target_x = self.random_target_mlp(target_x)
        predicted_x = self.predictor_network.forward(graph_batch, cond_batch)
        predicted_x = self.predictor_mlp(predicted_x)
        return torch.norm(predicted_x - target_x.detach(), dim=-1, p=2)

    def set_device(self, device: str):
        self.device = device
        self.random_target_network.to(device)
        self.predictor_network.to(device)
        self.random_target_mlp.to(device)
        self.predictor_mlp.to(device)

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_proxy_output(self, states: List[ReactionState]) -> ProxyOutput:
        valid_states_mask = [isinstance(state, ReactionStateTerminal) for state in states]
        valid_states = [state for state in states if isinstance(state, ReactionStateTerminal)]
        novelty = torch.zeros(len(states), device=self.device, dtype=torch.float)
        if len(valid_states) > 0:
            novelty[valid_states_mask] = self.compute_novelty(valid_states)

        return ProxyOutput(value=novelty, components=None)

    def update_using_trajectories(
        self, trajectories: Trajectories, update_idx: int
    ) -> Dict[str, float]:
        if update_idx == self.last_update_idx:
            return {}

        self.last_update_idx = update_idx

        self.optimizer.zero_grad()

        states = trajectories.get_last_states_flat()
        states = [state for state in states if isinstance(state, ReactionStateTerminal)]
        novelty = self.compute_novelty(states)
        loss = novelty.mean()

        loss.backward()
        self.optimizer.step()

        return {"novelty_loss": loss.item()}

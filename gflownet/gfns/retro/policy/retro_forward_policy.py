from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Type, Union

import gin
import numpy as np
import torch
from dgllife.utils import WeaveAtomFeaturizer
from torch import nn
from torch.distributions import Categorical
from torch_geometric.utils import to_dense_batch
from torchtyping import TensorType

from gflownet.api.env_base import TAction, TActionSpace, TState
from gflownet.api.policy_base import PolicyBase
from gflownet.gfns.retro.api.retro_api import (
    EarlyTerminalRetroState,
    EarlyTerminateRetroAction,
    EarlyTerminateRetroActionSpace,
    FirstPhaseRetroAction,
    FirstPhaseRetroActionSpace,
    FirstPhaseRetroState,
    MappingTuple,
    Molecule,
    Pattern,
    RetroAction,
    RetroActionSpace,
    RetroState,
    SecondPhaseRetroAction,
    SecondPhaseRetroActionSpace,
    SecondPhaseRetroState,
    ThirdPhaseRetroAction,
    ThirdPhaseRetroActionSpace,
    ThirdPhaseRetroState,
)
from gflownet.gfns.retro.api.retro_data_factory import RetroDataFactory
from gflownet.gfns.retro.policy.featurizers import (
    ATOM_TYPES,
    JointFeaturizer,
    RandomWalkPEFeaturizer,
    ReactantNodeFeaturizer,
)
from gflownet.gfns.retro.policy.gnns import EmbeddingGNN, GlobalReactivityAttention
from gflownet.utils.helpers import to_indices


@gin.configurable()
class RetroForwardPolicy(PolicyBase[RetroState, RetroActionSpace, RetroAction], nn.Module):
    def __init__(
        self,
        data_factory: RetroDataFactory,
        hidden_dim: int = 256,
        num_gnn_layers: int = 4,
        num_gnn_layers_third_phase: int = 3,
        num_attention_heads: int = 8,
        initial_logZ: float = 0.0,
        temperature: float = 1.0,
        random_walk_steps: int = 16,
        z_dim: int = 1,
    ):
        super().__init__()
        self.num_reactants_patterns = len(data_factory.get_reactant_patterns())
        self.num_product_patterns = len(data_factory.get_product_patterns())
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.product_embedding = EmbeddingGNN(
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            num_attention_heads=num_attention_heads,
            cache=True,
            use_attention=True,
            node_featurizer=JointFeaturizer(
                atom_featurizer=WeaveAtomFeaturizer(atom_types=ATOM_TYPES),
                pe_featurizer=RandomWalkPEFeaturizer(n_steps=random_walk_steps),
            ),
        )

        # First phase
        self.first_phase_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Second phase
        self.start_idx = self.num_reactants_patterns
        self.padding_idx = self.num_reactants_patterns + 1
        self.second_phase_reactant_idx_embedding = nn.Embedding(
            self.num_reactants_patterns + 2, hidden_dim, padding_idx=self.padding_idx
        )
        self.second_phase_product_pattern_idx_embedding = nn.Embedding(
            self.num_product_patterns, hidden_dim
        )
        self.second_phase_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_reactants_patterns),
        )

        # Third phase
        self.third_phase_pattern_embedding = EmbeddingGNN(
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers_third_phase,
            num_attention_heads=num_attention_heads,
            cache=True,
            use_attention=True,
            node_featurizer=JointFeaturizer(
                atom_featurizer=ReactantNodeFeaturizer(),
                pe_featurizer=RandomWalkPEFeaturizer(n_steps=random_walk_steps),
            ),
        )

        self.third_phase_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # log flow
        self.product_to_log_flow_idx: Dict[Molecule, int] = {}
        product_logZ = torch.empty(
            (len(data_factory.get_products()) + 10_000, z_dim), dtype=torch.float
        ).fill_(initial_logZ)
        product_logZ.requires_grad = True
        self.product_logZ = nn.Parameter(product_logZ)

        # forward dict
        self.action_space_type_to_forward_fn: Dict[Any, Callable] = {
            FirstPhaseRetroActionSpace: self._forward_first_phase,
            SecondPhaseRetroActionSpace: self._forward_second_phase,
            ThirdPhaseRetroActionSpace: self._forward_third_phase,
            EarlyTerminateRetroActionSpace: self._forward_early_terminate,
        }

        self.device = "cpu"

    def _embed_products(
        self, products: List[Molecule]
    ) -> Tuple[TensorType[float], Dict[Molecule, int]]:
        all_products = set(products)
        product_to_idx = {product: idx for idx, product in enumerate(all_products)}
        product_embedding = self.product_embedding.forward(list(all_products))
        return product_embedding, product_to_idx

    def sample_actions(
        self, states: List[RetroState], action_spaces: List[RetroActionSpace]
    ) -> List[RetroAction]:
        product_embedding, product_to_idx = self._embed_products(
            [state.product for state in states if not isinstance(state, EarlyTerminalRetroState)]
        )

        def _sample_phase_actions(
            action_space_type: Type[RetroActionSpace],
        ) -> Tuple[List[int], List[RetroAction]]:
            phase_indices = [
                idx
                for idx, action_space in enumerate(action_spaces)
                if isinstance(action_space, action_space_type)
            ]
            if len(phase_indices) == 0:
                return [], []
            phase_states = [states[idx] for idx in phase_indices]
            phase_action_spaces = [action_spaces[idx] for idx in phase_indices]
            log_probs = self.action_space_type_to_forward_fn[action_space_type](
                phase_states, phase_action_spaces, product_embedding, product_to_idx
            )
            phase_actions = self._sample_actions_from_log_probs(log_probs, phase_action_spaces)
            return phase_indices, phase_actions

        actions = []
        action_to_state_idx = []
        for action_space_type in self.action_space_type_to_forward_fn.keys():
            phase_indices, phase_actions = _sample_phase_actions(action_space_type)
            actions.extend(phase_actions)
            action_to_state_idx.extend(phase_indices)

        state_to_action_idx = [0] * len(states)
        for action_idx, state_idx in enumerate(action_to_state_idx):
            state_to_action_idx[state_idx] = action_idx

        return [actions[state_to_action_idx[state_idx]] for state_idx in range(len(states))]

    def _sample_actions_from_log_probs(
        self, log_probs: TensorType[float], action_spaces: List[RetroActionSpace]
    ) -> List[RetroAction]:
        action_indices = Categorical(probs=torch.exp(log_probs)).sample()
        return [
            action_space.get_action_at_idx(idx.item())
            for action_space, idx in zip(action_spaces, action_indices)
        ]

    def compute_action_log_probs(
        self,
        states: List[RetroState],
        action_spaces: List[RetroActionSpace],
        actions: List[RetroAction],
    ) -> TensorType[float]:
        product_embedding, product_to_idx = self._embed_products(
            [state.product for state in states if not isinstance(state, EarlyTerminalRetroState)]
        )

        def _compute_phase_log_probs(
            action_space_type: Type[RetroActionSpace],
        ) -> Tuple[List[int], TensorType[float]]:
            phase_indices = [
                idx
                for idx, action_space in enumerate(action_spaces)
                if isinstance(action_space, action_space_type)
            ]
            if len(phase_indices) == 0:
                return [], torch.tensor([], dtype=torch.float).to(self.device)

            phase_states = [states[idx] for idx in phase_indices]
            phase_action_spaces = [action_spaces[idx] for idx in phase_indices]
            phase_actions = [actions[idx] for idx in phase_indices]
            log_probs = self.action_space_type_to_forward_fn[action_space_type](
                phase_states, phase_action_spaces, product_embedding, product_to_idx
            )
            phase_log_probs = self._select_actions_log_probs(
                log_probs, phase_action_spaces, phase_actions
            )
            return phase_indices, phase_log_probs

        log_probs_list = []
        log_probs_to_state_idx = []
        for action_space_type in self.action_space_type_to_forward_fn.keys():
            phase_indices, phase_log_probs = _compute_phase_log_probs(action_space_type)
            log_probs_list.append(phase_log_probs)
            log_probs_to_state_idx.extend(phase_indices)

        log_probs = torch.cat(log_probs_list, dim=0)
        state_to_action_idx = torch.empty(len(states), dtype=torch.long)
        for action_idx, state_idx in enumerate(log_probs_to_state_idx):
            state_to_action_idx[state_idx] = action_idx
        state_to_action_idx = state_to_action_idx.to(self.device)

        return torch.index_select(log_probs, index=state_to_action_idx, dim=0).to(self.device)

    def _select_actions_log_probs(
        self,
        log_probs: TensorType[float],
        action_spaces: Sequence[RetroActionSpace],
        actions: Sequence[RetroAction],
    ) -> TensorType[float]:
        action_indices = [
            action_space.get_idx_of_action(action)  # type: ignore
            for action_space, action in zip(action_spaces, actions)
        ]
        max_num_actions = log_probs.shape[1]
        action_indices = [
            idx * max_num_actions + action_idx for idx, action_idx in enumerate(action_indices)
        ]
        action_tensor_indices = torch.tensor(action_indices).long().to(self.device)
        log_probs = torch.index_select(log_probs.view(-1), index=action_tensor_indices, dim=0)
        return log_probs

    def _forward_first_phase(
        self,
        states: List[FirstPhaseRetroState],
        action_spaces: List[FirstPhaseRetroActionSpace],
        product_embeddings: TensorType[float],
        product_to_idx: Dict[Molecule, int],
    ) -> TensorType[float]:
        # subgraph mask
        all_possible_actions = [
            action for action_space in action_spaces for action in action_space.possible_actions
        ]
        subgraphs_mask = torch.zeros(
            (len(all_possible_actions), product_embeddings.shape[1]), dtype=torch.bool
        )
        for idx, action in enumerate(all_possible_actions):
            subgraphs_mask[idx, action.subgraph_idx] = True
        subgraphs_mask = subgraphs_mask.to(self.device)

        # embedding from the global pool
        embeddings_indices = (
            torch.tensor([product_to_idx[state.product] for state in states]).long().to(self.device)
        )
        possible_actions_count = (
            torch.tensor([len(action_space.possible_actions) for action_space in action_spaces])
            .long()
            .to(self.device)
        )
        embeddings = torch.index_select(product_embeddings, index=embeddings_indices, dim=0)
        embeddings = torch.repeat_interleave(embeddings, possible_actions_count, dim=0)
        subgraph_embeddings = torch.masked_fill(embeddings, ~subgraphs_mask.unsqueeze(-1), 0.0)
        subgraph_embeddings = torch.sum(subgraph_embeddings, dim=1)

        score = self.first_phase_mlp(subgraph_embeddings).squeeze(-1)
        graph_indices = to_indices(possible_actions_count)
        score, _ = to_dense_batch(score, batch=graph_indices, fill_value=-float("inf"))
        log_probs = torch.log_softmax(score * self.temperature, dim=-1)
        return log_probs

    def _forward_second_phase(
        self,
        states: List[SecondPhaseRetroState],
        action_spaces: List[SecondPhaseRetroActionSpace],
        product_embeddings: TensorType[float],
        product_to_idx: Dict[Molecule, int],
    ) -> TensorType[float]:
        max_len = max([len(state.reactant_patterns) for state in states])
        reactant_indices = [
            [pattern.idx for pattern in state.reactant_patterns] for state in states
        ]
        reactant_indices = [
            [self.start_idx] + indices + [self.padding_idx] * (max_len - len(indices))
            for indices in reactant_indices
        ]
        reactant_indices = torch.tensor(reactant_indices).long().to(self.device)
        reactant_embeddings = self.second_phase_reactant_idx_embedding(reactant_indices)
        reactant_embeddings = torch.sum(reactant_embeddings, dim=1)

        product_pattern_indices = [state.product_pattern.idx for state in states]
        product_pattern_indices = torch.tensor(product_pattern_indices).long().to(self.device)
        product_pattern_embeddings = self.second_phase_product_pattern_idx_embedding(
            product_pattern_indices
        )

        # subgraph mask
        subgraph_mask = torch.zeros((len(states), product_embeddings.shape[1]), dtype=torch.bool)
        for idx, state in enumerate(states):
            subgraph_mask[idx, state.subgraph_idx] = True
        subgraph_mask = subgraph_mask.to(self.device)

        # embedding from the global pool
        embeddings_indices = (
            torch.tensor([product_to_idx[state.product] for state in states]).long().to(self.device)
        )
        embeddings = torch.index_select(product_embeddings, index=embeddings_indices, dim=0)

        # graphs and subgraphs embeddings
        subgraph_embeddings = torch.masked_fill(embeddings, ~subgraph_mask.unsqueeze(-1), 0.0)
        subgraph_embeddings = torch.sum(subgraph_embeddings, dim=1)

        # final score
        action_mask = torch.tensor(
            [action_space.actions_mask for action_space in action_spaces], dtype=torch.bool
        ).to(self.device)
        state_embeddings = torch.cat(
            [subgraph_embeddings, product_pattern_embeddings, reactant_embeddings], dim=-1
        )

        score = self.second_phase_mlp(state_embeddings)
        score = torch.masked_fill(score, ~action_mask, -float("inf"))
        log_probs = torch.log_softmax(score * self.temperature, dim=-1)
        return log_probs

    def _forward_third_phase(
        self,
        states: List[ThirdPhaseRetroState],
        action_spaces: List[ThirdPhaseRetroActionSpace],
        product_embeddings: TensorType[float],
        product_to_idx: Dict[Molecule, int],
    ) -> TensorType[float]:
        all_patterns = list(set(pattern for state in states for pattern in state.reactant_patterns))
        pattern_to_idx = {pattern: idx for idx, pattern in enumerate(all_patterns)}
        pattern_embeddings = self.third_phase_pattern_embedding(all_patterns)

        def _get_indices(
            state: ThirdPhaseRetroState, atom_mappings: Iterable[MappingTuple]
        ) -> List[Tuple[int, int]]:
            indices = []
            for mapping in atom_mappings:
                product_embedding_idx = product_to_idx[state.product]
                product_node_idx = state.subgraph_idx[mapping.product_node]
                product_node_embedding_idx = (
                    product_embedding_idx * product_embeddings.shape[1] + product_node_idx
                )

                pattern = state.reactant_patterns[mapping.reactant]
                pattern_embedding_idx = pattern_to_idx[pattern]
                pattern_node_idx = mapping.reactant_node
                pattern_node_embedding_idx = (
                    pattern_embedding_idx * pattern_embeddings.shape[1] + pattern_node_idx
                )
                indices.append((product_node_embedding_idx, pattern_node_embedding_idx))
            return indices

        indices_list = [
            _get_indices(state, action_space.possible_actions)
            for state, action_space in zip(states, action_spaces)
        ]
        indices_flatten = (
            torch.tensor([idx for indices in indices_list for idx in indices])
            .long()
            .to(self.device)
        )

        product_embeddings = product_embeddings.view(
            product_embeddings.shape[0] * product_embeddings.shape[1], -1
        )
        product_nodes_embeddings = torch.index_select(
            product_embeddings, index=indices_flatten[:, 0], dim=0
        )
        pattern_embeddings = pattern_embeddings.view(
            pattern_embeddings.shape[0] * pattern_embeddings.shape[1], -1
        )
        reactant_nodes_embeddings = torch.index_select(
            pattern_embeddings, index=indices_flatten[:, 1], dim=0
        )

        embeddings = torch.cat([product_nodes_embeddings, reactant_nodes_embeddings], dim=-1)
        scores = self.third_phase_mlp(embeddings).squeeze(-1)
        indices_list_sizes = (
            torch.tensor([len(indices) for indices in indices_list]).long().to(self.device)
        )
        indices_batch = to_indices(indices_list_sizes)
        scores, _ = to_dense_batch(scores, batch=indices_batch, fill_value=-float("inf"))
        log_probs = torch.log_softmax(scores, dim=-1)
        return log_probs

    def _forward_early_terminate(
        self,
        states: List[RetroState],
        action_spaces: List[RetroActionSpace],
        product_embeddings: TensorType[float],
        product_to_idx: Dict[Molecule, int],
    ) -> TensorType[float]:
        return torch.zeros((len(states), 1), device=self.device, dtype=torch.float32)

    def compute_states_log_flow(self, states: List[RetroState]) -> TensorType[float]:
        product_indices = [self._get_log_flow_idx(state.product) for state in states]  # type: ignore
        product_indices = torch.tensor(product_indices, dtype=torch.long).to(self.device)
        log_flow = torch.index_select(self.product_logZ, index=product_indices, dim=0)
        return log_flow.sum(-1)

    def _get_log_flow_idx(self, product: Molecule):
        if product not in self.product_to_log_flow_idx:
            self.product_to_log_flow_idx[product] = len(self.product_to_log_flow_idx)
        return self.product_to_log_flow_idx[product]

    def set_device(self, device: str):
        self.to(device)
        self.product_embedding.set_device(device)
        self.third_phase_pattern_embedding.set_device(device)
        self.device = device

    def clear_sampling_cache(self) -> None:
        pass

    def clear_action_embedding_cache(self) -> None:
        pass

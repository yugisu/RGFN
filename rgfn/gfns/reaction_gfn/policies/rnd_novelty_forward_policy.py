from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple, Type

import gin
import torch
from rdkit import Chem
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import Parameter
from torchtyping import TensorType

from rgfn.api.env_base import TAction, TActionSpace, TState
from rgfn.api.policy_base import PolicyBase
from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule, Reaction
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
    ReactionActionA,
    ReactionActionEarlyTerminate,
    ReactionActionSpace,
    ReactionActionSpace0,
    ReactionActionSpace0Invalid,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceC,
    ReactionActionSpaceEarlyTerminate,
    ReactionState,
    ReactionState0,
    ReactionStateA,
    ReactionStateB,
    ReactionStateC,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory
from rgfn.gfns.reaction_gfn.policies.action_embeddings import (
    ActionEmbeddingBase,
    FragmentOneHotEmbedding,
    ReactionsOneHotEmbedding,
)
from rgfn.gfns.reaction_gfn.policies.graph_transformer import (
    GraphTransformer,
    mol2graph,
    mols2batch,
)
from rgfn.gfns.reaction_gfn.policies.utils import one_hot, to_dense_embeddings
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace


@dataclass(frozen=True)
class SharedEmbeddings:
    molecule_to_idx: Dict[Molecule, int]
    molecule_reaction_to_idx: Dict[Tuple[Molecule, Reaction], int]
    all_target_embeddings: TensorType[float]
    all_predictor_embeddings: TensorType[float]


@gin.configurable()
class RNDNoveltyForwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, SharedEmbeddings],
    nn.Module,
):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 5,
        lr: float = 0.001,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.reactions = data_factory.get_reactions()
        self.num_a_actions = len(self.reactions) + 1
        self.num_b_actions = len(data_factory.get_fragments())

        def _make_gnn():
            return GraphTransformer(
                x_dim=71,
                e_dim=4,
                g_dim=len(self.reactions),
                num_layers=num_layers,
                num_heads=num_heads,
                num_emb=hidden_dim,
            )

        def _make_mlp(input_dim: int):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.target_gnn = _make_gnn()
        self.predictor_gnn = _make_gnn()

        self.target_action_embedding_a = ReactionsOneHotEmbedding(data_factory, hidden_dim)
        self.predictor_action_embedding_a = ReactionsOneHotEmbedding(data_factory, hidden_dim)
        self.target_mlp_a = _make_mlp(2 * hidden_dim)
        self.predictor_mlp_a = _make_mlp(2 * hidden_dim)

        self.target_action_embedding_b = FragmentOneHotEmbedding(data_factory, hidden_dim)
        self.predictor_action_embedding_b = FragmentOneHotEmbedding(data_factory, hidden_dim)
        self.target_mlp_b = _make_mlp(2 * hidden_dim)
        self.predictor_mlp_b = _make_mlp(2 * hidden_dim)

        self.target_mlp_c = _make_mlp(hidden_dim)
        self.predictor_mlp_c = _make_mlp(hidden_dim)

        self._action_space_type_to_forward_fn = {
            ReactionActionSpace0: self._forward_0,
            ReactionActionSpaceA: self._forward_a,
            ReactionActionSpaceB: self._forward_b,
            ReactionActionSpaceC: self._forward_c,
            ReactionActionSpaceEarlyTerminate: self._forward_early_terminate,
            ReactionActionSpace0Invalid: self._forward_early_terminate,
        }

        for network in [
            self.target_gnn,
            self.target_mlp_a,
            self.target_action_embedding_a,
            self.target_mlp_b,
            self.target_action_embedding_b,
            self.target_mlp_c,
        ]:
            for param in network.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self._device = "cpu"
        self.last_update_idx = -1
        self.temperature = temperature

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from self.predictor_gnn.parameters()
        yield from self.predictor_action_embedding_a.parameters()
        yield from self.predictor_action_embedding_b.parameters()
        yield from self.predictor_mlp_a.parameters()
        yield from self.predictor_mlp_b.parameters()
        yield from self.predictor_mlp_c.parameters()

    @property
    def action_space_to_forward_fn(
        self,
    ) -> Dict[
        Type[TIndexedActionSpace],
        Callable[[List[TState], List[TIndexedActionSpace], TSharedEmbeddings], TensorType[float]],
    ]:
        return self._action_space_type_to_forward_fn

    @property
    def device(self) -> str:
        return self._device

    def set_device(self, device: str):
        self.to(device)
        self._device = device

    def _forward_0(
        self,
        states: List[ReactionState0],
        action_spaces: List[ReactionActionSpace0],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_idx = shared_embeddings.molecule_to_idx[None]
        index = torch.tensor([embedding_idx] * self.num_b_actions).long().to(self.device)

        def _get_embeddings(all_state_embeddings: Tensor, all_action_embeddings: Tensor):
            state_embeddings = torch.index_select(
                all_state_embeddings,
                index=index,
                dim=0,
            )  # (num_fragments, hidden_dim)
            return torch.cat(
                [state_embeddings, all_action_embeddings], dim=-1
            )  # (num_fragments, 2 * hidden_dim)

        target_embedding = _get_embeddings(
            shared_embeddings.all_target_embeddings, self.target_action_embedding_b.get_embeddings()
        )
        predictor_embedding = _get_embeddings(
            shared_embeddings.all_predictor_embeddings,
            self.predictor_action_embedding_b.get_embeddings(),
        )

        target_embedding = self.target_mlp_b(target_embedding)  # (num_fragments, hidden_dim)
        predictor_embedding = self.predictor_mlp_b(
            predictor_embedding
        )  # (num_fragments, hidden_dim)

        logits = torch.norm(
            predictor_embedding - target_embedding.detach(), dim=-1, p=2
        )  # (num_fragments)
        logits = logits.repeat(len(states), 1)  # (n_states, num_fragments)
        return logits

    def _forward_a(
        self,
        states: List[ReactionStateA],
        action_spaces: List[ReactionActionSpaceA],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_indices = [shared_embeddings.molecule_to_idx[state.molecule] for state in states]
        embedding_indices = torch.tensor(embedding_indices).long().to(self.device)

        def _get_embeddings(all_state_embeddings: Tensor, all_action_embeddings: Tensor):
            state_embeddings = torch.index_select(
                all_state_embeddings,
                index=embedding_indices,
                dim=0,
            )
            state_embeddings = torch.repeat_interleave(
                state_embeddings, self.num_a_actions, dim=0
            )  # (n_states * num_reactions, hidden_dim)
            action_embeddings = torch.repeat_interleave(all_action_embeddings, len(states), dim=0)
            return torch.cat(
                [state_embeddings, action_embeddings], dim=-1
            )  # (n_states * num_reactions, 2 * hidden_dim)

        target_embeddings = _get_embeddings(
            shared_embeddings.all_target_embeddings, self.target_action_embedding_a.get_embeddings()
        )
        predictor_embeddings = _get_embeddings(
            shared_embeddings.all_predictor_embeddings,
            self.predictor_action_embedding_a.get_embeddings(),
        )
        target_embeddings = self.target_mlp_a(
            target_embeddings
        )  # (n_states * num_reactions, hidden_dim)
        predictor_embeddings = self.predictor_mlp_a(
            predictor_embeddings
        )  # (n_states * num_reactions, hidden_dim)

        logits = torch.norm(predictor_embeddings - target_embeddings.detach(), dim=-1, p=2)
        logits = logits.view(len(states), self.num_a_actions)  # (n_states, num_reactions)

        mask = torch.tensor(
            [action_space.possible_actions_mask for action_space in action_spaces]
        ).to(self.device)
        logits = torch.masked_fill(logits, ~mask, float("-inf"))
        return logits

    def _forward_b(
        self,
        states: List[ReactionStateB],
        action_spaces: List[ReactionActionSpaceB],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_indices = [
            shared_embeddings.molecule_reaction_to_idx[(state.molecule, state.reaction)]
            for state in states
        ]
        embedding_indices = torch.tensor(embedding_indices).long().to(self.device)
        actions_indices = [
            [action.idx for action in action_space.possible_actions]
            for action_space in action_spaces
        ]
        action_space_sizes = (
            torch.tensor([len(indices) for indices in actions_indices]).long().to(self.device)
        )
        actions_indices_flat = [idx for indices in actions_indices for idx in indices]
        action_indices_flat = torch.tensor(actions_indices_flat).long().to(self.device)

        def _get_embeddings(all_state_embeddings: Tensor, all_action_embeddings: Tensor):
            state_embeddings = torch.index_select(
                all_state_embeddings, index=embedding_indices, dim=0
            )  # (n_states, hidden_dim)
            action_embeddings = torch.index_select(
                all_action_embeddings, index=action_indices_flat, dim=0
            )  # (total_num_actions, hidden_dim)
            state_embeddings = torch.repeat_interleave(
                state_embeddings, action_space_sizes, dim=0
            )  # (total_num_actions, hidden_dim)
            return torch.cat(
                [state_embeddings, action_embeddings], dim=-1
            )  # (total_num_actions, 2 * hidden_dim)

        target_embeddings = _get_embeddings(
            shared_embeddings.all_target_embeddings, self.target_action_embedding_b.get_embeddings()
        )
        predictor_embeddings = _get_embeddings(
            shared_embeddings.all_predictor_embeddings,
            self.predictor_action_embedding_b.get_embeddings(),
        )
        target_embeddings = self.target_mlp_b(target_embeddings)  # (total_num_actions, hidden_dim)
        predictor_embeddings = self.predictor_mlp_b(
            predictor_embeddings
        )  # (total_num_actions, hidden_dim)

        logits = torch.norm(
            predictor_embeddings - target_embeddings.detach(), dim=-1, p=2
        )  # (total_num_actions)

        logits, mask = to_dense_embeddings(
            logits, action_space_sizes, fill_value=0
        )  # (batch_size, max_num_actions)

        logits = torch.masked_fill(logits, ~mask, float("-inf"))
        return logits

    def _forward_c(
        self,
        states: List[ReactionStateC],
        action_spaces: List[ReactionActionSpaceC],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_indices_list = []
        for action_space in action_spaces:
            embedding_indices = [
                shared_embeddings.molecule_to_idx[action.output_molecule]
                for action in action_space.possible_actions
            ]
            embedding_indices_list.append(embedding_indices)
        embedding_indices_flat = [idx for indices in embedding_indices_list for idx in indices]
        embedding_indices = torch.tensor(embedding_indices_flat).long().to(self.device)

        target_embeddings = torch.index_select(
            shared_embeddings.all_target_embeddings, index=embedding_indices, dim=0
        )
        target_embeddings = self.target_mlp_c(target_embeddings)

        predictor_embeddings = torch.index_select(
            shared_embeddings.all_predictor_embeddings, index=embedding_indices, dim=0
        )
        predictor_embeddings = self.predictor_mlp_c(predictor_embeddings)

        logits = torch.norm(
            predictor_embeddings - target_embeddings.detach(), dim=-1, p=2
        )  # (total_num_actions)
        logits, _ = to_dense_embeddings(
            logits, [len(indices) for indices in embedding_indices_list], fill_value=float("-inf")
        )
        return logits

    def _forward_early_terminate(
        self,
        states: List[ReactionState],
        action_spaces: List[ReactionActionSpaceEarlyTerminate],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        return torch.zeros((len(states), 1), device=self.device, dtype=torch.float32)

    def get_shared_embeddings(
        self, states: List[ReactionState], action_spaces: List[ReactionActionSpace]
    ) -> SharedEmbeddings:
        all_molecules = set()
        all_molecules_reactions = set()
        for state, action_space in zip(states, action_spaces):
            if isinstance(action_space, ReactionActionSpace0):
                all_molecules.add(None)
            elif isinstance(action_space, ReactionActionSpaceA):
                all_molecules.add(state.molecule)
            elif isinstance(action_space, ReactionActionSpaceB):
                all_molecules_reactions.add((state.molecule, state.reaction))
            elif isinstance(action_space, ReactionActionSpaceC):
                for action in action_space.possible_actions:
                    all_molecules.add(action.output_molecule)

        molecule_to_idx = {molecule: idx for idx, molecule in enumerate(all_molecules)}
        molecule_and_reaction_to_idx = {
            molecule_reaction: idx + len(molecule_to_idx)
            for idx, molecule_reaction in enumerate(all_molecules_reactions)
        }

        molecule_graphs = [
            mol2graph(mol.rdkit_mol if mol else None) for mol in molecule_to_idx.keys()
        ]
        reaction_cond = [one_hot(0, len(self.reactions))] * len(molecule_to_idx)

        molecule_and_reaction_graphs = [
            mol2graph(mol.rdkit_mol) for mol, _ in molecule_and_reaction_to_idx.keys()
        ]
        molecule_and_reaction_cond = [
            one_hot(r.idx, len(self.reactions)) for _, r in molecule_and_reaction_to_idx.keys()
        ]

        graphs = molecule_graphs + molecule_and_reaction_graphs
        conds = reaction_cond + molecule_and_reaction_cond

        graph_batch = mols2batch(graphs).to(self.device)
        cond_batch = torch.tensor(conds).float().to(self.device)

        target_embeddings = self.target_gnn(graph_batch, cond_batch).detach()
        predictor_embeddings = self.predictor_gnn(graph_batch, cond_batch)
        return SharedEmbeddings(
            molecule_to_idx=molecule_to_idx,
            molecule_reaction_to_idx=molecule_and_reaction_to_idx,
            all_target_embeddings=target_embeddings,
            all_predictor_embeddings=predictor_embeddings,
        )

    def _sample_actions_from_log_probs(
        self, log_probs: TensorType[float], action_spaces: List[TIndexedActionSpace]
    ) -> List[TAction]:
        """
        A helper function to sample actions from the log probabilities.

        Args:
            log_probs: log probabilities of the shape (N, max_num_actions)
            action_spaces: the list of action spaces of the length N.

        Returns:
            the list of sampled actions.
        """
        log_probs = torch.log_softmax(log_probs * self.temperature, dim=-1)
        return super()._sample_actions_from_log_probs(log_probs, action_spaces)

    def compute_states_log_flow(self, states: List[ReactionState]) -> TensorType[float]:
        raise NotImplementedError()

    def clear_action_embedding_cache(self) -> None:
        self.predictor_action_embedding_a.clear_cache()
        self.predictor_action_embedding_b.clear_cache()

    def clear_sampling_cache(self) -> None:
        pass

    def update_using_trajectories(
        self, trajectories: Trajectories[TState, TActionSpace, TAction], update_idx: int
    ) -> Dict[str, float]:
        if update_idx == self.last_update_idx:
            return {}
        self.last_update_idx = update_idx

        self.optimizer.zero_grad()
        states = trajectories.get_non_last_states_flat()
        action_spaces = trajectories.get_forward_action_spaces_flat()
        actions = trajectories.get_actions_flat()
        loss = self.compute_state_action_novelty(states, action_spaces, actions).mean()
        loss.backward()
        self.optimizer.step()
        return {"novelty_policy_loss": loss.item()}

    def compute_action_log_probs(
        self, states: List[TState], action_spaces: List[TIndexedActionSpace], actions: List[TAction]
    ) -> TensorType[float]:
        raise NotImplementedError()

    def compute_state_action_novelty(
        self, states: List[TState], action_spaces: List[TIndexedActionSpace], actions: List[TAction]
    ) -> TensorType[float]:
        # This part can be optimized: we don't need to compute novelty for the entire action spaces, but only
        # for the actions that were actually taken.
        return super().compute_action_log_probs(states, action_spaces, actions)

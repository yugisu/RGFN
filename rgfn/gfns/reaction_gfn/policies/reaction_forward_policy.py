from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type

import gin
import torch
from rdkit import Chem
from torch import nn
from torch.distributions import Categorical
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
from rgfn.gfns.reaction_gfn.policies.action_embeddings import FragmentEmbeddingBase
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

    all_embeddings: TensorType[float]


@gin.configurable()
class ReactionForwardPolicy(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, SharedEmbeddings],
    nn.Module,
):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        action_embedding_fn: FragmentEmbeddingBase,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 5,
        linear_output: bool = False,
        random_c: bool = False,
    ):
        super().__init__()
        self.reactions = data_factory.get_reactions()
        self.reaction_to_idx = {reaction: idx for idx, reaction in enumerate(self.reactions)}
        self.fragments = data_factory.get_fragments()
        self.random_c = random_c
        self.gnn = GraphTransformer(
            x_dim=71,
            e_dim=4,
            g_dim=len(self.reactions),
            num_layers=num_layers,
            num_heads=num_heads,
            num_emb=hidden_dim,
        )

        # State A -> Action A
        self.mlp_a = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(self.reactions) + 1),
            )
            if not linear_output
            else nn.Linear(hidden_dim, len(self.reactions) + 1)
        )

        # State B -> Action B
        self.mlp_b = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if not linear_output
            else nn.Linear(hidden_dim, hidden_dim)
        )
        self.b_action_embedding_fn = action_embedding_fn
        self.b_action_embedding_cache = None

        # State C -> Action C
        self.mlp_c = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            if not linear_output
            else nn.Linear(hidden_dim, 1)
        )

        self._action_space_type_to_forward_fn = {
            ReactionActionSpace0: self._forward_0,
            ReactionActionSpaceA: self._forward_a,
            ReactionActionSpaceB: self._forward_b,
            ReactionActionSpaceC: self._forward_c,
            ReactionActionSpaceEarlyTerminate: self._forward_early_terminate,
            ReactionActionSpace0Invalid: self._forward_early_terminate,
        }

        self._device = "cpu"

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
        self.b_action_embedding_fn.set_device(device)
        self._device = device

    def _forward_0(
        self,
        states: List[ReactionState0],
        action_spaces: List[ReactionActionSpace0],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_idx = shared_embeddings.molecule_to_idx[None]
        embedding = torch.index_select(
            shared_embeddings.all_embeddings,
            index=torch.tensor([embedding_idx]).long().to(self.device),
            dim=0,
        )
        embeddings = self.mlp_b(embedding)  # (1, hidden_dim)
        all_action_embeddings = self._get_b_action_embedding()  # (num_fragments, hidden_dim)
        logits = embeddings @ all_action_embeddings.T  # (1, num_fragments)
        log_prob = torch.log_softmax(logits, dim=1)
        x = log_prob.repeat(len(states), 1)
        return x

    def _get_b_action_embedding(self) -> TensorType[float]:
        if self.b_action_embedding_cache is None:
            self.b_action_embedding_cache = self.b_action_embedding_fn.get_embeddings()
        return self.b_action_embedding_cache

    def _forward_a(
        self,
        states: List[ReactionStateA],
        action_spaces: List[ReactionActionSpaceA],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        embedding_indices = [shared_embeddings.molecule_to_idx[state.molecule] for state in states]
        embedding_indices = torch.tensor(embedding_indices).long().to(self.device)
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=embedding_indices, dim=0
        )
        logits = self.mlp_a(embeddings)
        mask = torch.tensor(
            [action_space.possible_actions_mask for action_space in action_spaces]
        ).to(self.device)
        logits = torch.masked_fill(logits, ~mask, float("-inf"))
        return torch.log_softmax(logits, dim=1)

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
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=embedding_indices, dim=0
        )

        actions_indices = [
            [action.idx for action in action_space.possible_actions]
            for action_space in action_spaces
        ]
        actions_indices_flat = [idx for indices in actions_indices for idx in indices]
        action_indices_flat = torch.tensor(actions_indices_flat).long().to(self.device)
        all_action_embeddings = self._get_b_action_embedding()  # (num_fragments, hidden_dim)
        action_embeddings = torch.index_select(
            all_action_embeddings, index=action_indices_flat, dim=0
        )
        actions_embeddings, mask = to_dense_embeddings(
            action_embeddings, [len(indices) for indices in actions_indices], fill_value=0
        )  # (batch_size, max_num_actions, hidden_dim)

        embeddings = self.mlp_b(embeddings)  # (batch_size, hidden_dim)
        logits = torch.matmul(actions_embeddings, embeddings.unsqueeze(2)).squeeze(
            2
        )  # (batch_size, max_num_actions)
        logits = torch.masked_fill(logits, ~mask, float("-inf"))
        return torch.log_softmax(logits, dim=1)

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
        embeddings = torch.index_select(
            shared_embeddings.all_embeddings, index=embedding_indices, dim=0
        )
        logits = self.mlp_c(embeddings).squeeze(-1)
        if self.random_c:
            logits = torch.zeros_like(logits.detach())
        logits, _ = to_dense_embeddings(
            logits, [len(indices) for indices in embedding_indices_list], fill_value=float("-inf")
        )
        log_prob = torch.log_softmax(logits, dim=1)
        return log_prob

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

        embeddings = self.gnn(graph_batch, cond_batch)
        return SharedEmbeddings(
            molecule_to_idx=molecule_to_idx,
            molecule_reaction_to_idx=molecule_and_reaction_to_idx,
            all_embeddings=embeddings,
        )

    def compute_states_log_flow(self, states: List[ReactionState]) -> TensorType[float]:
        raise NotImplementedError()

    def clear_action_embedding_cache(self) -> None:
        self.b_action_embedding_cache = None

    def clear_sampling_cache(self) -> None:
        pass

    def update_using_trajectories(self, trajectories: Trajectories[TState, TActionSpace, TAction]):
        pass

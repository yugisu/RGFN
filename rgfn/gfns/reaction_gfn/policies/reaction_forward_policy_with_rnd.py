from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Type

import gin
import torch
from torch import nn
from torch.nn import Parameter
from torchtyping import TensorType

from rgfn.api.env_base import TAction, TActionSpace, TState
from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
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
from rgfn.shared.policies.few_phase_policy import FewPhasePolicyBase, TSharedEmbeddings
from rgfn.shared.policies.uniform_policy import TIndexedActionSpace

from .reaction_forward_policy import ReactionForwardPolicy
from .reaction_forward_policy import SharedEmbeddings as ForwardSharedEmbeddings
from .rnd_novelty_forward_policy import RNDNoveltyForwardPolicy
from .rnd_novelty_forward_policy import SharedEmbeddings as RNDSharedEmbeddings


@dataclass(frozen=True)
class SharedEmbeddings:
    forward_shared_embeddings: ForwardSharedEmbeddings
    rnd_shared_embeddings: RNDSharedEmbeddings


@gin.configurable()
class ReactionForwardPolicyWithRND(
    FewPhasePolicyBase[ReactionState, ReactionActionSpace, ReactionAction, SharedEmbeddings],
    nn.Module,
):
    def __init__(
        self,
        reaction_forward_policy: ReactionForwardPolicy,
        rnd_novelty_forward_policy: RNDNoveltyForwardPolicy,
    ):
        super().__init__()
        self.reaction_forward_policy = reaction_forward_policy
        self.rnd_novelty_forward_policy = rnd_novelty_forward_policy

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
        self.reaction_forward_policy.set_device(device)
        self.rnd_novelty_forward_policy.set_device(device)
        self._device = device

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.reaction_forward_policy.parameters()

    def _forward_0(
        self,
        states: List[ReactionState0],
        action_spaces: List[ReactionActionSpace0],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        logits_1 = self.reaction_forward_policy._forward_0(
            states, action_spaces, shared_embeddings.forward_shared_embeddings
        )
        logits_2 = self.rnd_novelty_forward_policy._forward_0(
            states, action_spaces, shared_embeddings.rnd_shared_embeddings
        )
        return logits_1 + logits_2.detach()

    def _forward_a(
        self,
        states: List[ReactionStateA],
        action_spaces: List[ReactionActionSpaceA],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        logits_1 = self.reaction_forward_policy._forward_a(
            states, action_spaces, shared_embeddings.forward_shared_embeddings
        )
        logits_2 = self.rnd_novelty_forward_policy._forward_a(
            states, action_spaces, shared_embeddings.rnd_shared_embeddings
        )
        return logits_1 + logits_2.detach()

    def _forward_b(
        self,
        states: List[ReactionStateB],
        action_spaces: List[ReactionActionSpaceB],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        logits_1 = self.reaction_forward_policy._forward_b(
            states, action_spaces, shared_embeddings.forward_shared_embeddings
        )
        logits_2 = self.rnd_novelty_forward_policy._forward_b(
            states, action_spaces, shared_embeddings.rnd_shared_embeddings
        )
        return logits_1 + logits_2.detach()

    def _forward_c(
        self,
        states: List[ReactionStateC],
        action_spaces: List[ReactionActionSpaceC],
        shared_embeddings: SharedEmbeddings,
    ) -> TensorType[float]:
        logits_1 = self.reaction_forward_policy._forward_c(
            states, action_spaces, shared_embeddings.forward_shared_embeddings
        )
        logits_2 = self.rnd_novelty_forward_policy._forward_c(
            states, action_spaces, shared_embeddings.rnd_shared_embeddings
        )
        return logits_1 + logits_2.detach()

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
        forward_shared_embeddings = self.reaction_forward_policy.get_shared_embeddings(
            states, action_spaces
        )
        rnd_shared_embeddings = self.rnd_novelty_forward_policy.get_shared_embeddings(
            states, action_spaces
        )
        return SharedEmbeddings(
            forward_shared_embeddings=forward_shared_embeddings,
            rnd_shared_embeddings=rnd_shared_embeddings,
        )

    def compute_states_log_flow(self, states: List[ReactionState]) -> TensorType[float]:
        raise NotImplementedError()

    def clear_action_embedding_cache(self) -> None:
        self.reaction_forward_policy.clear_action_embedding_cache()
        self.rnd_novelty_forward_policy.clear_action_embedding_cache()

    def clear_sampling_cache(self) -> None:
        self.reaction_forward_policy.clear_sampling_cache()
        self.rnd_novelty_forward_policy.clear_sampling_cache()

    def update_using_trajectories(
        self, trajectories: Trajectories[TState, TActionSpace, TAction], update_idx: int
    ) -> Dict[str, float]:
        output_1 = self.reaction_forward_policy.update_using_trajectories(trajectories, update_idx)
        output_2 = self.rnd_novelty_forward_policy.update_using_trajectories(
            trajectories, update_idx
        )
        return output_1 | output_2

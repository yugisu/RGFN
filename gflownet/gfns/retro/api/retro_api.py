from dataclasses import dataclass, field
from typing import Any, FrozenSet, List, Tuple

from gflownet.api.env_base import TAction
from gflownet.common.policies.uniform_policy import IndexedActionSpaceBase

from .data_structures import (
    Bag,
    MappingTuple,
    Molecule,
    Pattern,
    ReactantPattern,
    SortedList,
)


@dataclass(frozen=True)
class FirstPhaseRetroState:
    product: Molecule


@dataclass(frozen=True)
class FirstPhaseRetroAction:
    subgraph_idx: Tuple[int, ...]
    product_pattern: Pattern

    def __post_init__(self):
        if self.product_pattern.symmetric():
            if self.subgraph_idx[0] > self.subgraph_idx[-1]:
                subgraph_idx = tuple(reversed(self.subgraph_idx))
                object.__setattr__(self, "subgraph_idx", subgraph_idx)


@dataclass(frozen=True)
class FirstPhaseRetroActionSpace(IndexedActionSpaceBase[FirstPhaseRetroAction]):
    possible_actions: FrozenSet[FirstPhaseRetroAction]

    def num_unique_subgraphs(self) -> int:
        return len({tuple(sorted(action.subgraph_idx)) for action in self.possible_actions})

    def get_action_at_idx(self, idx: int) -> FirstPhaseRetroAction:
        return list(self.possible_actions)[idx]

    def get_idx_of_action(self, action: FirstPhaseRetroAction) -> int:
        return list(self.possible_actions).index(action)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))


@dataclass(frozen=True)
class SecondPhaseRetroState:
    product: Molecule
    subgraph_idx: Tuple[int, ...]
    product_pattern: Pattern
    reactant_patterns: Bag[ReactantPattern]


@dataclass(frozen=True)
class SecondPhaseRetroAction:
    reactant_pattern_idx: int


@dataclass(frozen=True)
class SecondPhaseRetroActionSpace(IndexedActionSpaceBase[SecondPhaseRetroAction]):
    actions_mask: List[bool]

    def get_action_at_idx(self, idx: int) -> SecondPhaseRetroAction:
        return SecondPhaseRetroAction(idx)

    def get_idx_of_action(self, action: SecondPhaseRetroAction) -> int:
        return action.reactant_pattern_idx

    def get_possible_actions_indices(self) -> List[int]:
        return [idx for idx, mask in enumerate(self.actions_mask) if mask]


@dataclass(frozen=True)
class ThirdPhaseRetroState:
    product: Molecule
    subgraph_idx: Tuple[int, ...]
    product_pattern: Pattern
    reactant_patterns: SortedList[ReactantPattern]
    atom_mapping: FrozenSet[MappingTuple]


@dataclass(frozen=True)
class ThirdPhaseRetroAction:
    mapping: MappingTuple


@dataclass(frozen=True)
class ThirdPhaseRetroActionSpace(IndexedActionSpaceBase[ThirdPhaseRetroAction]):
    possible_actions: FrozenSet[MappingTuple]

    def get_action_at_idx(self, idx: int) -> ThirdPhaseRetroAction:
        return ThirdPhaseRetroAction(list(self.possible_actions)[idx])

    def get_idx_of_action(self, action: ThirdPhaseRetroAction) -> int:
        return list(self.possible_actions).index(action.mapping)

    def get_possible_actions_indices(self) -> List[int]:
        return list(range(len(self.possible_actions)))


@dataclass(frozen=True)
class EarlyTerminateRetroAction:
    pass


@dataclass(frozen=True)
class EarlyTerminateRetroActionSpace(IndexedActionSpaceBase[EarlyTerminateRetroAction]):
    def get_action_at_idx(self, idx: int) -> EarlyTerminateRetroAction:
        return EarlyTerminateRetroAction()

    def get_idx_of_action(self, action: EarlyTerminateRetroAction) -> int:
        return 0

    def get_possible_actions_indices(self) -> List[int]:
        return [0]


@dataclass(frozen=True)
class EarlyTerminalRetroState:
    previous_state: Any


@dataclass(frozen=True)
class TerminalRetroState:
    product: Molecule = field(hash=True, compare=True)
    reactants: Bag[Molecule] = field(hash=True, compare=True)
    subgraph_idx: Tuple[int, ...] = field(hash=False, compare=False)
    product_pattern: Pattern = field(hash=False, compare=False)
    reactant_patterns: SortedList[ReactantPattern] = field(hash=False, compare=False)
    atom_mapping: FrozenSet[MappingTuple] = field(hash=False, compare=False)
    template: str = field(hash=False, compare=False)
    valid: bool = field(hash=False, compare=False)

    def __repr__(self):
        return (
            f"product={self.product.smiles}\n"
            f"reactants={[r.smiles for r in self.reactants]}\n"
            f"template={self.template}\n"
        )


RetroState = (
    FirstPhaseRetroState
    | SecondPhaseRetroState
    | ThirdPhaseRetroState
    | EarlyTerminalRetroState
    | TerminalRetroState
)

RetroAction = (
    FirstPhaseRetroAction
    | SecondPhaseRetroAction
    | ThirdPhaseRetroAction
    | EarlyTerminateRetroAction
)

RetroActionSpace = (
    FirstPhaseRetroActionSpace
    | SecondPhaseRetroActionSpace
    | ThirdPhaseRetroActionSpace
    | EarlyTerminateRetroActionSpace
)

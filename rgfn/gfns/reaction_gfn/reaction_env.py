from typing import Any, Dict, List, Tuple

import gin
from rdkit import RDLogger
from rdkit.Chem import Mol

from rgfn.api.env_base import EnvBase
from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.data_structures import Pattern
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    Molecule,
    ReactionAction,
    ReactionAction0,
    ReactionAction0Invalid,
    ReactionActionA,
    ReactionActionB,
    ReactionActionC,
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
    ReactionState0Invalid,
    ReactionStateA,
    ReactionStateB,
    ReactionStateC,
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory

RDLogger.DisableLog("rdApp.*")


@gin.configurable()
class ReactionEnv(EnvBase[ReactionState, ReactionActionSpace, ReactionAction]):
    def reversed(self) -> "EnvBase[ReactionState, ReactionActionSpace, ReactionAction]":
        env = super().reversed()
        if self.share_cache_with_reversed:
            env._cache = self._cache
            env._additional_backward_cache = self._additional_backward_cache
        return env

    def __init__(
        self,
        data_factory: ReactionDataFactory,
        max_num_reactions: int,
        share_cache_with_reversed: bool = True,
    ):
        super().__init__()
        self.reactions = data_factory.get_reactions()
        self.disconnections = data_factory.get_disconnections()
        self.anchored_reactions = data_factory.get_anchored_reactions()
        self.reaction_anchor_map = data_factory.get_reaction_anchor_map()
        self.anchored_disconnections = data_factory.get_anchored_disconnections()
        self.fragments = data_factory.get_fragments()
        self.max_num_reaction = max_num_reactions
        self._cache: Dict[Tuple[str, int], bool] = {}
        self._cache_size = 100000000
        self._additional_backward_cache: Dict[Any, Any] = {}
        self._additional_backward_cache_size = 10000000
        self.share_cache_with_reversed = share_cache_with_reversed
        self.all_actions_0 = tuple(
            ReactionAction0(fragment=f, idx=i) for i, f in enumerate(self.fragments)
        )

        self.all_actions_a = tuple(
            ReactionActionA(anchored_reaction=r, idx=i)
            for i, r in enumerate(self.anchored_reactions)
        ) + (
            ReactionActionA(anchored_reaction=None, idx=len(self.anchored_reactions)),
        )

        self.all_actions_b = tuple(
            ReactionActionB(fragment=f, idx=i) for i, f in enumerate(self.fragments)
        )

        self.pattern_to_compatible_fragments: Dict[Pattern, List[Molecule]] = {}

        for reaction in self.reactions:
            for pattern in reaction.left_side_patterns:
                fragments = []
                for i, fragment in enumerate(self.fragments):
                    if fragment.rdkit_mol.HasSubstructMatch(pattern.rdkit_pattern):
                        fragments.append(fragment)
                self.pattern_to_compatible_fragments[pattern] = fragments

        self.smiles_to_fragment_idx = {fragment.smiles: fragment.idx for fragment in self.fragments}
        self.fragment_smiles_set = set(self.smiles_to_fragment_idx.keys())

        self.forward_action_space_dict = {
            ReactionState0: self._get_forward_action_spaces_0,
            ReactionState0Invalid: self._get_forward_action_spaces_0_invalid,
            ReactionStateA: self._get_forward_action_spaces_a,
            ReactionStateB: self._get_forward_action_spaces_b,
            ReactionStateC: self._get_forward_action_spaces_c,
        }
        self.backward_action_space_dict = {
            ReactionStateA: self._get_backward_action_spaces_a,
            ReactionStateB: self._get_backward_action_spaces_b,
            ReactionStateC: self._get_backward_action_spaces_c,
            ReactionStateTerminal: self._get_backward_action_spaces_terminal,
            ReactionStateEarlyTerminal: self._get_backward_action_spaces_early_terminal,
        }
        self.forward_action_dict = {
            ReactionAction0: self._apply_forward_actions_0,
            ReactionAction0Invalid: self._apply_forward_actions_0_invalid,
            ReactionActionA: self._apply_forward_actions_a,
            ReactionActionB: self._apply_forward_actions_b,
            ReactionActionC: self._apply_forward_actions_c,
            ReactionActionEarlyTerminate: self._apply_forward_actions_early_terminate,
        }
        self.backward_action_dict = {
            ReactionAction0: self._apply_backward_actions_0,
            ReactionAction0Invalid: self._apply_backward_actions_0_invalid,
            ReactionActionA: self._apply_backward_actions_a,
            ReactionActionB: self._apply_backward_actions_b,
            ReactionActionC: self._apply_backward_actions_c,
            ReactionActionEarlyTerminate: self._apply_backward_actions_early_terminate,
        }

    def get_forward_action_spaces(self, states: List[ReactionState]) -> List[ReactionActionSpace]:
        action_spaces = []
        for state in states:
            action_space = self.forward_action_space_dict[type(state)](state)
            action_spaces.append(action_space)
        return action_spaces

    def _get_forward_action_spaces_0(self, state: ReactionState0) -> ReactionActionSpace0:
        return ReactionActionSpace0(
            all_actions=self.all_actions_0, possible_actions_mask=[True] * len(self.all_actions_0)
        )

    def _get_forward_action_spaces_0_invalid(
        self, state: ReactionState0Invalid
    ) -> ReactionActionSpace0Invalid:
        return ReactionActionSpace0Invalid()

    def _get_forward_action_spaces_a(
        self, state: ReactionStateA
    ) -> ReactionActionSpaceA | ReactionActionSpaceEarlyTerminate:
        mask = [False] * len(self.all_actions_a)
        if state.num_reactions > 0:
            mask[-1] = True

        if state.num_reactions < self.max_num_reaction:
            mol = state.molecule.rdkit_mol
            for i, action in enumerate(self.all_actions_a[:-1]):
                action: ReactionActionA
                anchored_reaction = action.anchored_reaction
                anchored_pattern = anchored_reaction.anchored_pattern
                if not mol.HasSubstructMatch(anchored_pattern.rdkit_pattern):
                    continue
                if all(
                    len(self.pattern_to_compatible_fragments[pattern]) > 0
                    for pattern in anchored_reaction.fragment_patterns
                ):
                    mask[i] = True

        if not any(mask):
            return ReactionActionSpaceEarlyTerminate()
        return ReactionActionSpaceA(all_actions=self.all_actions_a, possible_actions_mask=mask)

    def _get_forward_action_spaces_b(
        self, state: ReactionStateB
    ) -> ReactionActionSpaceB | ReactionActionSpaceEarlyTerminate:
        next_pattern = state.anchored_reaction.fragment_patterns[len(state.fragments)]
        possible_actions = [
            self.all_actions_b[fragment.idx]
            for fragment in self.pattern_to_compatible_fragments[next_pattern]
        ]

        if len(possible_actions) == 0:
            return ReactionActionSpaceEarlyTerminate()

        return ReactionActionSpaceB(possible_actions=tuple(possible_actions))

    def _get_forward_action_spaces_c(
        self, state: ReactionStateC
    ) -> ReactionActionSpaceC | ReactionActionSpaceEarlyTerminate:
        anchored_reaction = state.anchored_reaction
        reactants = [state.molecule] + list(state.fragments)
        products = anchored_reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])

        products_list = [Molecule(mol[0]) for mol in products]
        products_list = [mol for mol in products_list if mol.valid]
        products_set = set(products_list)

        anchored_disconnection = self.anchored_disconnections[anchored_reaction.idx]
        expected_reactants = set(reactants)
        possible_actions = []
        for product in products_set:
            prev_reactants_list = anchored_disconnection.rdkit_rxn.RunReactants([product.rdkit_mol])
            for prev_reactants in prev_reactants_list:
                prev_reactants = set(Molecule(prev) for prev in prev_reactants)
                if prev_reactants == expected_reactants:
                    action = ReactionActionC(
                        input_molecule=state.molecule,
                        input_reaction=state.anchored_reaction,
                        input_fragments=state.fragments,
                        output_molecule=product,
                    )
                    if not self.is_reversed:
                        self._cache[(product.smiles, state.num_reactions + 1)] = True
                    possible_actions.append(action)
                    break

        if len(possible_actions) == 0:
            return ReactionActionSpaceEarlyTerminate()
        return ReactionActionSpaceC(possible_actions=tuple(possible_actions))

    def get_backward_action_spaces(self, states: List[ReactionState]) -> List[ReactionActionSpace]:
        action_spaces = []
        for state in states:
            action_space = self.backward_action_space_dict[type(state)](state)
            action_spaces.append(action_space)
        return action_spaces

    def _get_backward_action_spaces_a(
        self, state: ReactionStateA
    ) -> ReactionActionSpace0 | ReactionActionSpace0Invalid | ReactionActionSpaceC:
        if state.num_reactions == 0:
            mask = [False] * len(self.all_actions_0)
            mask[self.smiles_to_fragment_idx[state.molecule.smiles]] = True
            return ReactionActionSpace0(all_actions=self.all_actions_0, possible_actions_mask=mask)

        if state in self._additional_backward_cache:
            return self._additional_backward_cache[state]

        possible_actions = []
        for anchored_reaction, anchored_disconnection in zip(
            self.anchored_reactions, self.anchored_disconnections
        ):
            reactants_list = anchored_disconnection.rdkit_rxn.RunReactants(
                [state.molecule.rdkit_mol]
            )
            reactants_tuple_set = set(
                tuple(Molecule(mol) for mol in reactants) for reactants in reactants_list
            )
            for reactants in reactants_tuple_set:
                previous_molecule = reactants[0]
                previous_fragments = reactants[1:]
                if all(
                    self._is_fragment(frag.smiles) for frag in previous_fragments
                ) and self._is_decomposable(previous_molecule, state.num_reactions - 1):
                    reactants_rdkit = [r.rdkit_mol for r in reactants]
                    new_products = anchored_reaction.rdkit_rxn.RunReactants(reactants_rdkit)
                    new_products_mols = [Molecule(mol[0]) for mol in new_products]
                    new_products_mols = [mol for mol in new_products_mols if mol.valid]
                    new_products_mols = set(new_products_mols)

                    if state.molecule in new_products_mols:
                        previous_fragments = tuple(
                            self.fragments[self.smiles_to_fragment_idx[fragment.smiles]]
                            for fragment in previous_fragments
                        )
                        action = ReactionActionC(
                            input_molecule=previous_molecule,
                            input_reaction=anchored_reaction,
                            input_fragments=previous_fragments,
                            output_molecule=state.molecule,
                        )
                        possible_actions.append(action)

        if len(possible_actions) == 0:
            print("No backward actions found for state: {}".format(state))
            return ReactionActionSpace0Invalid()

        action_space = ReactionActionSpaceC(possible_actions=tuple(possible_actions))
        self._additional_backward_cache[state] = action_space
        if len(self._additional_backward_cache) > self._additional_backward_cache_size:
            self._additional_backward_cache.popitem()
        while len(self._cache) > self._cache_size:
            self._cache.popitem()
        return action_space

    def _is_fragment(self, smiles: str) -> bool:
        return smiles in self.fragment_smiles_set

    def _lazy_is_fragment_check(self, molecules: List[Mol]) -> Tuple[List[Molecule], List[int]]:
        """
        Checks which of the given molecules are fragments and which need to be further decomposed.
        We don't need to check all fragments in the molecule. We can stop checking
        as soon as we more than one a non-fragment molecule, because we can only decompose one molecule.
        This way we can save few `Molecule()` calls.

        Returns:
            fragments: List of Molecule objects
            non_fragments_indices: List of indices of non-fragment molecules
        """
        fragments = []
        non_fragments_indices = []
        for idx, mol in enumerate(molecules):
            molecule = Molecule(mol)
            fragments.append(molecule)
            if not self._is_fragment(molecule.smiles):
                non_fragments_indices.append(idx)
                if len(non_fragments_indices) > 1:
                    break
        return fragments, non_fragments_indices

    def _is_decomposable(self, molecule: Molecule, n_reactions: int) -> bool:
        """
        Recursive helper for the decompose function. Returns true if the given Molecule
        can be disconnected fully into fragments in our library.

        Base case: is the molecule a fragment in our library? If yes, we're done.

        Recursive case: For each disconnect reaction, can we use this reaction to
            disconnect our molecule? Iterate through all fragment pairs generated by
            each reaction. If any fragment pair can be decomposed, return true.

        """

        if (molecule.smiles, n_reactions) in self._cache:
            return self._cache[(molecule.smiles, n_reactions)]

        if n_reactions == 0:
            return self._is_fragment(molecule.smiles)

        if not molecule.valid:
            self._cache[(molecule.smiles, n_reactions)] = False
            return False

        # Decompose the molecule by all reverse reactions possible.
        # If ANY of them work then return True.
        for reaction, disconnection in zip(self.reactions, self.disconnections):
            reactants_list = disconnection.rdkit_rxn.RunReactants((molecule.rdkit_mol,))
            for reactants in reactants_list:
                reactants, non_fragment_indices = self._lazy_is_fragment_check(reactants)
                if len(reactants) == 0:
                    continue
                if len(non_fragment_indices) == 0:
                    for fragment in reactants:
                        # we want to ensure that there is at least one fragment (coming from the library) in `reactants`
                        # that can be further decomposed with n_reactions - 1 reactions.
                        if self._is_decomposable(fragment, n_reactions - 1):
                            self._cache[(molecule.smiles, n_reactions)] = True
                            return True
                    continue
                if len(non_fragment_indices) == 1 and self._is_decomposable(
                    reactants[non_fragment_indices[0]], n_reactions - 1
                ):
                    # We should check the compatibility of forward and backward reaction here
                    # However, it's computationally expensive and we make another hack to
                    # get rid of empty action space issue in the backward sampling process.

                    # output_products = reaction.rdkit_rxn.RunReactants([r.rdkit_mol for r in reactants])
                    # for mol in output_products:
                    #     if molecule == Molecule(mol[0]):
                    #         self._cache[(molecule.smiles, n_reactions)] = True
                    #         return True

                    self._cache[(molecule.smiles, n_reactions)] = True
                    return True

        # We were unable to find a fully decomposable fragment pair.
        self._cache[(molecule.smiles, n_reactions)] = False
        return False

    def _get_backward_action_spaces_b(
        self, state: ReactionStateB
    ) -> ReactionActionSpaceA | ReactionActionSpaceB:
        if len(state.fragments) == 0:
            mask = [False] * len(self.all_actions_a)
            mask[state.anchored_reaction.idx] = True
            return ReactionActionSpaceA(all_actions=self.all_actions_a, possible_actions_mask=mask)
        previous_fragment_smiles = state.fragments[-1].smiles
        previous_fragment = self.fragments[self.smiles_to_fragment_idx[previous_fragment_smiles]]
        return ReactionActionSpaceB(possible_actions=(self.all_actions_b[previous_fragment.idx],))

    def _get_backward_action_spaces_c(
        self, state: ReactionStateC
    ) -> ReactionActionSpaceA | ReactionActionSpaceB:
        return self._get_backward_action_spaces_b(state)

    def _get_backward_action_spaces_terminal(
        self, state: ReactionStateTerminal
    ) -> ReactionActionSpaceA:
        mask = [False] * len(self.all_actions_a)
        mask[-1] = True
        return ReactionActionSpaceA(all_actions=self.all_actions_a, possible_actions_mask=mask)

    def _get_backward_action_spaces_early_terminal(
        self, state: ReactionStateEarlyTerminal
    ) -> ReactionActionSpace:
        return ReactionActionSpaceEarlyTerminate()

    def apply_forward_actions(
        self, states: List[ReactionState], actions: List[ReactionAction]
    ) -> List[ReactionState]:
        new_states = []
        for state, action in zip(states, actions):
            action_type = type(action)
            new_state = self.forward_action_dict[action_type](state, action)
            new_states.append(new_state)
        return new_states

    def _apply_forward_actions_0(
        self, state: ReactionState0, action: ReactionAction0
    ) -> ReactionStateA:
        return ReactionStateA(molecule=action.fragment, num_reactions=0)

    def _apply_forward_actions_0_invalid(
        self, state: ReactionState0Invalid, action: ReactionAction0Invalid
    ) -> ReactionState0Invalid:
        return state.previous_state

    def _apply_forward_actions_a(
        self, state: ReactionStateA, action: ReactionActionA
    ) -> ReactionStateB | ReactionStateC | ReactionStateTerminal:
        if action.anchored_reaction is None:
            return ReactionStateTerminal(molecule=state.molecule, num_reactions=state.num_reactions)
        if len(action.anchored_reaction.fragment_patterns) == 0:
            return ReactionStateC(
                molecule=state.molecule,
                anchored_reaction=action.anchored_reaction,
                fragments=(),
                num_reactions=state.num_reactions,
            )
        return ReactionStateB(
            molecule=state.molecule,
            anchored_reaction=action.anchored_reaction,
            num_reactions=state.num_reactions,
            fragments=(),
        )

    def _apply_forward_actions_b(
        self, state: ReactionStateB, action: ReactionActionB
    ) -> ReactionStateB | ReactionStateC:
        new_fragments = state.fragments + (action.fragment,)
        if len(new_fragments) == len(state.anchored_reaction.fragment_patterns):
            return ReactionStateC(
                molecule=state.molecule,
                anchored_reaction=state.anchored_reaction,
                fragments=new_fragments,
                num_reactions=state.num_reactions,
            )
        return ReactionStateB(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            fragments=new_fragments,
            num_reactions=state.num_reactions,
        )

    def _apply_forward_actions_c(
        self, state: ReactionStateC, action: ReactionActionC
    ) -> ReactionStateA:
        return ReactionStateA(
            molecule=action.output_molecule, num_reactions=state.num_reactions + 1
        )

    def _apply_forward_actions_early_terminate(
        self, state: ReactionState, action: ReactionActionEarlyTerminate
    ) -> ReactionStateEarlyTerminal:
        return ReactionStateEarlyTerminal(previous_state=state)

    def apply_backward_actions(
        self, states: List[ReactionState], actions: List[ReactionAction]
    ) -> List[ReactionState]:
        new_states = []
        for state, action in zip(states, actions):
            action_type = type(action)
            new_state = self.backward_action_dict[action_type](state, action)
            new_states.append(new_state)
        return new_states

    def _apply_backward_actions_0(
        self, state: ReactionStateA, action: ReactionAction0
    ) -> ReactionState0:
        return ReactionState0()

    def _apply_backward_actions_0_invalid(
        self, state: ReactionState, action: ReactionAction0Invalid
    ) -> ReactionState0Invalid:
        return ReactionState0Invalid(previous_state=state)

    def _apply_backward_actions_a(
        self,
        state: ReactionStateB | ReactionStateC | ReactionStateTerminal,
        action: ReactionActionA,
    ) -> ReactionStateA:
        return ReactionStateA(molecule=state.molecule, num_reactions=state.num_reactions)

    def _apply_backward_actions_b(
        self, state: ReactionStateB | ReactionStateC, action: ReactionActionB
    ) -> ReactionStateB:
        return ReactionStateB(
            molecule=state.molecule,
            anchored_reaction=state.anchored_reaction,
            num_reactions=state.num_reactions,
            fragments=state.fragments[:-1],
        )

    def _apply_backward_actions_c(
        self, state: ReactionStateA, action: ReactionActionC
    ) -> ReactionStateC:
        return ReactionStateC(
            molecule=action.input_molecule,
            anchored_reaction=action.input_reaction,
            fragments=action.input_fragments,
            num_reactions=state.num_reactions - 1,
        )

    def _apply_backward_actions_early_terminate(
        self, state: ReactionStateEarlyTerminal, action: ReactionActionEarlyTerminate
    ) -> ReactionState:
        return state.previous_state

    def get_terminal_mask(self, states: List[ReactionState]) -> List[bool]:
        return [
            isinstance(state, (ReactionStateTerminal, ReactionStateEarlyTerminal))
            for state in states
        ]

    def get_source_mask(self, states: List[ReactionState]) -> List[bool]:
        return [isinstance(state, (ReactionState0, ReactionState0Invalid)) for state in states]

    def sample_source_states(self, n_states: int) -> List[ReactionState]:
        return [ReactionState0()] * n_states

    def sample_terminal_states(self, n_states: int) -> List[ReactionState]:
        raise NotImplementedError()

    def get_num_source_states(self) -> int:
        raise NotImplementedError()

    def get_source_states_at_index(self, index: List[int]) -> List[TState]:
        raise NotImplementedError()

    def get_num_terminal_states(self) -> int:
        raise NotImplementedError()

    def get_terminal_states_at_index(self, index: List[int]) -> List[TState]:
        raise NotImplementedError()

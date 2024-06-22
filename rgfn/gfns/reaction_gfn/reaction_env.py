from typing import Any, Dict, List, Tuple, cast

import gin
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol

from rgfn.api.env_base import EnvBase, TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    Molecule,
    Reaction,
    ReactionAction,
    ReactionAction0,
    ReactionActionA,
    ReactionActionB,
    ReactionActionC,
    ReactionActionEarlyTerminate,
    ReactionActionSpace,
    ReactionActionSpace0,
    ReactionActionSpaceA,
    ReactionActionSpaceB,
    ReactionActionSpaceC,
    ReactionActionSpaceEarlyTerminate,
    ReactionState,
    ReactionState0,
    ReactionStateA,
    ReactionStateB,
    ReactionStateC,
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
from rgfn.gfns.reaction_gfn.api.reaction_data_factory import (
    ReactionDataFactory,
    canonicalize,
)

RDLogger.DisableLog("rdApp.*")


@gin.configurable()
class ReactionEnv(EnvBase[ReactionState, ReactionActionSpace, ReactionAction]):
    def __init__(self, data_factory: ReactionDataFactory, max_num_reactions: int):
        super().__init__()
        self.reactions = data_factory.get_reactions()
        self.disconnections = data_factory.get_disconnections()
        self.fragments = data_factory.get_fragments()
        self.max_num_reaction = max_num_reactions
        self._cache: Dict[Tuple[str, int], bool] = {}
        self._additional_backward_cache: Dict[Any, Any] = {}
        self._additional_backward_cache_size = 100000

        self.all_actions_0 = tuple(
            ReactionAction0(fragment=Molecule(f, idx=i), idx=i)
            for i, f in enumerate(self.fragments)
        )
        self.all_actions_a = [
            ReactionActionA(reaction=Reaction(r, idx=i), idx=i)
            for i, r in enumerate(self.reactions)
        ]
        self.all_actions_a.append(ReactionActionA(reaction=None, idx=len(self.reactions)))
        cast(List[ReactionActionA], self.all_actions_a)
        self.all_actions_a = tuple(self.all_actions_a)
        self.all_actions_b = tuple(
            ReactionActionB(fragment=Molecule(f, idx=i), idx=i)
            for i, f in enumerate(self.fragments)
        )
        self.compatible_fragments: Dict[str, List[str]] = {}
        self.compatible_indices: Dict[str, List[int]] = {}
        self.fragments = [action.fragment.smiles for action in self.all_actions_0]
        for reaction in self.reactions:
            self.compatible_fragments[reaction] = []
            self.compatible_indices[reaction] = []
            pattern = Chem.MolFromSmarts(reaction.split(" >> ")[0].split(".")[1])
            for i, fragment in enumerate(self.fragments):
                if Chem.MolFromSmiles(fragment).HasSubstructMatch(pattern):
                    self.compatible_fragments[reaction].append(fragment)
                    self.compatible_indices[reaction].append(i)

        self.reaction2index = {r: i for i, r in enumerate(self.reactions)}
        self.fragment2index = {f: i for i, f in enumerate(self.fragments)}

        self.forward_action_space_dict = {
            ReactionState0: self._get_forward_action_spaces_0,
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
            ReactionActionA: self._apply_forward_actions_a,
            ReactionActionB: self._apply_forward_actions_b,
            ReactionActionC: self._apply_forward_actions_c,
            ReactionActionEarlyTerminate: self._apply_forward_actions_early_terminate,
        }
        self.backward_action_dict = {
            ReactionStateA: self._apply_backward_actions_a,
            ReactionStateB: self._apply_backward_actions_b,
            ReactionStateC: self._apply_backward_actions_c,
            ReactionStateTerminal: self._apply_backward_actions_terminal,
            ReactionStateEarlyTerminal: self._apply_backward_actions_early_terminal,
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

    def _get_forward_action_spaces_a(
        self, state: ReactionStateA
    ) -> ReactionActionSpaceA | ReactionActionSpaceEarlyTerminate:
        mask = [False] * len(self.all_actions_a)
        if state.num_reactions > 0:
            mask[-1] = True

        if state.num_reactions < self.max_num_reaction:
            mol = state.molecule.rdkit_mol
            for i, action in enumerate(self.all_actions_a[:-1]):
                reaction = action.reaction.reaction
                pattern = Chem.MolFromSmarts(reaction.split(" >> ")[0].split(".")[0])
                if not mol.HasSubstructMatch(pattern):
                    continue
                pattern = Chem.MolFromSmarts(reaction.split(" >> ")[0].split(".")[1])
                for fragment in self.compatible_fragments[reaction]:
                    if Chem.MolFromSmiles(fragment).HasSubstructMatch(pattern):
                        mask[i] = True
                        break

        if not any(mask):
            return ReactionActionSpaceEarlyTerminate()
        return ReactionActionSpaceA(all_actions=self.all_actions_a, possible_actions_mask=mask)

    def _get_forward_action_spaces_b(
        self, state: ReactionStateB
    ) -> ReactionActionSpaceB | ReactionActionSpaceEarlyTerminate:
        possible_actions = [
            self.all_actions_b[i] for i in self.compatible_indices[state.reaction.reaction]
        ]

        if len(possible_actions) == 0:
            return ReactionActionSpaceEarlyTerminate()

        return ReactionActionSpaceB(possible_actions=tuple(possible_actions))

    def _get_forward_action_spaces_c(
        self, state: ReactionStateC
    ) -> ReactionActionSpaceC | ReactionActionSpaceEarlyTerminate:
        reaction = state.reaction.reaction
        reactants = [state.molecule.rdkit_mol, state.fragment.rdkit_mol]
        products = AllChem.ReactionFromSmarts(reaction).RunReactants(reactants)
        smiles_list = [Chem.MolToSmiles(mol[0]) for mol in products]
        molecule_list = set(Molecule(smiles) for smiles in smiles_list)

        disconnection = self.disconnections[self.reaction2index[reaction]]
        dis_rxn = AllChem.ReactionFromSmarts(disconnection)
        reactants_smiles = {state.molecule.smiles, state.fragment.smiles}
        possible_actions = []
        for molecule in molecule_list:
            prev_reactants_list = dis_rxn.RunReactants([molecule.rdkit_mol])
            for prev_reactants in prev_reactants_list:
                prev_reactants_smiles = {
                    canonicalize(prev_reactants) for prev_reactants in prev_reactants
                }
                if prev_reactants_smiles == reactants_smiles:
                    action = ReactionActionC(
                        input_molecule=state.molecule,
                        input_reaction=state.reaction,
                        input_fragment=state.fragment,
                        output_molecule=molecule,
                    )
                    self._cache[(molecule, state.num_reactions + 1)] = True
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
    ) -> ReactionActionSpace0 | ReactionActionSpaceC:
        if state.num_reactions == 0:
            mask = [False] * len(self.all_actions_0)
            fragment = state.molecule.smiles
            mask[self.fragment2index[fragment]] = True
            return ReactionActionSpace0(all_actions=self.all_actions_0, possible_actions_mask=mask)
        if state in self._additional_backward_cache:
            return self._additional_backward_cache[state]

        possible_actions = []
        for reaction, disconnection in zip(self.reactions, self.disconnections):
            rxn = AllChem.ReactionFromSmarts(disconnection)
            products = rxn.RunReactants((state.molecule.rdkit_mol,))
            parent_reaction_idx = self.reaction2index[reaction]
            parent_reaction = Reaction(reaction, parent_reaction_idx)
            for parent_molecule, parent_fragment in products:
                # Check if second reactant is a fragment (we are only checking the
                # second one since we impose the order in step function).
                if self._is_fragment(parent_fragment) and self._is_decomposable(
                    parent_molecule, state.num_reactions - 1
                ):
                    parent_molecule = Chem.MolToSmiles(parent_molecule)
                    parent_molecule = Molecule(parent_molecule)

                    parent_fragment = Chem.MolToSmiles(parent_fragment)
                    parent_fragment_idx = self.fragment2index[parent_fragment]
                    parent_fragment = Molecule(parent_fragment, parent_fragment_idx)

                    action = ReactionActionC(
                        input_molecule=parent_molecule,
                        input_reaction=parent_reaction,
                        input_fragment=parent_fragment,
                        output_molecule=state.molecule,
                    )

                    reactants = [action.input_molecule.rdkit_mol, action.input_fragment.rdkit_mol]

                    new_products = AllChem.ReactionFromSmarts(
                        parent_reaction.reaction
                    ).RunReactants(reactants)

                    all_product_smiles = [canonicalize(mol[0]) for mol in new_products]
                    if canonicalize(state.molecule.smiles) in all_product_smiles:
                        possible_actions.append(action)
        action_space = ReactionActionSpaceC(possible_actions=tuple(possible_actions))
        self._additional_backward_cache[state] = action_space
        if len(self._additional_backward_cache) > self._additional_backward_cache_size:
            self._additional_backward_cache.popitem()
        return action_space

    def _is_fragment(self, molecule: Mol) -> bool:
        return self.fragment2index.get(Chem.MolToSmiles(molecule)) is not None

    def _is_decomposable(self, molecule: Mol, n_reactions: int) -> bool:
        """
        Recursive helper for the decompose function. Returns true if the given Molecule
        can be disconnected fully into fragments in our library.

        Base case: is the molecule a fragment in our library? If yes, we're done.

        Recursive case: For each disconnect reaction, can we use this reaction to
            disconnect our molecule? Iterate through all fragment pairs generated by
            each reaction. If any fragment pair can be decomposed, return true.

        """

        smiles = Chem.MolToSmiles(molecule)
        if (smiles, n_reactions) in self._cache:
            return self._cache[(smiles, n_reactions)]

        if n_reactions == 0:
            return self._is_fragment(molecule)

        try:
            Chem.SanitizeMol(molecule)
        except:
            self._cache[(smiles, n_reactions)] = False
            return False

        # Decompose the molecule by all reverse reactions possible.
        # If ANY of them work then return True.
        for disconnection in self.disconnections:
            rxn = AllChem.ReactionFromSmarts(disconnection)
            frag_pairs = rxn.RunReactants((molecule,))
            for parent_molecule, parent_fragment in frag_pairs:
                # Both fragments in the pair have to be either decomposable or a
                # starter fragment for that pair to contain a viable parent.
                if self._is_fragment(parent_fragment):
                    if self._is_decomposable(parent_molecule, n_reactions - 1):
                        self._cache[(smiles, n_reactions)] = True
                        return True

        # We were unable to find a fully decomposable fragment pair.
        self._cache[(smiles, n_reactions)] = False
        return False

    def _get_backward_action_spaces_b(self, state: ReactionStateB) -> ReactionActionSpaceA:
        mask = [False] * len(self.all_actions_a)
        reaction = state.reaction.reaction
        mask[self.reaction2index[reaction]] = True
        return ReactionActionSpaceA(all_actions=self.all_actions_a, possible_actions_mask=mask)

    def _get_backward_action_spaces_c(self, state: ReactionStateC) -> ReactionActionSpaceB:
        mask = [False] * len(self.all_actions_b)
        mask[self.fragment2index[state.fragment.smiles]] = True
        possible_actions = (self.all_actions_b[self.fragment2index[state.fragment.smiles]],)
        return ReactionActionSpaceB(possible_actions=possible_actions)

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

    def _apply_forward_actions_a(
        self, state: ReactionStateA, action: ReactionActionA
    ) -> ReactionStateB | ReactionStateTerminal:
        if action.reaction is None:
            return ReactionStateTerminal(molecule=state.molecule, num_reactions=state.num_reactions)
        return ReactionStateB(
            molecule=state.molecule, reaction=action.reaction, num_reactions=state.num_reactions
        )

    def _apply_forward_actions_b(
        self, state: ReactionStateB, action: ReactionAction
    ) -> ReactionStateC:
        return ReactionStateC(
            molecule=state.molecule,
            reaction=state.reaction,
            fragment=action.fragment,
            num_reactions=state.num_reactions,
        )

    def _apply_forward_actions_c(
        self, state: ReactionStateC, action: ReactionAction
    ) -> ReactionStateA:
        return ReactionStateA(
            molecule=action.output_molecule, num_reactions=state.num_reactions + 1
        )

    def _apply_forward_actions_early_terminate(
        self, state: ReactionState, action: ReactionAction
    ) -> ReactionStateEarlyTerminal:
        return ReactionStateEarlyTerminal(previous_state=state)

    def apply_backward_actions(
        self, states: List[ReactionState], actions: List[ReactionAction]
    ) -> List[ReactionState]:
        new_states = []
        for state, action in zip(states, actions):
            state_type = type(state)
            new_state = self.backward_action_dict[state_type](state, action)
            new_states.append(new_state)
        return new_states

    def _apply_backward_actions_a(
        self, state: ReactionStateA, action: ReactionAction0 | ReactionActionC
    ) -> ReactionState0 | ReactionStateC:
        if isinstance(action, ReactionAction0):
            return ReactionState0()
        return ReactionStateC(
            molecule=action.input_molecule,
            reaction=action.input_reaction,
            fragment=action.input_fragment,
            num_reactions=state.num_reactions - 1,
        )

    def _apply_backward_actions_b(
        self, state: ReactionStateB, action: ReactionActionA
    ) -> ReactionStateA:
        return ReactionStateA(molecule=state.molecule, num_reactions=state.num_reactions)

    def _apply_backward_actions_c(
        self, state: ReactionStateC, action: ReactionActionB
    ) -> ReactionStateB:
        return ReactionStateB(
            molecule=state.molecule, reaction=state.reaction, num_reactions=state.num_reactions
        )

    def _apply_backward_actions_terminal(
        self, state: ReactionStateTerminal, action: ReactionActionA
    ) -> ReactionStateA:
        return ReactionStateA(molecule=state.molecule, num_reactions=state.num_reactions)

    def _apply_backward_actions_early_terminal(
        self, state: ReactionStateEarlyTerminal, action: ReactionAction
    ) -> ReactionState:
        return state.previous_state

    def get_terminal_mask(self, states: List[ReactionState]) -> List[bool]:
        return [
            isinstance(state, (ReactionStateTerminal, ReactionStateEarlyTerminal))
            for state in states
        ]

    def get_source_mask(self, states: List[ReactionState]) -> List[bool]:
        return [isinstance(state, ReactionState0) for state in states]

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

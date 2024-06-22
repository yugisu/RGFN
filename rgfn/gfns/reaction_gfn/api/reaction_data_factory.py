from copy import copy
from pathlib import Path
from typing import List

import gin
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from rgfn.gfns.reaction_gfn.api.reaction_api import Reaction


@gin.configurable()
class ReactionDataFactory:
    def __init__(
        self,
        reaction_path: str | Path,
        fragment_path: str | Path,
        reaction_subset: List[int] | None = None,
        swap_reactants: bool = True,
        docking: bool = False,
    ):
        if docking:
            sheet = "Reactions_Docking"
        else:
            sheet = "Reactions_NoDocking"
        self.reactions = list(pd.read_excel(reaction_path, sheet_name=sheet)["Reaction"])
        self.reactions = [r for r in self.reactions if isinstance(r, str)]
        self.reactions = [" >> ".join([x.strip() for x in r.split(">>")]) for r in self.reactions]

        self.reaction_subset = (
            list(range(len(self.reactions))) if not reaction_subset else reaction_subset
        )
        self.reactions = list(set(self.reactions))
        self.reactions = sorted(self.reactions)
        if swap_reactants:
            swapped_reactions = []
            for reaction in self.reactions:
                reactants, product = [x.strip() for x in reaction.split(">>")]
                reactants = reactants.split(".")
                swapped_reactions.append(f"{reactants[1]}.{reactants[0]} >> {product}")
            self.reactions += swapped_reactions
        assert len(set(self.reactions)) == len(self.reactions)

        # Calculate disconnections
        self.disconnections = [" >> ".join(r.split(" >> ")[::-1]) for r in self.reactions]

        # Load Fragments
        fragment_df = pd.read_csv(fragment_path)
        self.fragments = fragment_df["SMILES"].tolist()

        print(f"Using {len(self.fragments)} fragments from the subset.")

    def get_reactions(self) -> List[str]:
        return copy(self.reactions)

    def get_disconnections(self) -> List[str]:
        return copy(self.disconnections)

    def get_fragments(self) -> List[str]:
        return copy(self.fragments)


def canonicalize(mol):
    if isinstance(mol, Mol):
        smiles = Chem.MolToSmiles(mol)
    else:
        smiles = mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)

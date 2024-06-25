from copy import copy
from pathlib import Path
from typing import List

import gin
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import Reaction


@gin.configurable()
class ReactionDataFactory:
    def __init__(
        self,
        reaction_path: str | Path,
        fragment_path: str | Path,
        swap_reactants: bool = True,
        docking: bool = False,
    ):
        if docking:
            sheet = "Reactions_Docking"
        else:
            sheet = "Reactions_NoDocking"

        reactions = list(pd.read_excel(reaction_path, sheet_name=sheet)["Reaction"])
        reactions = [r for r in reactions if isinstance(r, str)]
        reactions = [" >> ".join([x.strip() for x in r.split(">>")]) for r in reactions]
        reactions = sorted(list(set(reactions)))

        if swap_reactants:
            swapped_reactions = []
            for reaction in reactions:
                reactants, product = [x.strip() for x in reaction.split(">>")]
                reactants = reactants.split(".")
                swapped_reactions.append(f"{reactants[1]}.{reactants[0]} >> {product}")
            reactions += swapped_reactions

        # Calculate disconnections
        disconnections = [" >> ".join(r.split(" >> ")[::-1]) for r in reactions]

        self.reactions = [Reaction(r, idx) for idx, r in enumerate(reactions)]
        self.disconnections = [Reaction(d, idx) for idx, d in enumerate(disconnections)]

        # Load Fragments
        fragment_df = pd.read_csv(fragment_path)
        fragments_list = fragment_df["SMILES"].tolist()

        self.fragments = [Molecule(f, idx=idx) for idx, f in enumerate(fragments_list)]

        assert len(set(self.reactions)) == len(self.reactions)
        assert len(set(self.fragments)) == len(self.fragments)

        print(f"Using {len(self.fragments)} fragments from the subset.")

    def get_reactions(self) -> List[Reaction]:
        return copy(self.reactions)

    def get_disconnections(self) -> List[Reaction]:
        return copy(self.disconnections)

    def get_fragments(self) -> List[Molecule]:
        return copy(self.fragments)

import abc
from dataclasses import InitVar, dataclass, field
from typing import Any, FrozenSet, Generic, List, NamedTuple, Tuple, TypeVar

from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmarts


@dataclass(frozen=True)
class Molecule:
    mol_or_smiles: InitVar[str | Chem.Mol] = field(init=True, repr=False, compare=False)
    smiles: str = field(init=False, hash=True, compare=True)
    rdkit_mol: Chem.Mol = field(init=False, repr=False, compare=False)
    idx: int | None = field(repr=False, compare=False, default=None, hash=False)
    valid: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, mol_or_smiles: str | Chem.Mol):
        rdkit_mol = (
            Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else mol_or_smiles
        )
        if rdkit_mol is None:
            canonical_smiles = mol_or_smiles
            valid = False
        else:
            if Chem.SanitizeMol(rdkit_mol, catchErrors=True) == 0:
                rdkit_mol = Chem.RemoveHs(rdkit_mol)
                valid = True
            else:
                valid = False
            canonical_smiles = Chem.MolToSmiles(rdkit_mol)

        object.__setattr__(self, "rdkit_mol", rdkit_mol)
        object.__setattr__(self, "smiles", canonical_smiles)
        object.__setattr__(self, "valid", valid)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.idx is not None:
            return f"F({self.idx})"
        else:
            return self.smiles


@dataclass(frozen=True)
class Reaction:
    reaction: str = field(hash=True, compare=True)
    rdkit_rxn: AllChem.ChemicalReaction = field(init=False, hash=False, compare=False)
    left_side_rdkit_patterns: Tuple[Chem.Mol, ...] = field(init=False, hash=False, compare=False)
    idx: int = field(hash=False, compare=False)

    def __post_init__(self):
        rxn = AllChem.ReactionFromSmarts(self.reaction)
        if rxn is None:
            raise ValueError(f"Invalid reaction SMILES: {self.reaction}")
        left_side_rdkit_patterns = tuple(
            MolFromSmarts(p) for p in self.reaction.split(" >> ")[0].split(".")
        )
        object.__setattr__(self, "rdkit_rxn", rxn)
        object.__setattr__(self, "left_side_rdkit_patterns", left_side_rdkit_patterns)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"R({self.reaction})"

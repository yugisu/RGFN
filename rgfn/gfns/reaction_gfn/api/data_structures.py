import abc
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Generic, List, NamedTuple, Tuple, TypeVar

from rdkit import Chem


@dataclass(frozen=True)
class Molecule:
    smiles: str = field(hash=True, compare=True)
    rdkit_mol: Chem.Mol = field(init=False, repr=False, compare=False)
    idx: int | None = field(repr=False, compare=False, default=None, hash=False)

    def __post_init__(self):
        rdkit_mol = Chem.MolFromSmiles(self.smiles)  # annotations induces order of atoms
        rdkit_mol = Chem.RemoveHs(rdkit_mol)
        for atom in rdkit_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical_smiles = Chem.MolToSmiles(rdkit_mol)
        rdkit_mol = Chem.MolFromSmiles(canonical_smiles)  # we set the canonical atoms order here
        if rdkit_mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles}")
        if "." in canonical_smiles:
            raise ValueError(
                f"Canonicalized SMILES contains multiple molecules: {canonical_smiles}"
            )
        object.__setattr__(self, "rdkit_mol", rdkit_mol)
        object.__setattr__(self, "smiles", canonical_smiles)

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
    idx: int = field(hash=False, compare=False)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"R({self.idx})"

from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Protocol,
    Tuple,
    TypeVar,
)

from rdkit import Chem


class Comparable(Protocol):
    def __lt__(self, __other) -> bool:
        ...


TComparable = TypeVar("TComparable", bound=Comparable)
THashable = TypeVar("THashable", bound=Hashable)


class Bag(Collection, Generic[THashable]):
    def __init__(self, values: Iterable[THashable] = ()) -> None:
        self._container: Dict[THashable, int] = {}
        self._count = 0
        for value in values:
            self._add(value)

    @classmethod
    def from_bags(cls, bags: Iterable["Bag[THashable]"]) -> "Bag[THashable]":
        container: Dict[THashable, int] = {}
        total_count = 0
        for bag in bags:
            for element, count in bag._container.items():
                container[element] = container.get(element, 0) + count
                total_count += count
        new_bag = cls()
        new_bag._container = container
        new_bag._count = total_count
        return new_bag

    def is_subset(self, other: "Bag[THashable]") -> bool:
        for element, count in self._container.items():
            if count > other._container.get(element, 0):
                return False
        return True

    def is_superset(self, other: "Bag[THashable]") -> bool:
        return other.is_subset(self)

    def _add(self, element: THashable) -> None:
        self._container[element] = self._container.get(element, 0) + 1
        self._count += 1

    def __iter__(self) -> Iterator[THashable]:
        for element, count in self._container.items():
            for _ in range(count):
                yield element

    def __contains__(self, element: Any) -> bool:
        return element in self._container

    def __eq__(self, other) -> bool:
        if isinstance(other, Bag):
            return self._container == other._container
        else:
            return False

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return repr(self._container)

    def __hash__(self) -> int:
        return hash(frozenset(self._container.items()))

    def __or__(self, other: Iterable[THashable]) -> "Bag[THashable]":
        return Bag.from_bags([self, Bag(other)])


class SortedList(Collection, Generic[TComparable]):
    def __init__(self, values: Iterable[TComparable] = ()) -> None:
        self._values = tuple(sorted(values))

    def __iter__(self) -> Iterator[TComparable]:
        return iter(self._values)

    def __contains__(self, element: Any) -> bool:
        return element in self._values

    def __eq__(self, other) -> bool:
        if isinstance(other, SortedList):
            return self._values == other._values
        else:
            return False

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return repr(self._values)

    def __hash__(self) -> int:
        return hash(self._values)

    def __getitem__(self, idx: int):
        return self._values[idx]


@dataclass(frozen=True)
class Molecule:
    smiles: str = field(hash=True, compare=True)
    rdkit_mol: Chem.Mol = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        rdkit_mol = Chem.MolFromSmiles(self.smiles)  # annotations induces order of atoms
        rdkit_mol = Chem.RemoveHs(rdkit_mol)
        for atom in rdkit_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical_smiles = Chem.MolToSmiles(rdkit_mol)
        rdkit_mol = Chem.MolFromSmiles(canonical_smiles)  # we set the canonical atoms order here
        if "." in canonical_smiles:
            raise ValueError(
                f"Canonicalized SMILES contains multiple molecules: {canonical_smiles}"
            )
        object.__setattr__(self, "rdkit_mol", rdkit_mol)
        object.__setattr__(self, "smiles", canonical_smiles)


@dataclass(frozen=True, order=True)
class Pattern:
    smarts: str = field(hash=True, compare=True)
    rdkit_mol: Chem.Mol = field(init=False, repr=False, compare=False)
    mappable_symbols: Bag[str] = field(init=False, repr=False, compare=False)
    idx: int | None = field(repr=False, compare=False, default=None)
    symbol_to_indices: Dict[str, List[int]] = field(init=False, repr=False, compare=False)

    def __hash__(self):
        if self.idx is None:
            return hash(self.smarts)
        else:
            return self.idx

    def symmetric(self) -> bool:
        symbol_list = [
            get_symbol(atom, include_aromaticity=True) for atom in self.rdkit_mol.GetAtoms()
        ]
        symbols = "".join(symbol_list)
        return symbols == symbols[::-1]

    def __post_init__(self):
        smarts = self.smarts[1:-1] if self.smarts.startswith("(") else self.smarts
        rdkit_mol = Chem.MolFromSmarts(smarts)
        mappable_list = []
        symbol_to_indices = defaultdict(list)
        for idx, atom in enumerate(rdkit_mol.GetAtoms()):
            if atom.GetAtomMapNum() != 0:
                atom.SetAtomMapNum(1)  # mappable
                mappable_list.append(get_symbol(atom))
                symbol_to_indices[get_symbol(atom)].append(idx)
            else:
                atom.SetAtomMapNum(0)  # not mappable

        canonical_smarts = Chem.MolToSmarts(rdkit_mol)
        object.__setattr__(self, "smarts", canonical_smarts)
        object.__setattr__(self, "rdkit_mol", rdkit_mol)
        object.__setattr__(self, "mappable_symbols", Bag(mappable_list))
        object.__setattr__(self, "symbol_to_indices", symbol_to_indices)


@dataclass(frozen=True, order=True)
class ReactantPattern(Pattern):
    charge_diffs: Tuple[int, ...] = field(hash=True, compare=True, default=())
    hydrogen_diffs: Tuple[int, ...] = field(hash=True, compare=True, default=())
    chiral_diffs: Tuple[int, ...] = field(hash=True, compare=True, default=())

    def __post_init__(self):
        super().__post_init__()
        rdkit_mol = self.rdkit_mol
        mappable_atoms = [atom for atom in rdkit_mol.GetAtoms() if atom.GetAtomMapNum() == 1]
        charge_diffs = self.charge_diffs if self.charge_diffs else tuple(0 for _ in mappable_atoms)
        hydrogen_diffs = (
            self.hydrogen_diffs if self.hydrogen_diffs else tuple(0 for _ in mappable_atoms)
        )
        chiral_diffs = self.chiral_diffs if self.chiral_diffs else tuple(0 for _ in mappable_atoms)
        for idx, atom in enumerate(mappable_atoms):
            atom.SetIntProp("charge_diff", charge_diffs[idx])
            atom.SetIntProp("hydrogen_diff", hydrogen_diffs[idx])
            atom.SetIntProp("chiral_diff", chiral_diffs[idx])
        object.__setattr__(self, "rdkit_mol", rdkit_mol)


@dataclass(frozen=True)
class Reaction:
    product: Molecule
    reactants: Bag[Molecule]


@dataclass(frozen=True)
class MappingTuple:
    product_node: int
    reactant: int
    reactant_node: int


def get_symbol(atom: Chem.Atom, include_aromaticity: bool = False) -> str:
    symbol = atom.GetSymbol()
    if include_aromaticity and atom.GetIsAromatic():
        symbol = symbol.lower()
    return symbol

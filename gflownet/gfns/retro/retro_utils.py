import copy
from typing import Any, Dict, Iterable

import torch
from rdkit import Chem
from rdkit.Chem import ChiralType
from torchtyping import TensorType

from gflownet.gfns.retro.api.data_structures import (
    Bag,
    Pattern,
    ReactantPattern,
    SortedList,
)
from gflownet.gfns.retro.api.retro_api import (
    FirstPhaseRetroActionSpace,
    FirstPhaseRetroState,
    MappingTuple,
    RetroActionSpace,
    RetroState,
    SecondPhaseRetroActionSpace,
    SecondPhaseRetroState,
    TerminalRetroState,
    ThirdPhaseRetroActionSpace,
    ThirdPhaseRetroState,
)


def gumbel_cdf(x: TensorType[float], mean: float = 0.0, beta: float = 1):
    return torch.exp(-torch.exp(-(x - mean) / beta))


def get_backward_template(
    product_pattern: Pattern,
    reactants_patterns: SortedList[ReactantPattern],
    atom_mapping: Iterable[MappingTuple],
) -> Dict[str, Any]:
    product_pattern_mol = copy.deepcopy(product_pattern.rdkit_mol)
    reactant_pattern_mols = [copy.deepcopy(pattern.rdkit_mol) for pattern in reactants_patterns]
    for mapping in atom_mapping:
        idx = mapping.product_node + 1
        product_pattern_mol.GetAtomWithIdx(mapping.product_node).SetAtomMapNum(idx)
        reactant_pattern_mols[mapping.reactant].GetAtomWithIdx(mapping.reactant_node).SetAtomMapNum(
            idx
        )

    product_smarts = Chem.MolToSmarts(product_pattern_mol)
    reactants_smarts_list = [f"({Chem.MolToSmarts(mol)})" for mol in reactant_pattern_mols]
    reactants_smarts = ".".join(reactants_smarts_list)
    template = f"({product_smarts})>>{reactants_smarts}"
    return {
        "template": template,
        "product_pattern_mol": product_pattern_mol,
        "reactant_pattern_mols": reactant_pattern_mols,
    }


chiral_type_map = {
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_TETRAHEDRAL_CW: -1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1,
}
chiral_type_map_inv = {v: k for k, v in chiral_type_map.items()}

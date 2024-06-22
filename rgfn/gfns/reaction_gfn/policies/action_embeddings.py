import abc
import math
from typing import List

import gin
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch import nn
from torch.nn import init
from torchtyping import TensorType

from rgfn.gfns.reaction_gfn.api.reaction_data_factory import ReactionDataFactory


class FragmentEmbeddingBase(abc.ABC, nn.Module):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fragments = data_factory.get_fragments()

    @abc.abstractmethod
    def get_embeddings(self) -> TensorType[float]:
        pass


@gin.configurable()
class FragmentOneHotEmbedding(FragmentEmbeddingBase):
    def __init__(self, data_factory: ReactionDataFactory, hidden_dim: int = 64):
        super().__init__(data_factory, hidden_dim)
        self.weights = nn.Parameter(
            torch.empty(len(self.fragments), hidden_dim), requires_grad=True
        )
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def get_embeddings(self) -> TensorType[float]:
        return self.weights


@gin.configurable()
class FragmentFingerprintEmbedding(FragmentEmbeddingBase):
    def __init__(
        self,
        data_factory: ReactionDataFactory,
        fingerprint_list: List[str],
        random_linear_compression: bool,
        hidden_dim: int = 64,
        one_hot_weight: float = 1.0,
    ):
        super().__init__(data_factory, hidden_dim)
        self.fingerprint_list = fingerprint_list

        self.one_hot_weight = one_hot_weight
        self.one_hot = nn.Parameter(
            torch.empty(len(self.fragments), hidden_dim), requires_grad=True
        )
        init.kaiming_uniform_(self.one_hot, a=math.sqrt(5))

        self.all_fingerprints = self._get_fingerprints()
        if random_linear_compression:
            random_projection = torch.empty(self.all_fingerprints.shape[-1], hidden_dim)
            init.kaiming_uniform_(random_projection, a=math.sqrt(5))
            self.all_fingerprints = torch.matmul(
                self.all_fingerprints, random_projection
            )  # (num_fragments, hidden_dim)
        self.all_fingerprints = nn.Parameter(self.all_fingerprints, requires_grad=False)
        self.linear = nn.Linear(self.all_fingerprints.shape[-1], hidden_dim)

    def _get_fingerprints(self) -> TensorType[float]:
        fps_list = []
        for smiles in self.fragments:
            mol = Chem.MolFromSmiles(smiles)
            Chem.RemoveHs(mol)
            for fp_type in self.fingerprint_list:
                fps = []
                if fp_type == "maccs":
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    array = np.zeros((0,), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, array)
                    fps.append(array)
                elif fp_type == "ecfp":
                    fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=20480)
                    array = np.zeros((0,), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, array)
                    fps.append(array)
            fps = np.concatenate(fps, axis=0)
            fps_list.append(fps)
        fps_numpy = np.stack(fps_list, axis=0)
        # remove zero columns
        fps_numpy = fps_numpy[:, np.any(fps_numpy, axis=0)]
        return torch.tensor(fps_numpy).float()

    def get_embeddings(self) -> TensorType[float]:
        fingerprints = self.linear(self.all_fingerprints)
        if self.one_hot_weight > 0:
            return fingerprints + self.one_hot_weight * self.one_hot
        return fingerprints

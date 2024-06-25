import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Set

import gin
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase
from rgfn.trainer.metrics.metric_base import MetricsBase


@gin.configurable()
class QED(MetricsBase):
    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        terminal_states = trajectories.get_last_states_flat()
        qed_scores_list = []
        for state in terminal_states:
            if isinstance(state, ReactionStateTerminal):
                qed_score = qed(state.molecule.rdkit_mol)
                qed_scores_list.append(qed_score)
        return {"qed": np.mean(qed_scores_list)}


@gin.configurable()
class NumScaffoldsFound(MetricsBase):
    def __init__(
        self,
        proxy_value_threshold_list: List[float],
        proxy_component_name: str | None,
        proxy_higher_better: bool = True,
    ):
        super().__init__()
        self.proxy_value_threshold_list = proxy_value_threshold_list
        self.proxy_higher_better = proxy_higher_better
        self.threshold_to_set: Dict[float, Set[str]] = {
            threshold: set() for threshold in proxy_value_threshold_list
        }
        self.proxy_component_name = proxy_component_name

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        reward_outputs = trajectories.get_reward_outputs()
        terminal_states = trajectories.get_last_states_flat()
        values = (
            reward_outputs.proxy
            if self.proxy_component_name is None
            else reward_outputs.proxy_components[self.proxy_component_name]
        )
        for state, proxy_value in zip(terminal_states, values):
            for threshold in self.proxy_value_threshold_list:
                if (self.proxy_higher_better and proxy_value.item() > threshold) or (
                    not self.proxy_higher_better and proxy_value.item() < threshold
                ):
                    self.threshold_to_set[threshold].add(
                        MurckoScaffoldSmiles(state.molecule.smiles)
                    )

        return {
            f"num_scaffolds_{threshold}": len(self.threshold_to_set[threshold])
            for threshold in self.proxy_value_threshold_list
        }


@gin.configurable()
class UniqueMolecules(MetricsBase):
    def __init__(self, run_dir: str, dump_every_n: int | None = None):
        super().__init__()
        self.molecules: Dict[Any, Any] = {}
        self.dump_every_n = dump_every_n
        self.iterations = 0
        self.dump_path = Path(run_dir) / "unique_molecules"
        self.dump_path.mkdir(exist_ok=True, parents=True)

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        terminal_states = trajectories.get_last_states_flat()
        proxy_scores = trajectories.get_reward_outputs().proxy
        proxy_terms = trajectories.get_reward_outputs().proxy_components
        for i, state in enumerate(terminal_states):
            if isinstance(state, ReactionStateTerminal):
                output = {"score": proxy_scores[i].item()}
                if proxy_terms is not None:
                    for name, values in proxy_terms.items():
                        output[f"term_{name}"] = values[i].item()

                self.molecules[state.molecule.smiles] = output

        if (
            self.dump_every_n is not None
            and (self.iterations % self.dump_every_n == 0)
            and self.iterations > 0
        ):
            with open(self.dump_path / f"molecules_{self.iterations}.json", "w") as fp:
                json.dump(self.molecules, fp)
        self.iterations += 1

        return {"num_unique_molecules": len(self.molecules)}


@gin.configurable()
class AllMolecules(MetricsBase):
    def __init__(self, run_dir: str, dump_every_n: int | None = None):
        super().__init__()
        self.molecules: list = []
        self.dump_every_n = dump_every_n
        self.iterations = 0
        self.dump_path = Path(run_dir) / "all_molecules"
        self.dump_path.mkdir(exist_ok=True, parents=True)

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        terminal_states = trajectories.get_last_states_flat()
        proxy_scores = trajectories.get_reward_outputs().proxy
        proxy_terms = trajectories.get_reward_outputs().proxy_components
        for i, state in enumerate(terminal_states):
            if isinstance(state, ReactionStateTerminal):
                output = {"score": proxy_scores[i].item()}
                if proxy_terms is not None:
                    for name, values in proxy_terms.items():
                        output[f"term_{name}"] = values[i].item()

                self.molecules.append((state.molecule.smiles, output))

        if (
            self.dump_every_n is not None
            and (self.iterations % self.dump_every_n == 0)
            and self.iterations > 0
        ):
            with open(self.dump_path / f"molecules_{self.iterations}.txt", "w") as fp:
                for smiles, output in self.molecules:
                    fp.write(f"{smiles}, {output}\n")
        self.iterations += 1

        return {"num_visited_molecules": len(self.molecules)}


@gin.configurable()
class TanimotoSimilarityModes(MetricsBase):
    def __init__(
        self,
        run_dir: str,
        proxy: CachedProxyBase,
        term_name: str = "value",
        proxy_term_threshold: float = -np.inf,
        similarity_threshold: float = 0.7,
        max_modes: int | None = 5000,
        compute_every_n: int = 1,
    ):
        super().__init__()
        self.proxy = proxy
        self.term_name = term_name
        self.proxy_term_threshold = proxy_term_threshold
        self.similarity_threshold = similarity_threshold
        self.max_modes = max_modes
        self.compute_every_n = compute_every_n
        self.iterations = 0
        self.dump_path = Path(run_dir) / "modes"
        self.dump_path.mkdir(exist_ok=True, parents=True)

    def _extract_top_sorted_smiles(self) -> Dict[str, float | Dict[str, float]]:
        """
        Fetches SMILES from proxy cache, extracts the ones with reward above thresholds,
        and sorts them by reward.
        """
        if isinstance(next(iter(self.proxy.cache.values())), float):
            cache = {k: {"value": v} for k, v in self.proxy.cache.items()}
        else:
            cache = self.proxy.cache

        d = {}
        for state, scores in cache.items():
            if (
                isinstance(state, ReactionStateTerminal)
                and scores[self.term_name] >= self.proxy_term_threshold
            ):
                d[state.molecule.smiles] = scores
        d = dict(sorted(d.items(), key=lambda item: item[1][self.term_name], reverse=True))

        return d

    def _extract_modes(self) -> Dict[str, float | Dict[str, float]]:
        d = self._extract_top_sorted_smiles()
        mols = [Chem.MolFromSmiles(x) for x in d.keys()]
        ecfps = [
            AllChem.GetMorganFingerprintAsBitVect(
                m, radius=3, nBits=2048, useFeatures=False, useChirality=False
            )
            for m in mols
        ]
        modes = []
        for mol, ecfp, r, smiles in zip(mols, ecfps, d.values(), d.keys()):
            if len(modes) >= self.max_modes:
                break
            is_mode = True
            for mode in modes:
                if DataStructs.TanimotoSimilarity(ecfp, mode[1]) > self.similarity_threshold:
                    is_mode = False
                    break
            if is_mode:
                modes.append((mol, ecfp, r, smiles))
        return {m[3]: m[2] for m in modes}

    @staticmethod
    def _modes_to_df(modes: Dict[str, float | Dict[str, float]]) -> pd.DataFrame:
        reward_terms = [k for k in next(iter(modes.values())).keys() if k != "value"]

        rows = []
        for smiles, scores in modes.items():
            reward = scores["value"]
            mol = Chem.MolFromSmiles(smiles)
            heavy_atoms = mol.GetNumHeavyAtoms()
            efficiency = reward / heavy_atoms
            row = (
                [
                    "",
                    f"{np.round(reward, 2):.2f}",
                ]
                + [f"{np.round(scores[term], 2):.2f}" for term in reward_terms]
                + [
                    f"{np.round(Descriptors.ExactMolWt(mol), 2):.2f}",
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    heavy_atoms,
                    f"{np.round(Descriptors.MolLogP(mol), 3):.3f}",
                    Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
                    f"{np.round(efficiency, 4):.4f}",
                    smiles,
                ]
            )
            rows.append(row)

        columns = (
            [
                "Molecule",
                "Reward",
            ]
            + [f"Reward ({term})" for term in reward_terms]
            + [
                "MW",
                "H-bond donors",
                "H-bond acceptors",
                "Heavy atoms",
                "cLogP",
                "Rotatable bonds",
                "Ligand efficiency",
                "SMILES",
            ]
        )

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _save_modes_xlsx(df: pd.DataFrame, file_path: Path | str):
        writer = pd.ExcelWriter(file_path, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="Molecules", index=False)
        worksheet = writer.sheets["Molecules"]
        worksheet.set_column(0, 0, 21)
        worksheet.set_column(1, len(df.columns), 15)

        directory = TemporaryDirectory()

        for i, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            image_path = Path(directory.name) / f"molecule_{i}.png"
            Draw.MolToFile(mol, filename=image_path, size=(150, 150))

            worksheet.set_row(i + 1, 120)
            worksheet.insert_image(i + 1, 0, image_path)

        writer.book.close()
        directory.cleanup()

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        if self.iterations % self.compute_every_n == 0:
            modes = self._extract_modes()

            if len(modes) > 0:
                df = self._modes_to_df(modes)
                self._save_modes_xlsx(df, self.dump_path / f"modes_{self.iterations}.xlsx")

            self.iterations += 1

            return {"num_modes": len(modes)}
        else:
            self.iterations += 1

            return {}


@gin.configurable()
class FractionEarlyTerminate(MetricsBase):
    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        terminal_states = trajectories.get_last_states_flat()
        num_early_terminate = sum(
            [1 for state in terminal_states if isinstance(state, ReactionStateEarlyTerminal)]
        )
        return {"fraction_early_terminate": num_early_terminate / len(terminal_states)}

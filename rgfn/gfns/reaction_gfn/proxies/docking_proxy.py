import subprocess
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import gin
import numpy as np
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

from rgfn.api.type_variables import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase

RECEPTOR_ROOT_PATH = Path(__file__).parents[4] / "data" / "targets"
RECEPTOR_CENTERS = {
    "Mpro": [-20.458, 18.109, -26.914],
    "TBLR1": [-1.014, 42.097, 39.750],
    "ClpP": [-38.127, 45.671, -20.898],
    "LRRK2_WD40": [-16.386, -15.911, 7.779],
    "sEH": [-13.4, 26.3, -13.3],
}
RECEPTOR_BOX_SIZES = {
    "Mpro": [18, 18, 18],
    "TBLR1": [18, 18, 18],
    "ClpP": [17, 17, 17],
    "LRRK2_WD40": [25, 25, 25],
    "sEH": [20.013, 16.3, 18.5],
}
RECEPTOR_PATHS = {k: RECEPTOR_ROOT_PATH / f"{k}.pdbqt" for k in RECEPTOR_CENTERS.keys()}


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@gin.configurable()
class DockingMoleculeProxy(CachedProxyBase[ReactionState]):
    def __init__(
        self,
        qv_dir: Union[Path, str],
        receptor_path: Optional[Union[Path, str]] = None,
        receptor_name: Optional[str] = None,
        center: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        norm: float = 1.0,
        failed_score: float = 0.0,
        conformer_attempts: int = 20,
        docking_attempts: int = 10,
        docking_batch_size: int = 25,
    ):
        super().__init__()

        if receptor_path is None and receptor_name is None:
            raise ValueError("Expected either receptor_path or receptor_name to be specified.")
        if receptor_path is not None and receptor_name is not None:
            raise ValueError(
                "Expected only one of receptor_path and receptor_name to be specified."
            )

        if receptor_name is not None:
            assert center is None

            self.receptor_path = RECEPTOR_PATHS[receptor_name]
            self.center = RECEPTOR_CENTERS[receptor_name]
        else:
            self.receptor_path = receptor_path
            self.center = center

        self.qv_dir = qv_dir
        self.size = RECEPTOR_BOX_SIZES[receptor_name] if size is None else size
        self.norm = norm
        self.failed_score = failed_score
        self.conformer_attempts = conformer_attempts
        self.docking_attempts = docking_attempts
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}
        self.batch_size = docking_batch_size

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_proxy_output(self, states: List[TState]) -> List[float]:
        smiles = [state.molecule.smiles for state in states]
        scores = []
        for chunk in _chunks(smiles, self.batch_size):
            scores += self.dock_batch_qv2gpu(chunk)
        scores = [np.clip(-x / self.norm, a_min=0, a_max=np.inf) for x in scores]
        return scores

    def _docking_attempt(self, smiles, n):
        """
        Uses customized QuickVina2-GPU (Tang et al.) implementation to
        calculate docking score against target of choice.

        1. Unique temp directory (for scores, config, and pdbqt inputs) is created.
        2. Molecules are protonated and relaxed, and converted to pdbqt.
        3. QV2GPU runs on the molecule pdbqts.
        4. Scores are read from the output file.
        5. Temp directory is removed.

        Note: Failure at any point in the pipeline (reading molecule, pdbqt conversion,
            score calculation) returns self.failed_score for that molecule.
        """
        ligand_directory = TemporaryDirectory(suffix="_ligand")
        config_directory = TemporaryDirectory(suffix="_config")
        config_path = str(Path(config_directory.name) / "config.txt")
        scores_path = str(Path(ligand_directory.name) / "scores.txt")

        qv2cfg = textwrap.dedent(
            f"""
            receptor = {self.receptor_path}
            ligand_directory = {ligand_directory.name}
            opencl_binary_path = {self.qv_dir}
            center_x = {self.center[0]}
            center_y = {self.center[1]}
            center_z = {self.center[2]}
            size_x = {self.size[0]}
            size_y = {self.size[1]}
            size_z = {self.size[2]}
            thread = 8000
            num_modes = 15
        """
        )

        with open(config_path, "w") as file:
            file.write(qv2cfg)

        initial_pdbqts = []
        docked_pdbqts = []
        indices = []
        count = 0

        for idx, smi in enumerate(smiles):
            attempt = 0
            pdbqt_string = None

            while pdbqt_string is None and (attempt == 0 or attempt < self.conformer_attempts):
                attempt += 1

                try:
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    AllChem.UFFOptimizeMolecule(mol)
                    preparator = MoleculePreparation()
                    mol_setups = preparator.prepare(mol)
                    setup = mol_setups[0]
                    pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(setup)
                except Exception as e:
                    print(f"Failed embedding attempt #{attempt} with error: '{e}'.")

            if pdbqt_string is None:
                continue

            initial_pdbqts.append(pdbqt_string)
            indices.append(idx)

            output_file_path = str(Path(ligand_directory.name) / f"{str(count)}.pdbqt")
            with open(output_file_path, "w") as file:
                file.write(pdbqt_string)

            count += 1

        command = [
            "./QuickVina2-GPU-2-1",
            "--config",
            config_path,
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.qv_dir,
        )

        if not Path(scores_path).exists():
            print(f"Failed score loading attempt #{n}: {result.stderr}")

            scores = None
        else:
            scores = [self.failed_score for _ in smiles]
            with open(scores_path, "r") as file:
                for i, line in enumerate(file):
                    try:
                        score = float(line.strip())

                    except ValueError:
                        print(f"Failed line reading attempt: '{line.strip()}'.")
                        score = self.failed_score

                    scores[indices[i]] = score

                    try:
                        out_lig_path = str(Path(ligand_directory.name) / f"{str(i)}_out.pdbqt")
                        with open(out_lig_path, "r") as file:
                            docked_pdbqt = file.read()

                    except ValueError:
                        print(f"Failed to read docked ligand output .pdbqt file: '{out_lig_path}'.")
                        docked_pdbqt = None

                    docked_pdbqts.append(docked_pdbqt)

        ligand_directory.cleanup()
        config_directory.cleanup()

        return scores, initial_pdbqts, docked_pdbqts

    def dock_batch_qv2gpu(self, smiles):
        for attempt in range(1, self.docking_attempts + 1):
            scores, initial_pdbqts, docked_pdbqts = self._docking_attempt(smiles, attempt)
            if scores is not None:
                return scores
        return [self.failed_score] * len(smiles)

import os
from pathlib import Path
from typing import List, Optional, Type, Union

import gin
import numpy as np

from rgfn import ROOT_DIR
from rgfn.api.env_base import TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.gfns.reaction_gfn.preparators.preparators import (
    BasePreparator,
    MeekoLigandPreparator,
)
from rgfn.gfns.reaction_gfn.proxies.docking_proxy.gnina_wrapper import GninaRescorer
from rgfn.gfns.reaction_gfn.proxies.docking_proxy.vinagpu_wrapper import VinaDocking
from rgfn.shared.proxies.cached_proxy import CachedProxyBase

RECEPTOR_ROOT_PATH = ROOT_DIR / "data" / "targets"
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
        preparator_class: Type[BasePreparator] = MeekoLigandPreparator,
        vina_mode: str = "QuickVina2",
        gnina: bool = False,
        receptor_path: Optional[Union[Path, str]] = None,
        receptor_name: Optional[str] = None,
        center: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        print_msgs: bool = False,
        norm: float = 1.0,
        failed_score: float = 0.0,
        conformer_attempts: int = 20,
        n_conformers: int = 1,
        docking_attempts: int = 10,
        docking_batch_size: int = 25,
        exhaustiveness: int = 8000,
        n_gpu: int = 1,
        n_cpu: Optional[int] = None,
    ):
        """
        Parameters:
            qv_dir (Union[Path, str]): Directory containing the Vina executable.
            preparator_class (Type[BasePreparator]): Class for preparing ligands. Default is MeekoLigandPreparator.
            vina_mode (str): Vina-GPU-2.1 implementation to use. Options are "QuickVina2", "AutoDock-Vina", or "QuickVina-W". Default is "QuickVina2".
            gnina (bool): Whether to use GNINA for rescoring. Default is False.
            receptor_path (Optional[Union[Path, str]]): Path to the receptor file. Required if receptor_name is not provided.
            receptor_name (Optional[str]): Name of the receptor. Required if receptor_path is not provided.
            center (Optional[List[float]]): Center coordinates of the docking box.
            size (Optional[List[float]]): Size of the docking box. If not provided, uses predefined sizes for known receptors.
            print_msgs (bool): Whether to print messages during docking. Default is False.
            norm (float): Normalization factor for docking scores. Default is 1.0.
            failed_score (float): Score to assign when docking fails. Default is 0.0.
            conformer_attempts (int): Number of attempts for conformer generation. Default is 20.
            n_conformers (int): Number of conformers to generate per molecule. Default is 1.
            docking_attempts (int): Number of docking attempts per molecule. Default is 10.
            docking_batch_size (int): Batch size for docking. Default is 25.
            exhaustiveness (int): Exhaustiveness parameter for Vina. Default is 8000. Note: Minimum is 1000.
            n_gpu (int): Number of GPUs to use. Default is 1.
            n_cpu (Optional[int]): Number of CPUs to use. If None, uses all available CPUs.
        """

        super().__init__()

        if receptor_path is None and receptor_name is None:
            raise ValueError("Expected either receptor_path or receptor_name to be specified.")
        if receptor_path is not None and receptor_name is not None:
            raise ValueError(
                "Expected only one of receptor_path and receptor_name to be specified."
            )
        if vina_mode not in ["QuickVina2", "AutoDock-Vina", "QuickVina-W"]:
            raise ValueError(
                "Argument 'vina_mode' must be one of 'QuickVina2', 'AutoDock-Vina', or 'QuickVina-W'."
            )

        if receptor_name is not None:
            assert center is None

            self.receptor_path = RECEPTOR_PATHS[receptor_name]
            self.center = RECEPTOR_CENTERS[receptor_name]
        else:
            self.receptor_path = receptor_path
            self.center = center

        self.vina_mode = vina_mode
        self.qv_dir = qv_dir
        self.gnina = gnina
        self.size = RECEPTOR_BOX_SIZES[receptor_name] if size is None else size
        self.print_msgs = print_msgs
        self.preparator_class = preparator_class
        self.norm = norm
        self.failed_score = failed_score
        self.conformer_attempts = conformer_attempts
        self.n_conformers = n_conformers
        self.docking_attempts = docking_attempts
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}
        self.batch_size = docking_batch_size
        self.exhaustiveness = exhaustiveness
        self.n_gpu = n_gpu
        self.n_cpu = n_cpu

        self.preparator = self.preparator_class(
            conformer_attempts=self.conformer_attempts,
            n_conformers=self.n_conformers,
            num_cpus=self.n_cpu,
        )

        vina_fullpath = os.path.realpath(f"{self.qv_dir}/{self.vina_mode}-GPU-2.1")
        self.docking_module_gpu = VinaDocking(
            f"./{self.vina_mode}-GPU-2-1",
            receptor_pdbqt_file=self.receptor_path,
            center_pos=self.center,
            size=self.size,
            n_conformers=self.n_conformers,
            vina_cwd=vina_fullpath,
            get_pose_str=True,
            preparator=self.preparator,
            timeout_duration=None,
            debug=False,
            print_msgs=self.print_msgs,
            print_vina_output=False,
            gpu_ids=list(range(self.n_gpu)),
            docking_attempts=self.docking_attempts,
            additional_vina_args={
                "thread": self.exhaustiveness,
                "opencl_binary_path": vina_fullpath,
                "num_modes": 1,
            },
        )

        if self.gnina:
            self.gnina_rescorer = GninaRescorer(
                receptor_pdbqt_file=self.receptor_path,
                pose_format="pdbqt",
                center_pos=self.center,
                size=self.size,
            )

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
            scores_chunk, docked_pdbqts = self.dock_batch_qv2gpu(chunk)
            if scores_chunk is not None:
                scores_chunk = [self.failed_score if s is None else s for s in scores_chunk]
                docked_pdbqts = ["" if p is None else p for p in docked_pdbqts]
                scores += scores_chunk
            else:
                scores += [self.failed_score] * len(chunk)

        # scores prior to this point are negative
        scores = [np.clip(-x / self.norm, a_min=0, a_max=np.inf) for x in scores]
        return scores

    def dock_batch_qv2gpu(self, smiles):
        """
        Uses customized QuickVina2-GPU (Tang et al.) implementation to
        calculate docking score against target of choice.

        Note: Failure at any point in the pipeline (reading molecule, pdbqt conversion,
            score calculation) returns self.failed_score for that molecule.
        """

        scores, docked_pdbqts = self.docking_module_gpu(smiles)

        if self.n_conformers == 1 or (not scores or not docked_pdbqts):
            return scores, docked_pdbqts

        else:
            all_best_scores = []
            all_best_poses = []
            if self.gnina:
                if self.print_msgs:
                    print("Rescoring with gnina...")
                pose_batches = _chunks(docked_pdbqts, self.n_conformers)
                for batch in pose_batches:
                    score, pose = self.gnina_rescorer(list(batch))
                    all_best_scores.append(score * -1)  # gnina returns a positive score
                    all_best_poses.append(pose)

                if self.print_msgs:
                    print("Rescoring complete.")

            else:  # we're just ranking the poses by qv2gpu score
                score_pose_batches = _chunks(list(zip(scores, docked_pdbqts)), self.n_conformers)
                for batch in score_pose_batches:
                    successful_docks = [tup for tup in batch if tup[0] is not None]

                    # Total conformer generation / docking failure for a single SMILES
                    if len(successful_docks) == 0:
                        all_best_scores.append(self.failed_score)
                        all_best_poses.append(None)

                    else:
                        best_pair = sorted(successful_docks, key=lambda x: x[0])[0]
                        all_best_scores.append(best_pair[0])
                        all_best_poses.append(best_pair[1])

            return all_best_scores, all_best_poses

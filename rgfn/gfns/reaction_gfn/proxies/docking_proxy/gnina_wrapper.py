# Adapted from code in PyVina https://github.com/Pixelatory/PyVina
# Copyright (c) 2024 Nicholas Aksamit
# Licensed under the MIT License

import re
import subprocess
import tempfile
from typing import List

from rgfn import ROOT_DIR

GNINA_PATH = ROOT_DIR / "external" / "gnina"


class GninaRescorer:
    """
    Rescores an ensemble of poses for the same ligand using Gnina.
    https://github.com/gnina
    """

    def __init__(
        self,
        receptor_pdbqt_file: str,
        pose_format: str,
        center_pos: List[int],
        size: List[int],
    ) -> None:
        """
        Parameters:
        - receptor_pdbqt_file: Cleaned receptor PDBQT file to use for docking
        - pose_format: file extension of poses to write to and rescore (e.g. "pdbqt", "sdf")
        - center_pos: 3-dim list containing (x,y,z) coordinates of grid box
        - size: 3-dim list containing sizing information of grid box in (x,y,z) directions
        """

        self.receptor_pdbqt_file = receptor_pdbqt_file
        self.pose_format = pose_format
        self.center_pos = center_pos
        self.size = size

    def __call__(self, poses: List[str]) -> int:
        """
        Rescores as many poses as possible. Discards failures.
        """
        all_pose_scores = []
        for pose in poses:
            try:
                all_pose_scores.append(self._rescore_pose(pose))
            except Exception as e:
                print(
                    f"Failed to rescore pose: {e} with gnina. This pose will be omitted from the final reranking."
                )
                all_pose_scores.append(0)

        sorted_poses = sorted(list(zip(all_pose_scores, poses)), key=lambda x: x[0], reverse=True)
        return sorted_poses[0]

    def _rescore_pose(self, pose: str) -> float:
        """
        Rescores a single pose with gnina.
        """

        if self.pose_format == "pdbqt":
            pose = self._trim_pdbqt(pose)

        with tempfile.NamedTemporaryFile(
            suffix=f".{self.pose_format}", mode="w", delete=False
        ) as temp_file:
            temp_file.write(pose)
            temp_file_path = temp_file.name

        full_cmd = (
            f"./gnina -r {self.receptor_pdbqt_file} "
            f"-l {temp_file_path} "
            f"--center_x {self.center_pos[0]} "
            f"--center_y {self.center_pos[1]} "
            f"--center_z {self.center_pos[2]} "
            f"--size_x {self.size[0]} "
            f"--size_y {self.size[1]} "
            f"--size_z {self.size[2]} "
            "--score_only"
        )

        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            start_new_session=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(GNINA_PATH),
        )
        stdout, _ = proc.communicate()

        # get the CNN score and return it
        pattern = r"CNNscore:\s+([\d.]+)"
        match = re.search(pattern, stdout.decode("UTF-8"))
        return float(match.group(1))

    @staticmethod
    def _trim_pdbqt(pose: str) -> str:
        """
        When dealing with pdbqts, we have to discard the "MODEL" lines at the beginning and end,
        otherwise gnina gets upset.
        """
        lines = pose.strip().split("\n")
        filtered_lines = lines[1:-1]
        filtered_pose = "\n".join(filtered_lines)
        return filtered_pose

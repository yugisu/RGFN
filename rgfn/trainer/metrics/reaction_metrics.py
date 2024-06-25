import json
from pathlib import Path
from typing import Any, Dict, List, Set

import gin
import numpy as np
from numpy import mean
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionStateEarlyTerminal,
    ReactionStateTerminal,
)
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
        self.molecules: Dict[Any, float] = {}
        self.dump_every_n = dump_every_n
        self.iterations = 0
        self.dump_path = Path(run_dir) / "unique_molecules"
        self.dump_path.mkdir(exist_ok=True, parents=True)

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        terminal_states = trajectories.get_last_states_flat()
        proxy = trajectories.get_reward_outputs().proxy
        for state, proxy_value in zip(terminal_states, proxy):
            if isinstance(state, ReactionStateTerminal):
                self.molecules[state.molecule.smiles] = proxy_value.item()

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
class FractionEarlyTerminate(MetricsBase):
    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, float]:
        terminal_states = trajectories.get_last_states_flat()
        num_early_terminate = sum(
            [1 for state in terminal_states if isinstance(state, ReactionStateEarlyTerminal)]
        )
        return {"fraction_early_terminate": num_early_terminate / len(terminal_states)}

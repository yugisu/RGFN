from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rgfn.api.trajectories import Trajectories


class MetricsBase(ABC):
    """
    The base class for metrics used in Trainer.
    """

    @abstractmethod
    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, Any]:
        ...

    def collect_files(self) -> List[Path | str]:
        return []


class MetricsList(MetricsBase):
    def __init__(self, metrics: Sequence[MetricsBase]):
        self.metrics = metrics

    def compute_metrics(self, trajectories: Trajectories) -> Dict[str, Any]:
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.compute_metrics(trajectories))
        return metrics

    def collect_files(self) -> List[Path | str]:
        file_paths = []
        for metric in self.metrics:
            file_paths.extend(metric.collect_files())
        return file_paths

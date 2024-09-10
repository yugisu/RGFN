import abc
from typing import Generic

from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState


class TrajectoryFilterBase(Generic[TState, TActionSpace, TAction], abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> Trajectories[TState, TActionSpace, TAction]:
        raise NotImplementedError


class IdentityTrajectoryFilter(TrajectoryFilterBase[TState, TActionSpace, TAction]):
    def __call__(
        self, trajectories: Trajectories[TState, TActionSpace, TAction]
    ) -> Trajectories[TState, TActionSpace, TAction]:
        return trajectories

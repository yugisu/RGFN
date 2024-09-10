import abc
from typing import Generic

import gin

from rgfn.api.trajectories import Trajectories
from rgfn.api.type_variables import TAction, TActionSpace, TState
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionAction,
    ReactionActionSpace,
    ReactionActionSpace0Invalid,
    ReactionState,
    ReactionState0,
)
from rgfn.trainer.trajectory_filters.trajectory_filter_base import TrajectoryFilterBase


@gin.configurable()
class RGFNTrajectoryFilter(
    TrajectoryFilterBase[ReactionState, ReactionActionSpace, ReactionAction]
):
    def __call__(
        self, trajectories: Trajectories[ReactionState, ReactionActionSpace, ReactionAction]
    ) -> Trajectories[ReactionState, ReactionActionSpace, ReactionAction]:
        source_states = trajectories.get_source_states_flat()
        valid_source_mask = [isinstance(state, ReactionState0) for state in source_states]
        return trajectories.masked_select(valid_source_mask)

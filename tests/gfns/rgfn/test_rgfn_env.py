from gfns.helpers.env_test_helpers import (
    helper__test_env__backward_forward_consistency,
    helper__test_env__forward_backward_consistency,
)

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__rgfn_env__forward_backward_consistency(rgfn_env: ReactionEnv, n_trajectories: int):
    helper__test_env__forward_backward_consistency(rgfn_env, n_trajectories=n_trajectories)


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__rgfn_env__backward_forward_consistency(rgfn_env: ReactionEnv, n_trajectories: int):
    helper__test_env__backward_forward_consistency(
        rgfn_env, n_trajectories=n_trajectories, sample_directly_from_reversed_env=False
    )

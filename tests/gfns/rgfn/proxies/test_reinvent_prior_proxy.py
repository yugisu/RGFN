from gfns.helpers.proxy_test_helpers import (
    helper__test_proxy__is_deterministic,
    helper__test_proxy__returns_sensible_values,
)

from rgfn import ReinventPriorProxy

from ..fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__reinvent_prior_proxy__returns_sensible_values(rgfn_env: ReactionEnv, n_trajectories: int):
    """
    Test whether the ReinventPriorProxy returns sensible values for the sampled trajectories.
    """
    helper__test_proxy__returns_sensible_values(
        env=rgfn_env, proxy=ReinventPriorProxy(), n_trajectories=n_trajectories
    )


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__reinvent_prior_proxy__is_deterministic(rgfn_env: ReactionEnv, n_trajectories: int):
    """
    Test whether the ReinventPriorProxy is deterministic.
    """
    helper__test_proxy__is_deterministic(
        env=rgfn_env, proxy=ReinventPriorProxy(), n_trajectories=10
    )

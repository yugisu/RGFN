# can return sensible log probs for any trajectory
from gfns.helpers.policy_test_helpers import (
    helper__test_forward_policy__returns_sensible_log_probs,
    helper__test_forward_policy__samples_only_allowed_actions,
)

from rgfn.gfns.reaction_gfn.policies.reaction_forward_policy import (
    ReactionForwardPolicy,
)

from .fixtures import *


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__rgfn_forward_policy__samples_only_allowed_actions(
    rgfn_forward_policy: ReactionForwardPolicy, rgfn_env: ReactionEnv, n_trajectories: int
):
    helper__test_forward_policy__samples_only_allowed_actions(
        rgfn_forward_policy, rgfn_env, n_trajectories
    )


@pytest.mark.parametrize("n_trajectories", [1, 10])
def test__rgfn_forward_policy__returns_sensible_log_probs(
    rgfn_forward_policy: ReactionForwardPolicy, rgfn_env: ReactionEnv, n_trajectories: int
):
    helper__test_forward_policy__returns_sensible_log_probs(
        rgfn_forward_policy, rgfn_env, n_trajectories
    )

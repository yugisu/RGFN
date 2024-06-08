import torch

from gflownet import RandomSampler, UniformPolicy
from gflownet.api.env_base import EnvBase, TAction, TState
from gflownet.api.proxy_base import ProxyBase
from gflownet.common.policies.uniform_policy import TIndexedActionSpace
from gflownet.utils.helpers import seed_everything


def helper__test_proxy__returns_sensible_values(
    env: EnvBase[TState, TIndexedActionSpace, TAction],
    proxy: ProxyBase[TState],
    n_trajectories: int,
):
    seed_everything(42)
    sampler = RandomSampler(
        policy=UniformPolicy(),
        env=env,
        reward=None,
    )

    trajectories = next(
        iter(sampler.get_trajectories_iterator(n_total_trajectories=n_trajectories, batch_size=-1))
    )

    states = trajectories.get_non_last_states_flat()
    proxy_output = proxy.compute_proxy_output(states)

    assert torch.isnan(proxy_output.value).sum() == 0
    assert torch.isinf(proxy_output.value).sum() == 0
    if proxy_output.components is not None:
        for component in proxy_output.components.values():
            assert torch.isnan(component).sum() == 0
            assert torch.isinf(component).sum() == 0

    if proxy.is_non_negative:
        assert (proxy_output.value < 0).sum() == 0

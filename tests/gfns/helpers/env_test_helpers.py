from gflownet import RandomSampler, UniformPolicy
from gflownet.api.env_base import EnvBase, TAction, TState
from gflownet.common.policies.uniform_policy import TIndexedActionSpace
from gflownet.utils.helpers import seed_everything


def helper__test_env__forward_backward_consistency(
    env: EnvBase[TState, TIndexedActionSpace, TAction], n_trajectories: int
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
    non_source_states = trajectories.get_non_source_states_flat()
    non_last_states = trajectories.get_non_last_states_flat()
    actions = trajectories.get_actions_flat()
    forward_action_spaces = trajectories.get_forward_action_spaces_flat()
    backward_action_spaces = trajectories.get_backward_action_spaces_flat()
    new_states = env.apply_backward_actions(non_source_states, actions)

    assert non_last_states == new_states
    for action, forward_action_space in zip(actions, forward_action_spaces):
        assert forward_action_space.is_action_allowed(action)

    for action, backward_action_space in zip(actions, backward_action_spaces):
        assert backward_action_space.is_action_allowed(action)


def helper__test_env__backward_forward_consistency(
    env: EnvBase[TState, TIndexedActionSpace, TAction],
    n_trajectories: int,
    sample_from_env: bool = True,
):
    seed_everything(42)

    if sample_from_env:
        sampler = RandomSampler(
            policy=UniformPolicy(),
            env=env.reversed(),
            reward=None,
        )

        trajectories = next(
            iter(
                sampler.get_trajectories_iterator(
                    n_total_trajectories=n_trajectories, batch_size=-1
                )
            )
        )
    else:
        sampler = RandomSampler(
            policy=UniformPolicy(),
            env=env,
            reward=None,
        )

        trajectories = next(
            iter(
                sampler.get_trajectories_iterator(
                    n_total_trajectories=n_trajectories, batch_size=-1
                )
            )
        )
        last_states = trajectories.get_last_states_flat()
        reverse_sampler = RandomSampler(
            policy=UniformPolicy(),
            env=env.reversed(),
            reward=None,
        )
        trajectories = reverse_sampler.sample_trajectories_from_sources(last_states)

    non_source_states = trajectories.get_non_source_states_flat()
    non_last_states = trajectories.get_non_last_states_flat()
    actions = trajectories.get_actions_flat()
    forward_action_spaces = trajectories.get_forward_action_spaces_flat()
    backward_action_spaces = trajectories.get_backward_action_spaces_flat()
    new_states = env.apply_forward_actions(non_last_states, actions)

    assert non_source_states == new_states
    for action, forward_action_space in zip(actions, forward_action_spaces):
        assert forward_action_space.is_action_allowed(action)

    for action, backward_action_space in zip(actions, backward_action_spaces):
        assert backward_action_space.is_action_allowed(action)

import abc
from abc import ABC
from typing import Dict, Generic, Iterator

from rgfn.api.env_base import TAction, TActionSpace, TState
from rgfn.api.sampler_base import SamplerBase
from rgfn.api.trajectories import Trajectories


class ReplayBufferBase(ABC, Generic[TState, TActionSpace, TAction]):
    """
    A base class for replay buffers. A replay buffer stores terminal states or trajectories and can sample them
    in backward direction using the provided sampler.

    Type parameters:
        TState: The type of the states.
        TActionSpace: The type of the action spaces.
        TAction: The type of the actions.
    """

    def __init__(self, sampler: SamplerBase[TState, TActionSpace, TAction]):
        assert sampler.env.is_reversed, "The environment should be reversed."
        self.sampler = sampler

    @abc.abstractmethod
    def add_trajectories(self, trajectories: Trajectories[TState, TActionSpace, TAction]):
        """
        Add trajectories to the replay buffer.

        Args:
            trajectories: trajectories to add.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def get_trajectories_iterator(
        self, n_total_trajectories: int, batch_size: int
    ) -> Iterator[Trajectories]:
        """
        Get an iterator that samples trajectories from the environment. It can be used to sampled trajectories in
            batched manner.
        Args:
            n_total_trajectories: total number of trajectories to sample.
            batch_size: the size of the batch. If -1, the batch size is equal to the number of n_total_trajectories.

        Returns:
            an iterator that samples trajectories.
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict:
        """
        Return the state of the replay buffer as a dictionary.

        Returns:
            a dictionary containing the state of the replay buffer.
        """
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state of the replay buffer from a dictionary.

        Args:
            state_dict: a dictionary containing the state of the replay buffer.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def set_device(self, device: str):
        """
        Set the device on which to perform the computations.

        Args:
            device: a device to set.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def clear_sampling_cache(self):
        """
        Clear the sampling cache of the replay buffer and underlying objects (e.g. samplers with policies). Some
            objects may use caching to speed up the sampling process.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def clear_action_embedding_cache(self):
        """
        Clear the action embedding cache of the replay buffer and underlying objects (e.g. samplers with policies). Some
           policies may embed and cache the actions.

        Returns:
            None
        """
        ...

    def update_using_trajectories(
        self, trajectories: Trajectories[TState, TActionSpace, TAction], update_idx: int
    ) -> Dict[str, float]:
        """
        Update the replay buffer using the trajectories. The replay buffer may use the trajectories to update the action counts.

        Args:
            trajectories: a `Trajectories` object containing the trajectories.
            update_idx: the index of the update. Used to avoid updating the replay buffer multiple times with the same data.
                The sampler may be shared by other objects that can call `update_using_trajectories` in
                `Trainer.update_using_trajectories`.
        Returns:
            A dict containing the metrics.
        """
        return self.sampler.update_using_trajectories(trajectories, update_idx)

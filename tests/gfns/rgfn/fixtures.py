from pathlib import Path

import pytest

from rgfn.gfns.reaction_gfn import ReactionDataFactory, ReactionEnv
from rgfn.gfns.reaction_gfn.policies.action_embeddings import FragmentOneHotEmbedding
from rgfn.gfns.reaction_gfn.policies.reaction_backward_policy import (
    ReactionBackwardPolicy,
)
from rgfn.gfns.reaction_gfn.policies.reaction_forward_policy import (
    ReactionForwardPolicy,
)


@pytest.fixture(scope="module")
def reaction_path() -> Path:
    return Path(__file__).parent / "../../../data/chemistry.xlsx"


@pytest.fixture(scope="module")
def rgfn_data_factory(reaction_path: Path) -> ReactionDataFactory:
    return ReactionDataFactory(
        reaction_path=reaction_path,
        docking=False,
    )


@pytest.fixture(scope="module")
def rgfn_data_factory_docking(
    reaction_path: Path, fragment_path_docking: Path
) -> ReactionDataFactory:
    return ReactionDataFactory(
        reaction_path=reaction_path,
        fragment_path=fragment_path_docking,
        docking=True,
    )


@pytest.fixture(scope="module")
def rgfn_env_docking(rgfn_data_factory_docking: ReactionDataFactory) -> ReactionEnv:
    return ReactionEnv(data_factory=rgfn_data_factory_docking)


@pytest.fixture(scope="module")
def rgfn_env(rgfn_data_factory: ReactionDataFactory) -> ReactionEnv:
    return ReactionEnv(data_factory=rgfn_data_factory, max_num_reactions=4)


@pytest.fixture(scope="module")
def rgfn_one_hot_action_embedding_fn(
    rgfn_data_factory: ReactionDataFactory,
) -> FragmentOneHotEmbedding:
    return FragmentOneHotEmbedding(
        data_factory=rgfn_data_factory,
    )


@pytest.fixture(scope="module")
def rgfn_forward_policy(
    rgfn_data_factory: ReactionDataFactory,
    rgfn_one_hot_action_embedding_fn: FragmentOneHotEmbedding,
) -> ReactionForwardPolicy:
    return ReactionForwardPolicy(
        data_factory=rgfn_data_factory, action_embedding_fn=rgfn_one_hot_action_embedding_fn
    )


@pytest.fixture(scope="module")
def rgfn_backward_policy(
    rgfn_data_factory: ReactionDataFactory,
    rgfn_one_hot_action_embedding_fn: FragmentOneHotEmbedding,
) -> ReactionBackwardPolicy:
    return ReactionBackwardPolicy(
        data_factory=rgfn_data_factory,
    )

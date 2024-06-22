from pathlib import Path

import pytest
from gfns.helpers.training_test_helpers import helper__test_training__runs_properly


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/rgfn_seh_proxy.gin",
    ],
)
def test__rgfn__trains_properly(config_path: str, tmp_path: Path):
    config_override_str = ""
    helper__test_training__runs_properly(config_path, config_override_str, tmp_path)

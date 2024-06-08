import os
from pathlib import Path

import gin

from gflownet import ROOT_DIR
from gflownet.trainer.trainer import Trainer
from gflownet.utils.helpers import seed_everything


def helper__test_training__runs_properly(
    config_path: str, config_override_str: str, tmp_path: Path
):
    os.chdir(ROOT_DIR)
    seed_everything(42)

    # save config override to file
    config_override_path = tmp_path / "config_override.gin"
    with open(config_override_path, "w") as f:
        f.write(config_override_str)

    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [config_path, config_override_path],
        bindings=[f'run_name="test"', "WandbLogger.mode='offline'", f"user_root_dir='{tmp_path}'"],
    )
    trainer = Trainer(n_iterations=1)
    trainer.train()
    trainer.close()

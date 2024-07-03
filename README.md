# RGFN

Code for "RGFN: Synthesizable Molecular Generation Using GFlowNets" [[arXiv]](https://arxiv.org/abs/2406.08506).
Repository based on [https://github.com/gmum/RetroGFN](https://github.com/gmum/RetroGFN).

## Setup

To create the conda environment, run the following commands:

```bash
conda create --name rgfn python=3.11.8 -y
conda activate rgfn

# If using CUDA:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==2.2.1+cu118 -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html

# If using CPU:
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==1.1.2 -f https://data.dgl.ai/wheels/torch-2.3/cpu/repo.html

pip install -e .

pip install pre-commit
pre-commit install
```

You also need to download the data
from [google drive](https://drive.google.com/drive/folders/1h53J3XT_usqnR87JvxJfvF15BOyjp17k): into `data` directory.

For convenience all experiments will be logged to `experiments` directory. If you want to store the experiments results
on other partition, you can create a symlink to the desired location:

```bash
ln -s <your_path> experiments
```

### Setup REINVENT priors

To setup environment for REINVENT priors, run the following commands:

```bash
pip install xxhash==3.4.1 molvs==0.1.1 tensorboard==2.16.2 tomli==2.0.1 pathos==0.3.2 numpy==1.26.4 mmpdb==2.1

# If using CUDA:
pip install torchvision==0.18.1+cu118 --no-deps --index-url https://download.pytorch.org/whl/cu118

# If using CPU:
pip install torchvision==0.18.1 --no-deps --index-url https://download.pytorch.org/whl/cpu
```

To download the code for REINVENT priors, run the following command:

```bash
sh external/setup_reinvent.sh
```

Note that the reinvent repository is heavy (~2GB). If you want to decrease its size after it has been downloaded
remove `external/reinvent/.git` directory (~1.2GB) and unused priors from `external/reinvent/priors` (~0.7GB).

In the case of issues with libxrender library, install it with conda:

```bash
conda install xorg-libxrender
```

or with apt-get:

```bash
sudo apt-get install libxrender1
```

### Setup GNEprop proxies

Note that GNEprop is part of a not-yet published project, and access to the code and the models is restricted. In particular, you will need an access to a private github repository for the instructions below to work.

To setup environment for GNEprop proxies, run the following commands:

```bash
pip install descriptastorus==2.6.1 "ray[tune]==2.31.0" ax-platform==0.4.0 matplotlib==3.9.0 seaborn==0.13.2 umap-learn==0.5.6 pytorch-lightning==1.9.5
pip install lightning-bolts==0.7.0 --no-deps
```

To download the code for GNEprop model, run the following command:

```bash
sh external/setup_gneprop.sh
```

You also need to download the models manually from [google drive](https://drive.google.com/drive/folders/1v1kmXnxDxrMc_UrlP3ug7f6Ekfs5Z8QC) and put them inside `external/gneprop/models`. The code will look for the checkpoints in that location.

Finally, note that GNEprop requires a system-wide installation of CUDA to work on GPU (loaded by, e.g., `module load cuda/11.8`).

## Train

To train the RGFN using sEH proxy, run:

```sh
python train.py --cfg configs/rgfn_seh_proxy.gin
```

The script will dump the results under `experiments/rgfn_seh_proxy/<timestamp>` directory. Our code uses gin-config
package that allows for lightweight models configuration along with dependency injection.

## Project Structure

### API

Under `rgfn.api`, the repository provides a flexible API that clearly separates the GFlowNet components. The states,
actions and action spaces can be represented as an arbitrary classes, which allows for easy implementation of GFlowNets
with non-static environments (e.g. with dynamic action spaces).

- `env_base.py` Base class for environments. It provides a minimal and flexible interface that can be used to implement
  environments with dynamic action spaces. An action space is a set of possible actions that can be taken from a state
  in forward (forward action space) and backward (backward action space) direction. The reward is decoupled from the
  environment, so that environment should only describe the possible transitions between states. The environment can be
  reversed to enable backward sampling of the trajectories.
- `policy_base.py` A base class for policies. Given the current batch of states, a policy samples corresponding actions.
  It also computes the log probabilities when chosen actions and following states are provided.
- `sampler_base.py` A base class for samplers. A sampler samples trajectories from the environment using a policy.
- `trajectories.py`. A trajectory is a sequence of states and actions sampled by a sampler using the environment and the
  policy. Every state has a corresponding forward and backward action space which describe the possible actions that can
  be taken from that state. Trajectories are stored in a batch manner. The terminal states in the trajectories are
  assigned with rewards.
- `reward_base.py`. A class representing the reward function. The reward function is a function of a proxy output that
  takes a batch of states and computes rewards that are used to train the policy.
- `proxy_base.py`. A base class for proxies. A proxy is a function that takes a batch of states and computes values that
  are then used to compute the reward.
- `objective_base.py`. A base class for GFN objectives. An objective is a function that takes a batch of trajectories
  and computes the loss (objective)
- `replay_buffer_base.py`. A base class for replay buffers. A replay buffer stores terminal states or trajectories and
  can sample them
  in backward direction using the provided sampler.

### Shared

Under `rgfn.shared`, the repository provides shared utilities that are used across the different GFlowNets
implementations, e.g. Trajectory Balance Objective, Conditioned Trajectory Balance Objective, uniform policy, cached
proxy base class, random samplers, reward_prioritized buffer, etc.

### GFNs

Under `rgfn.gfns`, the repository provides the implementation of the GFlowNets.

## Citation

```text
@article{koziarski2024rgfn,
  title={RGFN: Synthesizable Molecular Generation Using GFlowNets},
  author={Koziarski, Micha{\l} and Rekesh, Andrei and Shevchuk, Dmytro and van der Sloot, Almer and Gai{\'n}ski, Piotr and Bengio, Yoshua and Liu, Cheng-Hao and Tyers, Mike and Batey, Robert A},
  journal={arXiv preprint arXiv:2406.08506},
  year={2024}
}
```

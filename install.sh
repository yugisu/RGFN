conda create -n retro_gfn python=3.11 -y
conda activate retro_gfn

# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install  dgl -f https://data.dgl.ai/wheels/cpu/repo.html

# GPU
#conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
#pip install dgl==1.0.2+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

# common
pip install torchtyping tqdm pytest wandb gin-config matplotlib rdkit pandas torch-geometric more-itertools dgllife torchmetrics

# syntheseus
pip install omegaconf pydantic==1.10.5

# development
pip install pre-commit rdchiral notebook
pre-commit install

# Self-Regulating Prompt Expansion for Continual Learning (SCOPE)
This is the official implementation of DEDUCE in the paper in Pytorch.
## Datasets
- CIFAR-100
- CUB-200
- ImageNet-R

**NOTE:** Datasets are automatically downloaded in data/.

- This can be changed by changing the base_path function in utils/conf.py or using the --base_path argument.
- The data/ folder should not be tracked by git and is created automatically if missing.
## Installation
- To execute the codes for running experiments, run the following.
```bash
pip install -r requirements.txt
```
- New models can be added to the models/ folder.
- New datasets can be added to the datasets/ folder.
## Checkpoints
Create a folder pretrained/
- Sup-21K
- Sup-1K
- iBOT-1k
- DINO-1k
## Examples
### Run a model
Run the following commands under the project root directory. The scripts are set up for 1 GPUs.
sh experiments/cifar-100.sh
sh experiments/imagenet-r_all.sh
sh experiments/cub-200.sh
## Acknowledgements
Our implementation is based on https://github.com/aimagelab/mammoth

# Self-Regulating Prompt Expansion for Continual Learning (SCOPE)
## Datasets
- CIFAR-100
- CUB-200
- ImageNet-R

**NOTE:** Datasets are automatically downloaded in data/.
- This can be changed by changing the base_path function in utils/conf.py or using the --base_path argument.
- The data/ folder should not be tracked by git and is created automatically if missing.
## Installation
- python=3.8.18
- torch=2.0.0+cu118
- torchvision=0.15.1+cu118
- timm=0.9.12
- scikit-learn=1.3.2
- numpy
- pyaml
- pillow
- opencv-python
- pandas
- openpyxl (write results to a xlsx file)
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

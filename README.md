# CLIP-goes-3D

The official code release of CLIP goes 3D:Leveraging Prompt Tuning for Language Grounded 3D Recognition

![image](PATH)

This repository includes the pre-trained models, evaluation and training codes for pre-training, zero-shot, and fine-tuning experiments. It is built on the [Point-BERT](https://github.com/lulutang0608/Point-BERT) codebase. Please see the end of this document for a full list of code references.

## Environment set-up

The known working environment configuration is 

```
python 3.9
pytorch 1.12
CUDA 11.6
```

 
1. Install the conda virtual environment using the provided .yml file.
```
conda env create -f environment.yml 
```
(OR)

1. Install dependencies manually.
  ``` 
  pip install ftfy tqdm h5py geoopt einops open3d pyyaml regex tensorboardX termcolor yacs

  ```
  ```
  conda install -c anaconda scikit-image scikit-learn scipy
  ```

  ```
  pip install git+https://github.com/openai/CLIP.git
  ```

  Install PointNet ops from [here](https://github.com/erikwijmans/Pointnet2_PyTorch)

  ```
  pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
  ```
  ```
  cd ./extensions/chamfer_dist
  python setup.py develop
```

2. Build modified timm from scratch

  ```
  cd ./models/SLIP/pytorch-image-models
  pip install -e .
  ```


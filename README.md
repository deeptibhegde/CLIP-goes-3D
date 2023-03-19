# CLIP-goes-3D

Official code for the paper "CLIP goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition"

![image](docs/teaser.png)

This repository includes the pre-trained models, evaluation and training codes for pre-training, zero-shot, and fine-tuning experiments of CG3D. It is built on the [Point-BERT](https://github.com/lulutang0608/Point-BERT) codebase. Please see the end of this document for a full list of code references.

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
3. Install [PointNet ops](https://github.com/erikwijmans/Pointnet2_PyTorch)

   
   ```
   cd third_party/Pointnet2_PyTorch
   pip install -e .
   ```
   
4.  Install [PyGeM](https://mathlab.github.io/PyGeM/)
   
   ```
   cd third_party/PyGeM
   python setup.py install
   ```

## Dataset set-up

1. Download point cloud datasets for pre-training and fine-tuning.

  - Download [ShapeNetCore v2](https://shapenet.org/).
  - Download [ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
  - Download [ScanObjectNN](https://hkust-vgd.github.io/scanobjectnn/)

    Save and unzip the above datasets.
  
 2. Render views of textured CAD models of ShapeNet using [this](https://github.com/nv-tlabs/GET3D/blob/master/render_shapenet_data/README.md) repository such that the data is organized as 

  ```
  ├── data (this may be wherever you choose)
  │   ├── modelnet40_normal_resampled
  │   │   │── modelnet10/40_shape_names.txt
  │   │   │── modelnet10/40_train/test.txt 
  │   │   │── airplane
  │   │   │── ...
  │   │   │── laptop 
  │   ├── ShapeNet55
  │   │   │── train.txt
  │   │   │── test.txt
  │   │   │── shapenet_pc
  │   │   │   |── 03211117-62ac1e4559205e24f9702e673573a443.npy
  │   │   │   |── ...
  │   ├── shapenet_render
  │   │   │── train_img.txt
  │   │   │── val_img.txt
  │   │   │── shape_names.txt
  │   │   │── taxonomy.json
  │   │   │── camera
  │   │   │── img
  │   │   │   |── 02691156
  │   │   │   |── ...
  │   ├── ScanObjectNN
  │   │   │── main_split
  │   │   │── ...

  ```
## Pre-training
- Change data paths to relevant locations in ```cfgs/dataset_configs/```

- Pre-train PointTransformer on ShapeNet under the CG3D framework:

    ```
    python main_BERT.py  --exp_name {NAME FOR EXPT} --config cfgs/PRETRAIN_models/PointTransformerVPT.yaml  --pretrain    --out_dir {OUTPUT DIR PATH}  --text --image --clip --VL SLIP --visual_prompting --npoints 8192

    ```
    
- Pre-train PointMLP on ShapeNet under the CG3D framework:

   ```
   python main_BERT.py  --exp_name {NAME FOR EXPT} --config cfgs/PRETRAIN_models/PointMLP_VPT.yaml  --pretrain    --out_dir {OUTPUT DIR PATH}  --text --image --clip --VL SLIP --visual_prompting --npoints 1024

    ```





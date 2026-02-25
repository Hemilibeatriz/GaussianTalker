# GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with Audio-Driven 3D Gaussian Splatting (ACM MM 2024)
<a href="https://arxiv.org/abs/2404.16012v2"><img src="https://img.shields.io/badge/arXiv-2404.16012v2-%23B31B1B"></a>
<a href="https://ku-cvlab.github.io/GaussianTalker"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<br>

This is our official implementation of the paper 

"GaussianTalker: Real-Time High-Fidelity Talking Head Synthesis with Audio-Driven 3D Gaussian Splatting"

by [Kyusun Cho](https://github.com/kyustorm7)\*, [Joungbin Lee](https://github.com/joungbinlee)\*, [Heeji Yoon](https://github.com/yoon-heez)\*, [Yeobin Hong](https://github.com/yeobinhong), [Jaehoon Ko](https://github.com/mlnyang), Sangjun Ahn, [Seungryong Kim](https://cvlab.korea.ac.kr)<sup>&dagger;</sup>

## ⚡️News
**❗️2024.06.13:** We also generated the torso in the same space as the face using Gaussian splatting. **After cloning the torso branch**, you can train and render it in the same way to use it.


## Introduction
![image](./docs/structure.png)
<!-- <br> -->

For more information, please check out our [Paper](https://arxiv.org/abs/2404.16012v2) and our [Project page](https://ku-cvlab.github.io/GaussianTalker/).

## Installation
We implemented & tested **GaussianTalker** with NVIDIA RTX 3090 and A6000 GPU.

Run the below codes for the environment setting. ( details are in requirements.txt )

Updated by Hemilibeatriz

```bash
git clone https://github.com/Hemilibeatriz/GaussianTalker.git
cd GaussianTalker
git submodule update --init --recursive
conda create -n GaussianTalker python=3.8 -y
conda activate GaussianTalker

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    export CC=/usr/bin/gcc-9
    export CXX=/usr/bin/g++-9
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST="8.9"
    export CUB_HOME=$HOME/cub
nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
    unset CC
    unset CXX
    unset CUDA_HOME
    unset TORCH_CUDA_ARCH_LIST
    unset CUB_HOME

conda deactivate
conda activate GaussianTalker

which nvcc
nvcc --version
echo $CUDA_HOME

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e submodules/custom-bg-depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install pytorch3d==0.7.6
pip install tensorflow==2.12.0
pip install --upgrade "protobuf<=3.20.1"
```


## Download Dataset

We used talking portrait videos from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF), [GeneFace](https://github.com/yerfor/GeneFace) and [HDTF dataset](https://github.com/MRzzm/HDTF). 
These are static videos whose average length are about 3~5 minutes.

You can see an example video with the below line:

```
wget https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4?raw=true -O data/obama/obama.mp4
```

We also used [SynObama](https://grail.cs.washington.edu/projects/AudioToObama/) for cross-driven setting inference.


## Data Preparation

- prepare face-parsing model.

```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
```

- Download 3DMM model from [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) 

Put "01_MorphableModel.mat" to data_utils/face_tracking/3DMM/ 
    
```bash
cd data_utils/face_tracking
python convert_BFM.py
cd ../../
python data_utils/process.py ${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4 
```

- Obtain AU45 for eyes blinking
  
Run `FeatureExtraction` in [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), rename and move the output CSV file to `(your dataset dir)/(dataset name)/au.csv`.


```
├── (your dataset dir)
│   | (dataset name)
│       ├── gt_imgs
│           ├── 0.jpg
│           ├── 1.jgp
│           ├── 2.jgp
│           ├── ...
│       ├── ori_imgs
│           ├── 0.jpg
│           ├── 0.lms
│           ├── 1.jgp
│           ├── 1.lms
│           ├── ...
│       ├── parsing
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           ├── 3.png
│           ├── ...
│       ├── torso_imgs
│           ├── 0.png
│           ├── 1.png
│           ├── 2.png
│           ├── 3.png
│           ├── ...
│       ├── au.csv
│       ├── aud_ds.npy
│       ├── aud_novel.wav
│       ├── aud_train.wav
│       ├── aud.wav
│       ├── bc.jpg
│       ├── (dataset name).mp4
│       ├── track_params.pt
│       ├── transforms_train.json
│       ├── transforms_val.json
```

## Training


```bash
python train.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} --model_path ${YOUR_MODEL_DIR} --configs arguments/64_dim_1_transformer.py 
```


## Rendering

Please adjust the batch size to match your GPU settings.

```bash
python render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} --model_path ${YOUR_MODEL_DIR} --configs arguments/64_dim_1_transformer.py --iteration 10000 --batch 128
```
    
## Inference with custom audio

Please locate the files <custom_aud>.wav and <custom_aud>.npy in the following directory path: ${YOUR_DATASET_DIR}/${DATASET_NAME}.

```bash
python render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} --model_path ${YOUR_MODEL_DIR} --configs arguments/64_dim_1_transformer.py --iteration 10000 --batch 128 --custom_aud <custom_aud>.npy --custom_wav <custom_aud>.wav --skip_train --skip_test
```

## Citation
If you find our work useful in your research, please cite our work as:
```
@inproceedings{cho2024gaussiantalker,
  title={Gaussiantalker: Real-time talking head synthesis with 3d gaussian splatting},
  author={Cho, Kyusun and Lee, Joungbin and Yoon, Heeji and Hong, Yeobin and Ko, Jaehoon and Ahn, Sangjun and Kim, Seungryong},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={10985--10994},
  year={2024}
}
```

This is the official code for [Learning RoI Transformer for Detecting Oriented Objects in Aerial Images](https://arxiv.org/abs/1812.00155)

This code is based on deformable convolution network

mmdetection version is on the way

## Requirements: Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet).

2. Python 2.7. We recommend using Anaconda2 as it already includes many common packages. We do not support Python 3 yet, if you want to use Python 3 you need to modify the code to make it work.

3. Python packages might missing: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running
	```
	pip install -r requirements.txt
	```
4. For Windows users, Visual Studio 2015 is needed to compile cython module.

## Installation

1. Clone the RoI Transformer repository, and we'll call the directory that you cloned RoI Transformer as ${RoI_ROOT}

```
git clone git@github.com:dingjiansw101/RoITransformer_DOTA.git
```

2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

	**Note: The MXNet's Custom Op cannot execute parallelly using multi-gpus after this [PR](https://github.com/apache/incubator-mxnet/pull/6928). We strongly suggest the user rollback to version [MXNet@(commit 998378a)](https://github.com/dmlc/mxnet/tree/998378a) for training (following Section 3.2 - 3.5).**

	***Build from source (Since there are custom c++ operators, We need to complie the MXNet from source.)***

	3.1 Clone MXNet and checkout to [MXNet@(commit 998378a)](https://github.com/dmlc/mxnet/tree/998378a) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout 998378a
	git submodule update
	# if it's the first time to checkout, just use: git submodule update --init --recursive
	```
	3.2 Copy the c++ operators to MXNet source
	```
	cp fpn/operator_cxx/* mxnet/src/operator/contrib
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	3.4 Install the MXNet Python binding by

	***Note: If you will actively switch between different versions of MXNet, please follow 3.5 instead of 3.4***
	```
	cd python
	sudo python setup.py install
	```
	3.5 For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)`, and modify `MXNET_VERSION` in `./experiments/rfcn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.

4. complie dota_kit

    ```
    sudo apt-get install swig
    cd ${RoI_ROOT}/dota_kit
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
    ```

## Prepare DOTA Data:

1.Prepare script
   put your original dota data (before split) in path_to_data
   make sure it looks like
   ```
   path_to_data/train/images,
   path_to_data/train/labelTxt,
   path_to_data/val/images,
   path_to_data/val/labelTxt,
   path_to_data/test/images

   cd ${RoI_ROOT}/prepare_data
   python prepare_data.py --data_path path_to_data --num_process 32
   ```
2.Create soft link

   ```
   cd ${RoI_ROOT}
   mkdir data
   cd data
   ln -s path_to_data dota_1024
   ```

## Pretrained Models

We provide trained convnet models.

1. To use the demo with our pre-trained RoI Transformer models for DOTA, please download manually from [Google Drive](https://drive.google.com/drive/folders/1kUBsH2v5DK6QjqDoMmyx16bW7gUlEgn1?usp=sharing), or [BaiduYun](https://pan.baidu.com/s/14KBADK41S5hOO8NQVQlbWA) (Extraction code: fucc)
 and put it under the following folder.
    Make sure it look like this:
    ```
        ./output/rcnn/DOTA/resnet_v1_101_dota_RoITransformer_trainval_rcnn_end2end/train/rcnn_dota-0040.params
        ./output/fpn/DOTA/resnet_v1_101_dota_rotbox_light_head_RoITransformer_trainval_fpn_end2end/train/fpn_DOTA_oriented-0008.params
    ```
## Training & Testing

```
cd ${RoI_ROOT}
```

1.training


```
  sh train_dota_light_RoITransformer.sh
```

2.testing

```
  sh test_dota_light_RoITransformer.sh
```


---------------------------------------------------

© Microsoft, 2017. Licensed under an MIT license.


If you find RoI Transformer and DOTA data useful in your research, please consider citing:
```
@inproceedings{ding2019learning,
  title={Learning RoI Transformer for Oriented Object Detection in Aerial Images},
  author={Ding, Jian and Xue, Nan and Long, Yang and Xia, Gui-Song and Lu, Qikai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2849--2858},
  year={2019}
}
@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}
```


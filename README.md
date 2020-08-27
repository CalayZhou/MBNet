# MBNet
Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems (ECCV 2020)
- paper download: https://arxiv.org/pdf/2008.03043.pdf
- the introduction PPT: https://github.com/CalayZhou/MBNet/blob/master/MBNet-3128.pdf
- video demo: https://www.bilibili.com/video/BV1Hi4y137aS


# Usage
## 1. Dependencies
This code is tested on [Ubuntu18.04, tensorflow1.14, keras2.1.6, python3.6,cuda10.0,cudnn7.6]. 
 
 
 >make sure the GPU enviroment is the same as above (cuda10.0,cudnn7.6), otherwise you may have to compile the `nms` and `utils` according to https://github.com/endernewton/tf-faster-rcnn. Besides, check the keras version is keras2.1, i find there may be some mistakes if the keras version is higher. To be as simple as possible, I recommend installing the dependencies with Anaconda as follows:
 
 ```
1. conda create -n python36 python=3.6
2. conda activate python36
3. conda install cudatoolkit=10.0
4. conda install cudnn=7.6
5. conda install tensorflow-gpu=1.14
6. conda install keras=2.1
7. conda install opencv
8. python demo.py
```

## 2. Prerequisites
- train data (download to ./data/kaist)\
[Baidu Cloud](https://pan.baidu.com/s/1gunujXdb2TPBeibp6fsUQg)(extract code: `ABCD`) or [KAIST website](https://soonminhwang.github.io/rgbt-ped-detection/)
- test data (download to ./data/kaist_test)\
[Baidu Cloud](https://pan.baidu.com/s/1xQMEnHkmV29_Jq1pk_ERVw)(extract code: `ABCD`) or [Google Drive](https://drive.google.com/file/d/1XNSF4GhYNc4J6WrhrLTlYck6ddbW5m8b/view?usp=sharing)
- ResNet50 pretrained model (download to ./data/models/)\
[Baidu Cloud](https://pan.baidu.com/s/1f9gy1u_TL6SMo2UNwKFQ7w)(extract code: `ABCD`) or [Google Drive](https://drive.google.com/file/d/1RPdCCRjuyP13tREDv4LJAi7my8rS3UxW/view?usp=sharing)
- MBNet model(download to ./data/models/)\
[Baidu Cloud](https://pan.baidu.com/s/11HOz3LM8dkZxOkwEo9wWlQ)(extract code: `ABCD`) or [Google Drive](https://drive.google.com/file/d/1WP6MoOfztkzUtVQf_ScwKhUh4LE6O7Kx/view?usp=sharing)

## 3. Demo example
### 3.1 Demo images

> 1. Check the [MBNet model](https://pan.baidu.com/s/11HOz3LM8dkZxOkwEo9wWlQ) is available at ./data/models/resnet_e7_l224.hdf5
> 2. Run the script: `python demo.py`
> 3. The detection result is saved at ./data/kaist_demo/.
### 3.2 Demo video

> 1. Check the [MBNet model](https://pan.baidu.com/s/11HOz3LM8dkZxOkwEo9wWlQ) is available at ./data/models/resnet_e7_l224.hdf5
> 2. Set weight_path , test_file , lwir_test_file in `demo_video.py`
> 3. Run the script: `python demo_video.py`
> 4. The detection result videos saved at MBNet directory.

## 4. Evaluate model performance
> 1. check the [MBNet model](https://pan.baidu.com/s/11HOz3LM8dkZxOkwEo9wWlQ) is available at ./data/models/resnet_e7_l224.hdf5 and the test data is available at ./data/kaist_test.
> 2. Run the script: `python test.py`
> 3. The test results are saved at ./data/result/.
> 3. open the [KAISTdevkit-matlab-wrapper](https://github.com/CalayZhou/MBNet/tree/master/KAISTdevkit-matlab-wrapper) and run the `demo_test.m`.

## 5. Train your own model
> 1. Check the [ResNet50 pretrained model](https://pan.baidu.com/s/1f9gy1u_TL6SMo2UNwKFQ7w) is available at ./data/models/double_resnet.hdf5 and the train data is available at ./data/kaist.
> 2. Run the script: `python train.py`
> 3. The trained models are saved at ./output.
> 4. Evaluate model performance as above.

## 6. Comparison with other Methods
Please download the Matlab implemented comparison code [[Baidu Cloud](https://pan.baidu.com/s/1ogNMx0vGcrdn9dLSRRsk6Q)(extract code: `ABCD`) or [Google Drive](https://drive.google.com/file/d/1h0-VwZrnJH8zBVvk5r5bWt3ekq4KLe45/view?usp=sharing)] and run the script according to the README.txt.


## 7. Acknowledgements
Thanks to Liu Wei, this pipeline is largely built on his ALFNet code available at: https://github.com/liuwei16/ALFNet.




# Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{MBNet-ECCV2020,
    author = {Kailai Zhou and Linsen Chen and Xun Cao},
    title = {Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems},
    booktitle = ECCV,
    year = {2020}
}
```


# MBNet
Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems (ECCV 2020)
- paper download: https://arxiv.org/pdf/2008.03043.pdf
- the introduction PPT: https://github.com/CalayZhou/MBNet/blob/master/MBNet-3128.pdf
- video demo: https://www.bilibili.com/video/BV1Hi4y137aS


# Usage
## 1. Dependencies
This code is tested on [Ubuntu18.04, tensorflow1.14, numpy1.16, Python3.6,cuda10.0,cudnn7.6]. 
 
 
 >make sure the GPU enviroment is the same as above (cuda10.0,cudnn7.6), otherwise you may have to compile the `nms` and `utils` according to https://github.com/endernewton/tf-faster-rcnn.
 
## 2. Prerequisites
- train data\
[Baidu Cloud]() & [KAIST website]()
- test data\
[Baidu Cloud]() & [Google Drive]()
- ResNet50 pretrained model\
[Baidu Cloud]() & [Google Drive]()
- MBNet model\
[Baidu Cloud]() & [Google Drive]()

## 3. demo example

> 1. Check the [MBNet model]() is available at ./data/models/resnet_e7_l224.hdf5
> 2. Run the script: `python demo.py`
> 3. The detection result is saved at ./data/kaist_demo/.

## 4. Evaluate model performance
> 1. check the [MBNet model](https://www.jianshu.com/p/191d1e21f7ed) is available at ./data/models/resnet_e7_l224.hdf5 and the test data is available at ./data/kaist_test.
> 2. Run the script: `python test.py`
> 3. The test results are saved at ./data/result/.
> 3. open the [KAISTdevkit-matlab-wrapper](https://www.jianshu.com/p/191d1e21f7ed) and run the `demo_test.m`.

## 5. Train your own model
> 1. Check the [ResNet50 pretrained model](https://www.jianshu.com/p/191d1e21f7ed) is available at ./data/models/double_resnet.hdf5 and the train data is available at ./data/kaist.
> 2. Run the script: `python train.py`
> 3. The traine models are saved at ./output.
> 4. Evaluate model performance as above.

## 6. Comparison with other Methods
Please download the comparison code ([[Matlab implemented](https://soonminhwang.github.io/rgbt-ped-detection/misc/CVPR15_Pedestrian_Benchmark.pdf)]) and run the script according to the README.txt.


## 7. Acknowledgements
Thanks to Liu Wei, this pipeline is largely built on his ALFNet code available at: https://github.com/liuwei16/ALFNet




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






## PREDATOR: Registration of 3D Point Clouds with Low Overlap (CVPR 2021)
This repository provides implementation using sparse convolution backbone. It represents the official implementation of the paper:

### [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)

\*[Shengyu Huang](https://shengyuh.github.io), \*[Zan Gojcic](https://zgojcic.github.io/), [Mikhail Usvyatsov](https://aelphy.github.io), [Andreas Wieser](https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html), [Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)\
|[ETH Zurich](https://igp.ethz.ch/) | \* Equal contribution

For more information, please see the [project website](https://overlappredator.github.io)

![Predator_teaser](assets/teaser_predator.jpg?raw=true)



### Contact
If you have any questions, please let us know: 
- Shengyu Huang {shengyu.huang@geod.baug.ethz.ch}
- Zan Gojcic {zan.gojcic@geod.baug.ethz.ch}

### News
- 2021-03-12: pre-trained model release
- 2021-02-28: codebase release


### Instructions
This code has been tested on 
- Python 3.8.5, PyTorch 1.7.1, CUDA 11.2, gcc 9.3.0, GeForce RTX 3090/GeForce GTX 1080Ti

#### Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/ShengyuH/OverlapPredator.Mink.git
virtualenv predator; source predator/bin/activate
cd OverlapPredator.Mink; pip install -r requirements.txt
```
in your working folder.
If you come across problem when installing ```MinkowskiEngine```, please have a look [here](https://github.com/NVIDIA/MinkowskiEngine)



### Train on 3DMatch(Indoor)
After creating the virtual environment and downloading the datasets, Predator can be trained using:
```shell
python main.py configs/train/indoor.yaml
```

### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{huang2020predator,
  title={PREDATOR: Registration of 3D Point Clouds with Low Overlap},
  author={Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, Konrad Schindler},
  journal={CVPR},
  year={2021}
}
```

### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [FCGF](https://github.com/chrischoy/FCGF) (KITTI preprocessing)
- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch) (KPConv backbone)
- [3DSmoothNet](https://github.com/zgojcic/3DSmoothNet) (3DMatch preparation)
- [MultiviewReg](https://github.com/zgojcic/3D_multiview_reg) (3DMatch benchmark)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) (Transformer part)
- [DGCNN](https://github.com/WangYueFt/dgcnn) (self-gnn)
- [RPMNet](https://github.com/yewzijian/RPMNet) (ModelNet preprocessing and evaluation)

 We thank the respective authors for open sourcing their methods. We would also like to thank Reviewer 2 for valuable inputs.
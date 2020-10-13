# NAS-SCAM
This is the code assoiciated with MICCAI 2020 paper titled "NAS-SCAM: Neural Architecture Search-Based Spatial and Channel Joint Attention Module for Nuclei Semantic Segmentation and Classification"
In the code, we presents attention module architecture searching for nuclei semantic segmentation and classification. Which has two features: 1) the architecture of attention module can be automated search by NAS, 2) multiple attention modules in the same network can be searched out different architectures.
The dataset is from the challenge of [MoNuSAC 2020](https://biomedicalimaging.org/2020/wp-content/uploads/static-html-to-wp/data/dff0d41695bbae509355435cd32ecf5d/index-26.htm). We make the preprocessed dataset available in [Baidu Netdisk](). Simply changing the dir in train_functions.py can make the program properly run. We make the following preprocessing: 1) color normalization, 2) [overlap cropping](https://github.com/ZuhaoLiu/overlap-crop-and-recover-image). A detailed description of our dataset will be provided soon.

# Requirements
- pytorch
- numpy
- optparse
- pillow
- nibabel
- tqdm
- scipy

# Citation
If you think this code can help you, please kindly cite our paper.
```
@inproceedings{liu2020scam,
  title={NAS-SCAM: Neural Architecture Search-Based Spatial and Channel Joint Attention Module for Nuclei Semantic Segmentation and Classification},
  author={Liu, Zuhao and Wang, Huan and Zhang, Shaoting and Wang, Guotai and Qi, Jin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={263--272},
  year={2020},
  organization={Springer}
}
```

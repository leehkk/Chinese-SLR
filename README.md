# Chinese-SLR
The datasets will be released soon

# Installation

First, build the codebase by running:
```
git clone https://github.com/leehkk/Chinese-SLR
cd Chinese-SLR
```
Then, create a conda virtual environment and activate it:
```
conda env create -f environment.yaml
source activate Swin
```
# Usage

## Training the RGB Model

Training the default Swin can be done using the following command:

```
python run.py
```
You may need to change location of your dataset in the run.py

## Inference


# Acknowledgements

Our work is built on top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [SL-GCN]([https://github.com/rwightman/pytorch-image-models](https://github.com/jackyjsy/CVPR21Chal-SLR)). We thank the authors for releasing their code. If you use our model, please consider citing these works as well:

```BibTeX
@inproceedings{jiang2021skeleton,
  title={Skeleton Aware Multi-modal Sign Language Recognition},
  author={Jiang, Songyao and Sun, Bin and Wang, Lichen and Bai, Yue and Li, Kunpeng and Fu, Yun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2021}
}
```

```BibTeX
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

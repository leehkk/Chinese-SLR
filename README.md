# Chinese-SLR
The datasets will be released soon

# Installation
First, create a conda virtual environment and activate it:
```
conda create --name Swin python=3.8
source activate Swin
```
Next, you need to make sure that you have a CUDA version of PyTorch installed
Then, install the following packages:

- tensorboard: `pip install tensorboard`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- fvcore: `pip install 'git+https://github.com/facebookresearch/fvcore'`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- Numpy: `pip install numpy`

Lastly, build the codebase by running:
```
git clone https://github.com/leehkk/Chinese-SLR
cd Chinese-SLR
```
# Usage

## Training the RGB Model

Training the SPL Swin can be done using the following command:

```
python Swin/run_spl.py
```
You may need to change location of your dataset in the run.py

## Inference

We have provided some demo videos for use, and the full dataset will be released soon

Inference can be done using the following command:
```
python Swin/inference.py
```

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

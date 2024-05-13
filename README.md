# RoboDreamer

This is the official repo for the paper:

> **[RoboDreamer: Learning Compositional World Models for Robot Imagination](https://robovideo.github.io/)**  \
> Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, Chuang Gan \
> [website](https://robovideo.github.io/) | [arxiv](https://arxiv.org/abs/2404.12377)

## Installation


```bash
conda create -n rtx python==3.9
conda activate rtx
pip install -r requirement.txt
```

## Dataset

This repo contains example dataset in `datasets/` 

You can download dataset from [open-x](https://robotics-transformer-x.github.io/)

## Training

```bash
torchrun train_rtx.py
```

You can set your own hyperparameters by using `config.py`

## Citation

```bib
@article{zhou2024robodreamer,
  title={RoboDreamer: Learning Compositional World Models for Robot Imagination},
  author={Zhou, Siyuan and Du, Yilun and Chen, Jiaben and Li, Yandong and Yeung, Dit-Yan and Gan, Chuang},
  journal={arXiv preprint arXiv:2404.12377},
  year={2024}
}
```

## Acknowledgements

This codebase is modified from the following repositories:
[imagen-pytorch](https://github.com/lucidrains/imagen-pytorch) and
[AVDC](https://github.com/flow-diffusion/AVDC)
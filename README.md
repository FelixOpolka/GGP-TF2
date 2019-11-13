# GGP-TF2

This repository contains the implementation of the the Graph Gaussian Process model introduced in '[Bayesian semi-supervised learning with graph Gaussian processes](https://arxiv.org/abs/1809.04379)'.

```
@inproceedings{ng2018gaussian,
  title={Bayesian semi-supervised learning with graph Gaussian processes},
  author={Ng, Yin Cheng and Colombo, Nicolo and Silva, Ricardo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

It is an updated version of the semi-supervised classification model of the original implementation accompanying the paper: [yincheng/GGP](https://github.com/yincheng/GGP). It was updated to use GPflow 1.9 and TensorFlow 2.0, which allowed to greatly simplify the implementation.

The resulting models achieve the accuracies reported in the paper.

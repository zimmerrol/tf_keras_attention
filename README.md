# 2D (Visual) Attention for TensorFlow and Keras

The blog post [*Attention in Neural Networks and How to Use It*](http://akosiorek.github.io/ml/2017/10/14/visual-attention.html#mjx-eqn-att) by Adam Kosiorek shows a easy way to use Gaussian Attention on 2D data (e.g. images). However, this code can not directly be used for arbitrary batch sizes and numbers of channels in the input data. The changes required to use the described method on this kind of data are included in this repository.

The repository contains the pure *TensorFlow* based implementation and a *Keras* *Layer* which is wrapped around the code. Furthermore, a small example notebook is added. 
# ZoomTrack

Authors: [Yutong KOU](https://Kou-99.github.io), [Jin Gao](https://people.ucas.edu.cn/~jgao?language=en), [Bing Li](http://www.ia.cas.cn/sourcedb_ia_cas/cn/iaexpert/201707/t20170715_4833365.html), Gang Wang, [Weiming Hu](https://people.ucas.ac.cn/~huweiming?language=en), Yizheng Wang and Liang Li

This repo is the official implementation of ***ZoomTrack: Target-aware Non-uniform Resizing for Efficient Visual Tracking*** 

[[arXiv]](https://arxiv.org/abs/2310.10071)

# Introduction
Recently, the transformer has enabled the speed-oriented trackers to approach state-of-the-art (SOTA) performance with high-speed thanks to the smaller input size or the lighter feature extraction backbone, though they still substantially lag behind their corresponding performance-oriented versions. In this paper, we demonstrate that it is possible to narrow or even close this gap while achieving high tracking speed based on the smaller input size. To this end, we non-uniformly resize the cropped image to have a smaller input size while the resolution of the area where the target is more likely to appear is higher and vice versa. This enables us to solve the dilemma of attending to a larger visual field while retaining more raw information for the target despite a smaller input size. Our formulation for the non-uniform resizing can be efficiently solved through quadratic programming (QP) and naturally integrated into most of the crop-based local trackers. Comprehensive experiments on five challenging datasets based on two kinds of transformer trackers, i.e., OSTrack and TransT, demonstrate consistent improvements over them. In particular, applying our method to the speed-oriented version of OSTrack even outperforms its performance-oriented counterpart by 0.6% AUC on TNL2K, while running 50% faster and saving over 55% MACs.

Code and model weights will be available soon.

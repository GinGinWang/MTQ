# Multivariate Triangular Quantiles for Novelty Detection
Pytorch implementation to replicate experiments in the  NIPS2019 paper, Multivariate Triangular Quantiles for Novelty Detection (published soon).


# Datasets
* Thyroid: http://odds.cs.stonybrook.edu/thyroid-disease-dataset/
* KDDCUP: http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled_10_percent.gz
* MNIST/Fashion MNIST: directly download for you by torchvision

# Models
This implementation includes the following models for novelty detection. 
* LSA: Autoencoder part in [2] 
* LSA_EN[2]: Autoencoder + Density Estimator in [2]
* LSA_SOS: Autoencoder + TQM based on SOS-flow
* LSA_MAF: Autoencoder + TQM based on MAF-flow
* EN:  density estimator in [2]
* SOS: TQM based on SOS-Flow [3]
* MAF: TQM based on  MAF-Flow [1]

The LSA_SOS and LSA_MAF are two instances for our TQM. In our paper, we mainly apply LSA_SOS as the novelty detection model.
# Environments
This code runs on Python >=3.6.
Set up enviroment by:
```
pip install -r requirements.txt
```
# Demo
In demo.sh, we give some examples to show the how to train/test a model on one dataset

# Reference

Our implementation based on the following reference/code:
* Autoencoder(https://github.com/aimagelab/novelty-detection): Pytorch implementation to replicate experiments in the CVPR19 paper "Latent Space Autoregression for Novelty Detection"


* MAF-flow(https://github.com/ikostrikov/pytorch-flows):  PyTorch implementations of [Masked Autoregressive Flow](https://arxiv.org/abs/1705.07057)

* SOS-flow: PyTorch implementations of [Sum-of-Square Polynomial Flow](https://arxiv.org/abs/1905.02325) 


[1] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for density estimation[C]//Advances in Neural Information Processing Systems. 2017: 2338-2347.

[2] Abati D, Porrello A, Calderara S, et al. Latent Space Autoregression for Novelty Detection[C]//International Conference on Computer Vision and Pattern Recognition. 2019. 

[3] Jaini, P., Selby, K. A., & Yu, Y. (2019). Sum-of-Squares Polynomial Flow. arXiv preprint arXiv:1905.02325.





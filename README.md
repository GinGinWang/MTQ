# Multivariate Triangular Quantiles for Novelty Detection
Pytorch to replicate experiments in the  NIPS2019 paper Multivariate Triangular Quantiles for Novelty Detection (published soon).

Our implementation based on the following reference/code:
* Autoencoder(https://github.com/aimagelab/novelty-detection): Pytorch code to replicate experiments in the CVPR19 paper "Latent Space Autoregression for Novelty Detection"


* MAF-flow(https://github.com/ikostrikov/pytorch-flows):  PyTorch implementations of [Masked Autoregressive Flow](https://arxiv.org/abs/1705.07057)

* SOS-flow: PyTorch implementations of [Sum-of-Square Polynomial Flow](https://arxiv.org/abs/1905.02325) 

### Datasets
* Thyroid: http://odds.cs.stonybrook.edu/thyroid-disease-dataset/
* KDDCUP: http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled_10_percent.gz
* MNIST 
* Fashion MNIST

### *Model*

#### *LSA*
* Only use the Autoencoder part in [2] 
* loss = reconstruction loss

#### *LSA_EN*
* same as [2]: 
Autoencoder + Density Estimator (estimated network)

#### *LSA_SOS*
* Autoencoder + TQM based on SOS-flow

#### *LSA_MAF*
* Autoencoder + TQM based on MAF-flow

#### *EN*
* Only use density estimator EN
* loss = negative llk

#### *SOS*: 
* Only use density estimator SOS
* loss = negative llk

#### *MAF*: 
* Only use density estimator MAF
* loss = negative llk


[1] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for density estimation[C]//Advances in Neural Information Processing Systems. 2017: 2338-2347.

[2] Abati D, Porrello A, Calderara S, et al. Latent Space Autoregression for Novelty Detection[C]//International Conference on Computer Vision and Pattern Recognition. 2019. 





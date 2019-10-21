# Multivariate Triangular Quantiles for Novelty Detection
To be present in NIPS 2019 by Jingjing Wang

## Introduction
This is the official implementation for ``Multivariate Triangular Quantiles for Novelty Detection``.



### Content

* **transform_sos.py** -  SoS flow.
* **transfrom_maf.py** - MAF flow.

* **LSA_mnist.py** - LSA autoencoder with/without density estimator for mnist.
* **LSA_cifar.py** - LSA autoencoder with/without density estimator for cifar10.

* **train.py** - code for training novelty dector.
* **test.py** - code for testing/run novelty dector.


### Datasets
* Thyroid: http://odds.cs.stonybrook.edu/thyroid-disease-dataset/
* KDDCUP: 
* 
### *Results*
** Results names denote the model name and parameters.

* "{model_name}_{dataset}_cd{cd}_ptr{pretrained}_fix{fixed}_nml{score_normed}_nlr{novel_ratio}_b{num_blocks}_h{hidden_size}_lam{lam}.txt"
* cd: use reconstruction loss as one dimension of latent vector
* pretrained: intialize the autoencoder with pretrained model
* fix: fix autoencoder when training the density estimator
* nml: normalize the novelty score by minimum and maximum in validate set
* novel_ratio: the ratio of novel points in test set, 1 means we use the whole test set.
* num_blocks: Blocks number of SOS-flow or MAF-flow
* hidden_size: the number of hidden neurons in conditioner network of SOS and MAF
* lam: trade off between the reconstruction loss and negative llk in objective function

### *Model*
An example to build and train model in train_mnist.sh. Args:
* --NoAuencoder: only use estimator
* --Combine_density: use reconstruction loss as one dimension of latent vector

#### *LSA*
* Only use the Autoencoder part in [2] 
* loss = reconstruction loss

#### *LSA_EN*
* same as [2]: 
Autoencoder + Density Estimator (estimated network)
* loss = reconstruction loss+ negative llk

#### *LSA_SOS*
* replace density estimator in *LSA_EN* with SoS flow
* loss = reconstruction loss+ negative llk

#### *LSA_MAF*
* replace density estimator in *LSA_EN* with MAF flow
* loss = reconstruction loss+ negative llk

#### *EN*
* Only use density estimator EN
* loss = negative llk

#### *SOS*: 
* Only use density estimator SOS
* loss = negative llk

#### *MAF*: 
* Only use density estimator MAF
* loss = negative llk


### Metric 
* Metric: 'Class', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS', 'Precision',
                'F1',
                'Recall',
                'Threshold'

First, we only consider AUROC-based metric: 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS'



[1] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for density estimation[C]//Advances in Neural Information Processing Systems. 2017: 2338-2347.

[2] Abati D, Porrello A, Calderara S, et al. Latent Space Autoregression for Novelty Detection[C]//International Conference on Computer Vision and Pattern Recognition. 2019. 





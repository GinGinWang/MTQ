# Novelty Detection Using SoSflow density estimator


### Content

* **transform_sos.py** -  SoS flow.
* **transfrom_maf.py** - MAF flow.
* **LSA_mnist.py** - LSA autoencoder with/without density estimator.

* **train.py** - code for training novelty dector.
* **test.py** - code for testing/run novelty dector.

### Model
## *LSA*
* Only use autoencoder  of [2] 
* loss = reconstruction loss

## *LSA_EN*: same as [2], Autoencoder + Density Estimator (estimated network), loss = reconstruction loss+ negative llk

## *LSA_SOS*: replace density estimator in *LSA_EN* with SoS flow, loss = reconstruction loss+ negative llk

## *LSA_MAF*: replace density estimator in *LSA_EN* with MAF flow, loss = reconstruction loss+ negative llk

## *EN*: Only use density estimator EN, loss = negative llk

## *SOS*: Only use density estimator SOS, loss = negative llk

## *MAF*: Only use density estimator MAF, loss = negative llk


### Metric 
Metric: 'Class', 'AUROC-LLK', 'AUROC-REC', 'AUROC-NS', 'Precision',
                'F1',
                'Recall',
                'Threshold'


[1] Papamakarios G, Pavlakou T, Murray I. Masked autoregressive flow for density estimation[C]//Advances in Neural Information Processing Systems. 2017: 2338-2347.

[2] Abati D, Porrello A, Calderara S, et al. Latent Space Autoregression for Novelty Detection[C]//International Conference on Computer Vision and Pattern Recognition. 2019. 





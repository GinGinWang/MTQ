# Novel Detection Using SoSflow density estimator


### Content

* **sosflow** - code for sosflow.
* **train.py** - code for training novelty dector.
* **test.py** - code for testing/run novelty dector.

### How to run
## For sosflows

## For SoSLSA

You will need to run **train.py** first.

For **train_AAE.py**, you need to call *main* function:

    train_AAE.main(
      folding_id,
      inliner_classes,
      total_classes,
      folds=5
    )
  
   Args:
   -  folding_id: Id of the fold. For MNIST, 5 folds are generated, so folding_id must be in range [0..5]
   -  inliner_classes: List of classes considered inliers.
   -  total_classes: Total count of classes.
   -  folds: Number of folds.
   
After autoencoder was trained, from **test.py**, you need to call *main* function:

    novelty_detector.main(
      folding_id,
      inliner_classes,
      total_classes,
      folds=5
    )
  
   Set of arguments is the same.

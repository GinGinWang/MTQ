# Novel Detection Using SoSflow density estimator


### Content

* **sosflow** - code for sosflow.
* **train.py** - code for training novelty dector.
* **test.py** - code for testing/run novelty dector.

### How to run

## For SoSLSA

You will need to run **train.py** first.

  python train.py mninst
   
After autoencoder was trained, from **test.py**, you need to call *main* function:  
   Set of arguments is the same.

   python test.py mnist

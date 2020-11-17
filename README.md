## Overview

### Version 1:
Gaussian mixture model (GMM) fitting with Expectation Maximization (EM). 
This is implemented manually with lots of comments to help link the hairy math to a concrete
implementation.

### Version 2:
GMM fitted using Variational Inference. Again, this is implemented manually with lots of comments
to help show how the scary math of variational inference maps to actual code.

### Version 3:
GMM fitted using the Edward library using XXX inference. 
This is a simpler implementation that leverages a more robust library for inference over 
probabilistic models. I added a bunch of comments here too to show how this implementation links
to version 2.

## Setup

Use Python 3.8.6, required dependencies are listed in the requirements.txt.
Note: I was unable to install pytorch with Python 3.9, so please use 3.8. 



### References
* https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html
* https://github.com/angusturner/generative_models/blob/master/examples/GMM.ipynb
* https://towardsdatascience.com/variational-inference-in-bayesian-multivariate-gaussian-mixture-model-41c8cc4d82d7
* http://edwardlib.org/tutorials/unsupervised
* http://edwardlib.org/api/ed/VariationalInference


### TODO
1. implement variational inference
1. implement using [Edward](http://edwardlib.org/) or [Tensorflow](https://blog.tensorflow.org/2018/12/an-introduction-to-probabilistic.html)?
1. create a notebook to provide more background material / equations etc.

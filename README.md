# ADMMSVM
### Alternating directions method of multipliers with applications in distributed support vector problems


------------

## What is this?
It is the implementation made for my master dissertation that was defensed in University of Campinas (UNICAMP), Brazil.
This dissertation lies in the area of mathematical optimization, specifically convex optimization, and it investigate
the use of the *Alternating Directions Method of Multipliers (ADMM)* to make the *Support Vector Machine (SVM)*
applicable over distributed classification problem, in which the training data was distributed and can not be shared
between the node.

## Description
The dissertation can be found [here](https://github.com/caiodadauto/Distributed-SVM/blob/master/dissertation/dissertation.pdf)(Portuguese version)
and the presentation is [here](https://github.com/caiodadauto/Distributed-SVM/blob/master/dissertation/presentation.pdf)(Portuguese version);
Although the dissertation there is in Portuguese, all references is on English, specially, take a look to this
[reference](https://dl.acm.org/citation.cfm?id=1859906), almost whole dissertation is based in this one.

## Usage
To run this implementation, it is require the following packages,
* `netwokx==1.8`
* `numpy>=1.13`
* `scikit-learn==0.18`
* `matplotlib==2.0.1`
* `seaborn==0.7.1`
* `mpi4py==2.0.0`

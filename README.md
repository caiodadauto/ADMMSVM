# ADMMSVM
--

ADMMSVM is the implementation made for my master dissertation, University of Campinas (UNICAMP), Brazil, with the title **Alternating directions
method of multipliers with applications in distributed support vector problems**.
This dissertation lies in the area of mathematical optimization, specifically convex optimization, and it investigates
the use of the *Alternating Directions Method of Multipliers (ADMM)* to make the *Support Vector Machine (SVM)*
applicable over distributed classification problem, in which the training data are distributed and can not be shared
between the node.

## Description
--
The dissertation can be found [here](https://github.com/caiodadauto/Distributed-SVM/blob/master/dissertation/dissertation.pdf) ( in Portuguese)
and the presentation is [here](https://github.com/caiodadauto/Distributed-SVM/blob/master/dissertation/presentation.pdf) (in Portuguese).
Although the dissertation is in Portuguese, this [reference](https://dl.acm.org/citation.cfm?id=1859906) is helpful to understanding this
work, since my dissertation is based on it.

## Usage
--
To run this implementation, it is require the following packages,
* `netwokx==1.8`
* `numpy>=1.13`
* `scikit-learn==0.18`
* `matplotlib==2.0.1`
* `seaborn==0.7.1`
* `mpi4py==2.0.0`

## Numerical Evaluation
--
### Simple Case
Centralized SVM             |   Distributed ADMMSVM
:-------------------------:|:-------------------------:
![]()  |  ![]()--

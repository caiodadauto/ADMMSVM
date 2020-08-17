# ADMMSVM

ADMMSVM is the implementation made for my master dissertation, University of Campinas (UNICAMP), Brazil, with the title **Alternating directions
method of multipliers with applications in distributed support vector problems**.
This dissertation lies in the area of mathematical optimization, specifically convex optimization, and it investigates
the use of the *Alternating Directions Method of Multipliers (ADMM)* to make the *Support Vector Machine (SVM)*
applicable over distributed classification problem, in which the training data are distributed and can not be shared
between the node.

## Description

The dissertation can be found [here](https://github.com/caiodadauto/Distributed-SVM/blob/master/dissertation/dissertation.pdf) ( in Portuguese)
and the presentation is [here](https://github.com/caiodadauto/Distributed-SVM/blob/master/dissertation/presentation.pdf) (in Portuguese).
Although the dissertation is in Portuguese, this [reference](https://dl.acm.org/citation.cfm?id=1859906) is helpful to understanding this
work, since my dissertation is based on it.

## Usage

To run this implementation, it is require the following packages,
* `netwokx==1.8`
* `numpy>=1.13`
* `scikit-learn==0.18`
* `matplotlib==2.0.1`
* `seaborn==0.7.1`
* `mpi4py==2.0.0`

## Numerical Evaluation

All centralized SVM results are achieved using `sklearn`.

### Simple Linear Case

The graph below shows the linear approximation for a random simple dataset generated using `sklearn`.

![](https://github.com/caiodadauto/ADMMSVM/blob/master/img/simple.png)

### Simple Non-Linear Case

The graphs below show the non-linear approximation using both centralized and distributed SVM applied to a dataset modeled as a chess table. Note that the
non-linear approximation is not the same in these two scenarios.

Centralized SVM             |   Distributed ADMMSVM
:-------------------------:|:-------------------------:
![](https://github.com/caiodadauto/ADMMSVM/blob/master/img/central_non_linear_classifier.png)  |  ![](https://github.com/caiodadauto/ADMMSVM/blob/master/img/dist_non_linear_classifier_0.png)--

Furthermore, the following graph illustrates a possible wrong SVM estimation if only a part of the data are accessible to the model.

![](https://github.com/caiodadauto/ADMMSVM/blob/master/img/local_non_linear_classifier.png)

### Real Dataset (Non-Linear Case)

The graph below shows the accuracy over ADMMSVM iterations for different hyper-parameters compared with centralized SVM results.
This numerical evaluation uses the [Pima Indians diabetes database](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

![](https://github.com/caiodadauto/ADMMSVM/blob/master/img/risk_plot_0.png)

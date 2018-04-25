# Hierarchicaly Method



## Basic Idea

* A physical based model $f_p(x)$

  * Rigid-body-dynamics (RBD): This is a physics-based model derived from rigid-body-dynamics. The RBD result is taken from [1], p. 24.

    > [1] Rasmussen, C.E., Williams, C.K.I.: Gaussian Processes for Machine Learning. MITPress, Cambridge (2006)					



* A ML model $f_m(x)$ with prediction confidence based on  conformal prediction

  * See [2]

    > [2] A Tutorial on Conformal Prediction  Journal of Machine Learning Research 9 (2008) 371-421 Submitted 8/07; Published 3/08

  * $f_m(x)$  aims to learn the residual  $y-f_p(x)$, and for each prediction the  it has a prediction confidence function $A(x_i, Z)$ 

  * The final output

  	$\hat y=\min( \max(\mbox{UB} f_p(x)+f_m(x)\times A(x,Z)),\mbox{UB})$

* The flexibiltiy is to define a  good $A(x_i, Z)$

  * We need to cluseter the data set  to several clusters, and assign xi to cluster.
    * For the decision tree.  We do not cluster, because it has already sperarete the data.
    * **Called local-weighted conformal predict/conditional comform predict?**
  * .$A(x_i, Z)$ can be the distance to the average or the distance to the average/ all other nodes distance to the avearge of the cluster  











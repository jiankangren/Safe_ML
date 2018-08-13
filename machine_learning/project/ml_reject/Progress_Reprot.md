### ML with Reject Option



#### Reject Option

These works implicitly assume that 

​	1. The modelled probability $P(Y|x)$ is trustworthy



- P. Bartlett and M. Wegkamp. Classification with a reject option using a hinge loss. Journal of Machine Learning Research

- R. Herbei and M. Wegkamp. Classification with reject option. Canadian Journal of Statistics, 4(4): 709–721, 2006. 

- Grandvalet, Y., Rakotomamonjy, A., Keshet, J., & Canu, S. (2009). Support vector machines with a reject option. In *Advances in neural information processing systems* (pp. 537-544).

  > Basic idea: train to approximate P(Y=1|x) since the optimal threshold for p+ p_ is known once the cost for FP FN, FalseReject, Postive Reject is known.
  >
  > For logistic  regression, such technique can  directly be used by log-like function. But log-like function does not lead to sparse solutions.  
  >
  > Thus, they use a double hinge loss function such that only P(Y|x) around uncertain value, it generete cost. All high confidence prediction generate 0 cost.

> Reject if $|f(x)|\leq $  threshold $\tau$.
>
> Risk: $d\times P(reject)+P(wrong )$
>
> The problem is that if we have reject cost d, how to minimize the risk.
>
> Cost of making a wrong decision is 1 and that of utilizing the reject option is d > 0
>
> $η(x) = P(Y = 1|X = x)$
>
> How well does $\hat η(X)$ estimate $η(X)$? Based on  liear combinations of base functions

* M. Wegkamp. Lasso type classifiers with a reject option. Electronic Journal of Statistics, 1, 155-168 (2007).

#### Model Uncertainty

Yarin Gal,  Zoubin Ghahramani, Dropout as a Bayesian approximation: representing model uncertainty in deep learning

> The dropout information is used derive variance of prediction as a gaussian process.

 



- **Q2**:  Based on Q1, we can list the possible limitations of these methods. For example, there are some obvious problems

  - The confidence based on decision boundary does not consider the space where are is no data
  - Cannot handle regression problems.

  To address the above problems, a potential strategy is cluster the empty space and space with dense data. Or we can start with the classification problem.  The key is to sampling the empty space.

- **Q3**:  How to evaluate the proposed mechanism?   We need to come up with a metric to compare with other  methods, and find an application scenario.





###Potential Benchmark

####Pumadyn datasets 

This is a family of datasets synthetically generated from a realistic simulation of the dynamics of a Unimation Puma 560 robot arm. There are eight datastets in this family . In this repository we only have two of them. They are all variations on the same model; a realistic simulation of the dynamics of a Puma 560 robot arm. The task in these datasets is to predict the angular accelaration of one of the robot arm's links. The inputs include angular positions, velocities and torques of the robot arm. The family has been specifically generated for the delve environment and so the individual datasets span the corners of a cube whose dimensions represent:

- Number of inputs (8 or 32).
- degree of non-linearity (fairly linear or non-linear)
- amount of noise in the output (moderate or high).

- Source: [DELVE repository](http://www.cs.toronto.edu/~delve/) of data.
- Characteristics: Both data sets contain 8192 (4500+3692) cases. For *puma32H* we have 32 continuous attributes, while for *puma8NH* the cases are described by 8 continuous variables.
- Download : [puma32H.tgz](http://www.dcc.fc.up.pt/~ltorgo/Regression/puma32H.tgz) (1178678 bytes); [puma8NH.tgz](http://www.dcc.fc.up.pt/~ltorgo/Regression/puma8NH.tgz) (307293 bytes)



#### SARCOS



####Kinematics

This is data set is concerned with the forward kinematics of an 8 link robot arm. Among the existing variants of this data set we have used the variant *8nm*, which is known to be highly non-linear and medium noisy.





#### Delta Elevators

 

This data set is also obtained from the task of controlling the elevators of a F16 aircraft, although the target variable and attributes are different from the *elevators* domain. The target variable here is a variation instead of an absolute value, and there was some pre-selection of the attributes.



####Delta Ailerons

This data set is also obtained from the task of controlling the ailerons of a F16 aircraft, although the target variable and attributes are different from the *ailerons* domain. The target variable here is a variation instead of an absolute value, and there was some pre-selection of the attributes.

 

 

### Classification

http://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems





**detection_of_IoT_botnet_attacks_N_BaIoT Data Set**  

http://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT



**BLE RSSI Dataset for Indoor localization and Navigation Data Set**  

http://archive.ics.uci.edu/ml/datasets/BLE+RSSI+Dataset+for+Indoor+localization+and+Navigation





### Progress

* Step 1: Finish the pure python implementation

* Step 2:  C implementation

  
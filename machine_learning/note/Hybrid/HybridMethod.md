## Combine Analytical Model and Machine Learning Model



#### Paper ID1 



>  Reference
>
> *  Enhancing Performance Prediction Robustness by Combining Analytical Modeling and Machine Learning



**Basic Idea**:

* Analytical Model
* A classification algorithm  to learn the regions  where the analytical model is not accurate
* Samples $(y^i,x^i)$ are classified good or bad based on the error of the analytical model.
* Bad samples are used to train the ML model
* Given x, first a classifer would determine whether the x would have high accuracy if we use the analytical model. 
* If it is classified into bad case, then we can use the ML model
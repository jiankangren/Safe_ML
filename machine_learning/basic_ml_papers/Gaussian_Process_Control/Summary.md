## Learning and Control using Gaussian Processes
#### GP For Dynamic System
1. Input : $$x_t=[y_{t-l}:y_{t-1}, \underbrace{u_{t-m}:u_t}_{control~ input},\underbrace{w_{t-p}:w_t}_{exogenous~disturbance~input}]$$
2. Model:  $$y=Pr(Y|X,\theta)$$
3.  Parameter $$\theta$$ is for mean function $$m(x)$$ and corvariance function kernel $$k(x,x')$$
#### Train Data- Optimal Experiment Desgin
1. data that maximizes variance-which can be directly computed
2. data   that maximizes information gain- need to be approximated
#### Prediction
1. consider output result N future steps
2.  choose y that minimize a cost function (but the no intuition)
####  Update Model
1. choose new comming data with max info gain.
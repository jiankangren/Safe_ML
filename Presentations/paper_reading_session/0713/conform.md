##Conformal Inference 



### Problem 

Consider i.i.d. regression data  $Z_1..Z_n \sim P$, where $Z_i=(X_i=(X_i(1),X_i(2),\ldots,X_i(d)),Y_i)$, 

Regression function
$$
\mu(x)=\mathbb E(Y|x)
$$
We want a prediction band $C(X_i)$ that 
$$
P(Y_{i}\in C(X_i))\geq 1-\alpha
$$






### Naive Method

Suppose $X_1...X_n$ be i.i.d. samples of a scalar random variable, then  given another iid sample $X_{n+1}$, 
$$
P(X_{n+1}\leq \hat q_{1-\alpha})\geq 1-\alpha\\
 \hat q_{1-\alpha}=\begin{cases}X_{\lceil(n+1)(1-\alpha) \rceil}~\mbox{ if} (n+1)(1-\alpha)\leq n \\\infty~\mbox{ otherwise }  \end{cases}
$$


Example,  1,2,3,4,5,6,7,8,9,10, than, $P(X_{11}\leq 10)\geq 1-0.1=0.9$

Following the idea described above, we can form the prediction interval defined by 
$$
C(X_{n+1})=[\hat \mu(X_{n+1})-\hat F_n^{-1}(1-\alpha),\hat \mu(X_{n+1})+\hat F_n^{-1}(1-\alpha)]
$$




* $\hat F_n$: empirical distribution of the fitted residuals $|Y_i -\hat \mu(X_i)|$

* $\hat F_n^{-1}(1-\alpha)$ is  the $(1-\alpha)$- quantile  of $\hat F_n$ 

  ![](/Users/sunshine8641/GitHub/Safe_ML_Project/Presentations/paper_reading_session/0713/conformal.png)

* Property: deliver proper **finite-sample coverage** without any assumptions on distribution P





### Conformal Prediction Sets 



In the naive method,  the fitted residual distribution can often be biased downwards, and the fitted function does not consider the new point. 

* For there $X_{n+1}$, there are a set of trial values $\mathcal Y_{trial}$
* For each  $y\in \mathcal Y_{trial}$,  fit the function $\hat \mu_y=\mathcal A(Z_1,...Z_n,(X_{n+1},y))$
* Compute residual $R_{y,i}=|\hat \mu_y(X_i)-Y_i|$ for i=1...n+1
* Compute quantile $\pi(y)=\frac{\sum_{i=1}^{n+1}\mathbb I (R_{y,i}\leq R_{y,n+1})}{n+1}$
* $C_{conf}=\{y\in  \mathcal Y_{trial}:  \pi(y)(n+1)\leq \lceil (n+1)(1-\alpha)\rceil \}$



###Split Conformal Prediction Sets

The original conformal prediction method studied in the last subsection is **computationally intensive.** 



1. Randomly split {1, . . . , n} into two equal-sized subsets $\mathbb I_1,\mathbb  I_2 $ 
2.   fit the function $\hat \mu_y=\mathcal A(\mathbb I_1)$
3. Compute residual $R_i=|Y_i-\mu_y(X_i)|$ for $i\in \mathbb I_2$
4. d=the kth smallest value in $\{R_i :i\in \mathbb I_2\}$,   where $k=⌈(n/2+1)(1−α)⌉ $
5. $C_{split}=[\mu_y(x)-d,\mu_y(x)+d]$


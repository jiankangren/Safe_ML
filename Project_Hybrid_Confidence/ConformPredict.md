## Conformal Inference

### Concept and Notation



- **Confidence** :  the probability that the correct value is within  predictions  before we acquire the data.
- **Credit**


- $\Gamma^{\varepsilon}$ : such that the correct label y is within the region with probability at least $1-\varepsilon$

  - Regression:  $\Gamma^p$ is a region of interval around the prediction $\hat y$
  - Classification:  $\Gamma^p$ is a set of (ideal case just one)  values   

- $y^n \in \Gamma^{\varepsilon}((x^1,y^1),…(x^{n-1},y^{n-1}),x^n)$

- For $\varepsilon_1\leq \varepsilon_2\Rightarrow \Gamma^{\varepsilon_2} \subseteq \Gamma^{\varepsilon_1}$

- **Bag**:  $B=\rmoustache a_1,a_2..a_N  \lmoustache$   a collections of elements in which repetition is allowed

- **Non-Conformity Measure**    $A(B,z)$  how different an example z if from the examples in bag B.  

  - E.g.,   $A(B,z)=d(\hat z(B),z)$  and function $\hat z(B)$ is an abstract B.
    - $A(B,z)=|\bar z_B-z|=|\frac{n\bar z_B+z}{n+1}-z|$ , i.e., average value of  numbers in B
    - note that, the importance is $\hat z(B)$ and the choice of d function does not matter.

  ​

### Applicable  Condition

- **Two Application Case**:
  - Obaserve $z_1...z_{n-1}$ and predict $z_n$
  - Obaserve $(x_1,y_1)….(x_{n-1},y_{n-1}),x_n$  predict $y_n$


- The claim of 95% confidence for a 95% conformal prediction region is valid under **exchangeability**, no matter what the probability distribution the examples follow and no matter what non-conformity measure is used to construct the conformal prediction region.

- The efficiency of  conformal prediction will depend on the probability distribution and the nonconformity measure.

- **Exchangeability**:  for $z_1,z_2,…z_N$ , the N! different orderings are equally like (a litter weaker than they are drawn independant from a distribution)

  - Exchangeability implies **same distribution**
  - Exchangeability does not require **indepdence**.   For a two variables case, Exchangeability only requires$P(z_1=H\& z_2=T)=P(z_1=T\& z_2=H)$.     

- > Definition of **Exchangeability**: 
  >
  > For a bag of size N, and for any examples $a_1,…,a_N$,  is equal to the probability that successive random drawings from the bag B without replacement produces first $a_N$, then $a_{N−1}$, and so on, until the last element remaining in the bag is $a_1$.
  >
  > - 如果对于任何数据集合，如果我们随机打乱它，那么肯定还是符合条件的。 可是现实中就是可能开始遇到的数据是集中在某一块，后来的数据集中在另一块。我们无法保证。





- **Property**:
  - Prediction region is independant of the  **Non-Conformity Measure**, but its effectiveness depends on it.
  - The $1-\epsilon $ predictor  is correct for $1-\epsilon $ time if the **Exchangeability** holds.



### Steps:

#### Determine **Non-Conformity Measure**    $A(B,z)$

​				$A(B,z): = d(\hat z(B),z)$  

and the choice of d is not important,  e.g.,    $d(z,z')=|z-z'|$, or  $(z-z')^2$

 what important is the choice of **point predictor**  $\hat z(B)$. 

For example:

- $\hat z(B)=\bar z_B$ where $\bar z_B$ is the average value of points in bag B,  and the value can include or not include the new value.

- **Nearest Neighbour Classification**.  :    $A(B,z)=\frac{\mbox{min distance of points with the same label}}{\mbox{min distance of the sample with different label}}$.   Here the label the guessed label you can choose,

- **Regression** :  given $(x_1,y_1)….x_n$, $A(B,z)= |y-\hat y|$  where $\hat y$ is the prediction from the regression model

  ###  

#### Algorithm For Non Feature Prediction

```python
Set measure A, level e, and data set z_1..z_n

Provisionally set z_n=value 
for i in 1-n:
	alpha_i= A(B-z_i,z_i)
p_z:=number of i such with a_i>=a_n/ n #可以理解为，样本里,比z_n更不可靠的比例, 越大越好啊
if p_z>e：  #可以理解为，样本里比z_n更不可靠的比例 超过了设定， 那么  
	include  value in valid prediction region # 可以看到pz越大，这个confidence越可靠。 事实上，p_z>e是最起码条件，如果不符合 肯定不成立，比如p=0.02,e=0.01， 虽然成立，但也说明了，z是个在边缘徘徊，相对已经很奇特的样本了。
	

```

**Understanding**:   if we want to accpet the value for $z_n$ with confidence $1-\epsilon$, then $\epsilon$ means that among the n samples,  at least $\epsilon N$ samples should have $\alpha_i\geq a_n$.  Since $p_z$ denotes the fraction of such samples mad if it greater than  $\epsilon$, then we can accpet $z_n=value$. 



**Note** there migth be muliple values that satisfy the condition.   

#### Algorithm For Feature Prediction

```python
Provisionally set z_n=(x_n,y_n=y'')
for i in 1-n:
	alpha_i= A(B-z_i,z_i)
p_y= #{i such with a_i>=a_n}/ n
if p_y>e:
	include y'' in the prediction region
```



#### Trick:

**Credibility** :   i.e., the p-value: 表达了x符合的比例，越高越好

- it is better to report the p value for a known to be false label  e.g., $\empty$,.  If this value is large, it means the prediction is less creditable



#### Regression

The general steps are the same.  

-  For example of nearest neighbour.  Suppose $a_n=|y-1.55|$, and $\epsilon=0.04$, then there must be at least $k=\epsilon N$ samples have  $a_i>a_n$.  Thus the kth largetest $a_i>|y-1.55|$ and we have provide a bound for $y$
- For linear gression of least squares,  then we can compute $a_i=f_i(y_n)$ and bound $y_n$





### Previous Method: Fiasher's Predication Interval

For a give $x$,   if $\hat y$ conforms to a normal distribution.  Suppose we have tried predict $x$ for n-1 times, then once we compute the stardard  variance, 

$s^2_{n-1}=\frac{1}{n-2}\sum_{i=1}^{n-1} (\hat y_i-\bar y)^2$，  then $\frac{y_n-\bar y}{s_{n-1}}\sqrt{\frac{n-1}{n}}$, conforms to a t-distribution with n-2 degress of freedom. Thus, for a certain error bound $\varepsilon$,  we have 

​							$y_n=\bar y_{n-1} +/-  t^{\varepsilon }_{n-2}s_{n-1}\sqrt{\frac{n-1}{n}}$

and this range will contain $y_n$ with probability $1-\varepsilon$ regardless of mean and variance.




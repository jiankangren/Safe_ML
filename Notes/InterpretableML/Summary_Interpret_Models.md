## Basic Interpretable Models	

###Sparse Linear Model



###Discretization Models (E.g, DT)

#### RuleFit

> Reference
>
> * Friedman, Jerome H, and Bogdan E Popescu. 2008. “Predictive Learning via Rule Ensembles.” *The Annals of Applied Statistics*. JSTOR, 916–54.
> * Fokkema, Marjolein, and Benjamin Christoffersen. 2017. *Pre: Prediction Rule Ensembles*. 

Wouldn’t it be convenient to have a model that is as simple and interpretable as linear models, but that also integrates feature interactions? RuleFit fills this gap. RuleFit fits a sparse linear model with the original features and also a set of new features which are decision rules. These new features capture interactions between the original features. RuleFit generates these features automatically from decision trees.



**Notation**:

* input variable $x_j\in S_j$ ,and $s_{jm}\subseteq S_j$ is a sub set of values

* Base learner

  ​			$r_m(x)=\prod_{j=1}^n I(x_j\in s_{jm})\in \{0,1\}$  and $s_{jm}\neq S_j$

  ​			$s_{jm}=(t_{jm},u_{jm}]$

  ​			$r_m(x)=\begin{cases}I(3<age\leq13)\\  I(sex=male)\\ I(3< sarary\leq 4000)\\\end{cases}$

*  Use decision tree to generate rules. The rule of a terminal node is the product of each split condition

  * For the $m_{th}$ tree, there are $t_m$ terminal nodes, then the total numder of rules $K=\sum_{m=1}^M 2(t_m-1)$  <font color=red>why </font>
  * $F(x)=\hat a_0+\sum_{m=1}^K \hat a_k r_k(x)$
  * $\{\hat a_k\}_0^K=\arg \min_{a_k}\sum_{i=1}^N L(y^i,F(x^i))+\lambda\sum_{k=1}^K |a_k|$

* $F(x)=\hat a_0+\sum_{m=1}^K \hat a_k r_k(x)+\sum_{j=1}^nb_j l_j(x_j)$



> Advantage:
>
> * Even if there are many rules in the model, they do not apply to each instance, so for one instance only a handful of rules are important (non-zero weights). This improves local interpretability.
>
> Disadvantage: 
>
> * Could generate a lot of rules and hence has low interpretability.
> * The rules maybe overlap or incorherent.
> * An anecdotal drawback: The papers claim good performance of RuleFit - often close to the predictive performance of Random Forests! - yet in the few cases where I personally tried it, the performance was disappointing.

###  Prototype/Case Based Method (E.g., KNN)







## Ensemble Models	

#### Hierarchical Ensembl

> Reference

- Nusser,S.,Otte,C.,Hauptmann,W.:Interpretableensemblesoflocalmodelsforsafety-related applications. In: Proceedings of 16th European Symposium on Artificial NeuralNetworks (ESANN 2008), Brugge, Belgium, pp. 301–306 (2008)

- [13]  Nusser,S.,Otte,C.,Hauptmann,W.,Kruse,R.:Learningverifiableensemblesforclas-sification problems with high safety requirements. In: Wang, L.S.L., Hong, T.P. (eds.)Intelligent Soft Computation and Evolving Data Mining: Integrating Advanced Tech-nology, pp. 405–431. IGI Global (2009)

  ​

Given training set   $$\mathcal X=\{x^1,x^2…x^M\}$$ of N diemention ,i .e.,  $$x^k=(x_1^k,x_2^k…x_N^k)$$ and output $$y^k\in \{0,1\}$$.  We project the N dimension to an aribitray subspace $$\pi_{\beta} (\mathcal X)$$, where $$\beta\subset \{1,2,3..N\}$$.   

Submodel    is defined   and input dimension is limited to two for visulization i.e., $$|\beta_j|\leq 2$$

​			$$g_j:  g_j\left(\pi_{\beta_j}(x^k)\right)=\hat y^k_j$$

​	and the outout is based on votes of the nodes in the tree

​		$$f(x^k)=\frac{1}{|models|} \sum_{i\in models}\frac{1}{1+\exp(\hat y_j^k)}$$



​	In the tree-like model, each node split is based on a strong classifier like SVM.  The general steps are as follows.

1. Set dimension limit $$d_{\mbox{limit}}$$


1. Solve $$\min_{\beta_j }|y- g_j\left(\pi_{\beta_j}(\mathcal X)\right)|$$
2. if new splited nodes are not pure, then build child trees.

#### Non-Hierarchical Ensemble

Need to set the default class that  $$c_{pref}$$ that must not be misclassified.     Thus if a   sub-model $   \forall x^k\mbox{ with } g_j\left(\pi_{\beta_j}(x^k)\right)=c_{pref},  $   then $y^k=\hat y^k=c_{pref}$.      The output is based on an OR logic 



> Disadvantage:
>
> * The input space separation is sensible to the given data.
> * The interpretability is limited.
>
> Advantage
>
> * Each sub model can be visulaized.









##  Model-Agnostic

#### Lime 

> Reference

* “Why Should I Trust You?”  Explaining the Predictions of Any Classifier

**Basic Idea:**

An explanation model $g\in G$, where $G$ is a set of interpretable models, and the domain of g is $\{0,1\}^{d'}$ .

- $\Omega (g)$: the extent of g's interpretability  e.g., depth of a tree.

- $f:R^d\rightarrow R$: denotes the origin model.

- $\pi_x (z)$ : the distance between $z$ to $x$, the weight of sample $z$. 

- $\mathcal L(f,g,\pi_x )$: denotes how **unfaithful**  g approximates f, the smaller, the closer.

- Objective is 

  ​					 $     \xi (x)=\arg \min_{g\in G}\mathcal  L(f,g,\pi_x )+\Omega(g)$

  Example functions:

  ​					$\mathcal L(f,g,\pi_x )=\sum_{z,z'} \pi_x(z)(f(z)-g(z'))^2$   

  ​					$g(z')=\vec w_g^T z'$

  ​					$\Omega(g)=\infty\mathbb I(|\vec w_g|>K)$

  ​					$\pi_x(z)=\exp(\frac{-D(x,z)^2}{\sigma^2})$

  ​

**Prediction Explain  Steps**:

1. Drawn samples  $z=(x_1',x_2',\ldots,x_d')$ uniformly around $x=(x_1,x_2,\ldots,x_d)$.
2. $z'\in \{0,1\}^{d'}$ denotes  a fraction of non-zero elements of $x'$. E.g.,  $z'=(x_1',x_5',x_8')$
3. Given data set $\mathcal Z$,    we optimize  $\xi(x)$ to get an explanation model.
4. We can choose different possibe models for $g$

**Pick Up Steps**:

1. Set budget B which denotes the number of samples to look at.

2. Given a set of instances $X ~ (|X|=n)$, construct a explain matrix $\mathcal W^{n\times d'}$, where $w_{i,j}$ denotes the importance of the  jth component in explaining instance i.

3. Let $I_j$ denotes the global importance of dimension j. For text mining,  

   ​		$I_j=\sqrt{\sum _i w_{i,j}}$,  and for images, it should be some features of super-pixels like histogram

4. summary: 在有限的budget 下， 让$\max \sum I_j (\exist i~W_{i,j}>0)$ .

**Usage in Model Selection**

1. Add noisy features.
2. Select classifiers with fewer untrustworthy predictions.


#### Anchors:  LIME Based On If-Then Rule

> Reference 

* Anchors: High-Precision Model-Agnostic Explanations

**Basic Idea:**

* A: a rule set of predicates, and $A(x)=1$ if all feature predicates are true for x. E.g., $A=\{not,bad\}$.

- Let $D(z|A)$  denotes the conditional distribution when A applies. E.g.,  $x_1='not',x_2='bad'$
- A is an anchor if :   $\mbox{prec}(A)=\mathbb E_{\mathcal D(z|A)}[f(x)=f(z)]$  and    $\mbox{prec}\geq\pi$ where z is a sample from  $D(z|A)$ where $A(z)=1$.
- If multiple anchors meet this criterion,  $P(\mbox{prec}(A)\geq\pi)\geq 1-\sigma$, then those has a larger input space are preferred (larger coverage)
- Coverage $\mbox{cov}(A)=\mathbb E_{D(z)} [A(z)]$

  ​


> Advantage: 
>
> * Anchors enable users to predict how a model would behave on unseen instances, and provide the coverage (region where explanation applies)
>
> Disadvantage:
>
> * Overly specific anchors: Predictions that are near a boundary of the black box model’s decision function, or predictions of very rare classes may require very specific “sufficient conditions”,
>   and thus their anchors may be complex and provide low coverage.
> * Potentially conflicting anchors: When using the anchor approach “in the wild”, two or more anchors with different predictions may apply to the same test instance
> * For multi-label classification setting, it is not clear if the best option would be to explain each label individually or the set of predicted labels as a single label. The former could overwhelm the user if the number of labels is too large,  while the latter may lead to non intuitive, or overly complex
>   explanations.



####  Shapley Value Explanations

> Reference

* Explaining prediction models and individual predictions with feature contributions.

**Shapley Value**:

The Shapley value is one way to distribute the total gains to the players, assuming that they all collaborate.  其实就是**各种情况下的边际贡献**

- A  set  N (of n players/features?)

- characteristic  function $v$ that maps subsets of players to the real numbers, and $v(\empty)=0$.

  - $v(S)$, called the worth of coalition describes the total expected sum of payoffs the members of $S$ can obtain by cooperation.

- According to the Shapley value, the amount that player i gets given in a coalitional game $(v,N)$ is    

  ​				$\phi_i=\sum_{S\subseteq  N\setminus i}\left[v(S\cup i)-v(S)\right]\frac{|S|!(n-|S|-1)!}{n!}$

  ​					$= \frac{1}{n!}\sum_{\mathcal O\in \pi(n)}(   v(Pr^i(\mathcal O)\cup i) -    v(Pr^i(\mathcal O)  )$,  here $\pi(n)$ is all the possible permutations, and $Pr^i(\mathcal O) $ a permuation of features precede feature i



**Basic Idea: **

* $S$  is a subset of features


* $ v(S)(x)=\mathbb E\left[f(x)|x_i~\forall i \in S\right]-\mathbb E\left[f(x)\right]$: the contribution of a subset of feature values in a particular
  instance is the change in expectation caused by observing those feature values.
  * $\mathbb E\left[f(x)\right]$ : the average output of all the input space
* How to approximate $v(S)(x)$? 
  * $v(S)(x)=\sum_{x'\in A} p(x')\left( f(\tau(x,x',S))-f(x')  \right)$
  * $\tau(x,x',S)=(z_1,z_2…z_n)$ where $z_i=x_i$ if $i\in S$.  Otherwise $z_i=x_i'$
*  $\phi_i(x)= \frac{1}{n!}\sum_{\mathcal O\in \pi(n)}\left(\sum_{x'\in A} p(x')\left( f(\tau(x,x',Pr^i(\mathcal O)\cup i))-f(\tau(x,x',Pr^i(\mathcal O))  \right)\right)$
* Finally, we can use of randomly samping to compute $\phi_i(x)$

```python
phi_i(x)=0
for i in xrange(0,m):
    randomly select permutation mathcal O and instance x' 
    tau1=(x_1,x_2,..x_i,x_i+1',x_i+2'..x_n')
    tau2=(x_1,x_2,..x_i',x_i+1',x_i+2'..x_n')
    phi_i(x)+=f(tau1)-f(tau2)
phi_i(x)/=m
```

> Advantage:
>
> * The Shapley value is the only explanation method with a solid theory. The axioms - efficiency, symmetry, dummy, additivity - give the explanation a reasonable foundation.
> * The difference between the prediction and the average prediction is fairly distributed among the features values of the instance
>
> Disadvantage:
>
> * The Shapley value needs a lot of computation time. In 99.9% of the real world problems the approximate solution - not the exact one - is feasible
> * The Shapley value is the wrong explanation method if you seek sparse explanations. Humans prefer selective explanations , like LIME produces.
> * The Shapley value returns a simple value per feature, and not a prediction model like LIME. This means it can’t be used to make statements about changes in the prediction for changes in the input.



#### Variable Importance Measures

> Reference
>
> * Fisher, Aaron, Cynthia Rudin, and Francesca Dominici. 2018. “Model Class Reliance: Variable Importance Measures for any Machine Learning Model Class, from the ‘Rashomon’ Perspective.” 



**Notations**:

* $Z^a=(Y^a, X_1^a,X_2^a)$, $Z^b=(Y^b, X_1^b,X_2^b)$ are two randomly data set

* $h_f(z^a,z^b)$ : loss of model f on $z^b$  if $x^b$  is  replaced by $x^a$, i.e.,

  ​		$h_f(z^a,z^b)=L(f,(y^b,x_1^a,x_2^b))$

* $e_{switch}(f)=\mathbb E[h_f(Z^a,Z^b)]$: the expected loss of f  on  $Z^b$  if $X^b$  is  replaced by $X^a$

  * Or we can define $e_{switch}(f)=\frac{1}{2}\mathbb E[h_f(Z^a,Z^b)+h_f(Z^b,Z^a)]$

* $e_{orig}(f)=\mathbb E[h_f(Z^a,Z^a)]=\mathbb E[L(f,Z)]$

* **Model Reliance (MR):**  $MR(f)=\frac{e_{switch}(f)}{e_{orig}(f)}$: the higher MR, the greater reliance of f on $X_1$, and $MR(f)=1$ denotes no reliance on $X_1$

  * or we can define MR as the difference rather the ratio


**Estimation MR**

Given f and data set Z,

*  $\hat e_{orig}(f)=\frac{1}{n}\sum_i^n L(f,Z_{i:})$
*  $\hat e_{switch}(f)=\frac{1}{n(n-1)}\sum_i^n\sum_{j\neq i}h_f(Z_{i:},Z_{j:})$

**Mathmatical Property**  (Require to know  causal, conditional treatment effect of a binary treatment).



####  Partial Dependence Plot

> Reference
>
> * The partial dependence plot shows the marginal effect of a feature on the predicted outcome (J. H. Friedman [2001](https://christophm.github.io/interpretable-ml-book/pdp.html#ref-friedman2001greedy)). 



**Basic Idea**:

* Suppose we want the study $X_S$, and the other varaibles are $X_C$.

* The partial dependence function for regression is 

  ​			$\int \hat f(X_S,X_C) d_{P(X_C)}=\frac{1}{n}\sum_i^n f(X_S,X_{Ci})$

#### Individual Conditional Expectation (ICE) Plot

An ICE plot visualises the dependence of the predicted response on a feature for each instance separately, resulting in multiple lines, one for each instance, compared to one line in partial dependence plots.
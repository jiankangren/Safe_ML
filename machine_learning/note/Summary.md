#Hybrid Methods

### Interpretable Models+Black Box Models 

### Analytical Model+Machine Learning

The ML corrects the output of the analytical model. For unknown inputs it is designed to produce a correction factor close to one so that the output in that case is determined by the analytical model. 

> The advantage of their approach is that the  analytical model guarantees a baseline performance which the  ML can optimize in its trusted input regions. The disadvantage of this approach is that it is only applicable on problems where an analytical model can be provided.

###### Related Work

- Schlang, M., Feldkeller, B., Lang, B., Poppe, T., & Runkler, T. (1999). Neural computation in steel industry. In Proceedings of European Control Conference ’99 (pp. 1–6). Karlsruhe, Germany: Verlag rubicon.


### Reachability Analysis+ML

###### Related Work

Guaranteed Safe Online Learning of a Bounded SystemJeremy H. Gillula†and Claire J. Tomli

> use reachbiltiy analysis to derive safe set



##  Bound Uncertainty

### Model Average

Train several models initialized with different parameters and stduy their standard deviation $$\sigma_f (x)$$ .

> Disadvantage:  there is no guarantee that the uncertainly estimation is correct.

​		

## Interpretable Machine Learning

### Learn Rejection+ Mining Rarity

Density Estimation and deactivating the prediction in low-density regions.



##### Learning with Rejection (Corinna Cortes)

* classifier $h(x)=\begin{split}1\\\circledR\\-1 \end{split}$,    rejector $r(x)\leq 0$
* Loss Function $L(h,r,x,y)=1_{yh(x)\leq0}1_{r(x)>0}+c(x)1_{r(x)\leq 0}$
* $h\in \mathcal H,~r\in \mathcal R$, then let $\hat R_S(h,r)=\mathbb E_{(x,y)\in S} [L(h,r,x,y)]$
* Confidence Rejection Model:
  * $r(x)=|h(x)|-\gamma$
  * ​

		### 	Mining with Rarity: A Unifying Framework: 

* Rarity:

  * Rare classes, or,	more generally, class imbalance

      Rare cases,  a small region of the instance space- have in fact  been referred to as within-class imbalances 						

* Relative  (low percentage) or Absolute Rarity


* Evaluation Metrics：	
  * ROC /AUC curve
  * precision=$\frac{TP}{TP+FP}$
  * recall=$\frac{TP}{TP+FN}$


* Identify Small Disjunct： only small disjuncts that are “meaningful” should be kept. A small disjunct is a disjunct that covers only a few training examples, and the vast majority of errors are concentrated in the smaller disjunct [1]

   * Perhaps, they are nosiy data.
   * Signifitance test : disjuncts with few examples cannot pass [2]


   ​			
   ​		
   ​	


* Data Fragmentation (e.g., decision tree)
  Many data mining algorithms employ a divide-and-conquer approach, where the original problem is decomposed into smaller and smaller problems, which results in the instance space being partitioned into smaller and smaller pieces.
  * Deficiency:  harder to find rarity.
*  Non-Greedy Seach
* Learn only rare cases.
* Setment data to improve relative rarity.
* Add the value of identifying positive rare cases.
* Sampling: eliminate or minize rarity by altering the distribution of training set.


1.  G. M. Weiss, and H. Hirsh. A quantitative study of small  disjuncts
2.  R. C. Holte, L. E. Acker, and B. W. Porter. Concept learningand the problem of small disjuncts. 
3.  M. V. Joshi, R. C. Agarwal, and V. Kumar. Mining needlesin a haystack: classifying rare classes via two-phase rule in-duction In SIGMOD 
   ​			
   ​		
   ​	



# Global Interpretable Models

#### Ensembles of Low-Dimensional Submodels

##### Hierarchical Ensemble Models:

------

Given training set   $$\mathcal X=\{x^1,x^2…x^M\}$$ of N diemention ,i .e.,  $$x^k=(x_1^k,x_2^k…x_N^k)$$ and output $$y^k\in \{0,1\}$$.  We project the N dimension to an aribitray subspace $$\pi_{\beta} (\mathcal X)$$, where $$\beta\subset \{1,2,3..N\}$$.   

Submodel    is defined   and input dimension is limited to two for visulization i.e., $$|\beta_j|\leq 2$$

​			$$g_j:  g_j\left(\pi_{\beta_j}(x^k)\right)=\hat y^k_j$$

​	and the outout is based on votes of the nodes in the tree

​		$$f(x^k)=\frac{1}{|models|} \sum_{i\in models}\frac{1}{1+\exp(\hat y_j^k)}$$



​	In the tree-like model, each node split is based on a strong classifier like SVM.  The 			general steps are as follows.

1. Set dimension limit $$d_{\mbox{limit}}$$


2. Solve $$\min_{\beta_j }|y- g_j\left(\pi_{\beta_j}(\mathcal X)\right)|$$
3. if new splited nodes are not pure, then build child trees.


------

##### Non-Hierarhical Ensemble Model:

------

Need to set the default class that  $$c_{pref}$$ that must not be misclassified.     Thus if a   sub-model $   \forall x^k\mbox{ with } g_j\left(\pi_{\beta_j}(x^k)\right)=c_{pref},  $   then $y^k=\hat y^k=c_{pref}$.      The output is based on an OR logic 

​					$f(x^k)=\or_{j\in Models} g_j(\pi_{\beta_j}(x^k))$

###### Related Work

* Nusser,S.,Otte,C.,Hauptmann,W.:Interpretableensemblesoflocalmodelsforsafety-related applications. In: Proceedings of 16th European Symposium on Artificial NeuralNetworks (ESANN 2008), Brugge, Belgium, pp. 301–306 (2008)
* [13]  Nusser,S.,Otte,C.,Hauptmann,W.,Kruse,R.:Learningverifiableensemblesforclas-sification problems with high safety requirements. In: Wang, L.S.L., Hong, T.P. (eds.)Intelligent Soft Computation and Evolving Data Mining: Integrating Advanced Tech-nology, pp. 405–431. IGI Global (2009)


------




##### Multivariate Adaptive Regression Splines (MARS) (适合高纬度-线性回归的推广)

MARS Basic:

* $$h_{ij}(x)=(x_i-t_j)_+=\begin{split}x_i-t_j&\mbox{ if }x_i>t_j\\0&\mbox{ otherwise }\end{split}$$

	 对于每个输入变量 $$x_i$$, 将样本中的数值作为节点， i.e.,	

  ​                     $$C=\{(x_i-t_j)_+,(t_j-x_i)_+\},t_j\in{x_i^1,x_i^2….}$$

  如果所有观察值不同,总共有 $$N\times 2 \times M$$ 个基函数

* Model:

  ​        $f(x^k)=\beta_0+\sum_{m\in C} \beta_m h_m(x^k)$

###### Related Work

* Friedman, J.H.: Multivariate Adaptive Regression Splines. The Annals of Statis-

tics 19(1), 1–67 (1991)



##### Generalized Unbiased Interaction detection and estimation 

###### Related Work

* Loh,W.:Regressionbyparts:Fittingvisuallyinterpretablemodelswithguide.In:Chen,C., Ha ̈rdle, W., Unwin, A. (eds.) Handbook of Computational Statistics, pp. 447–468(2008)

##### Symbolic Regression

Model:  $ f(x)=\sum  g_i (\pi_{\beta_i(x)})$, where $g_i$ is predefined function block like sin, max, linear, exp.

> Disadvantage: Computationally Intensive.



# Rule Based Method?

### Covering Algorithm 

 An alternative approach, the so-called covering or separate-and-conquer algo-rithm, relies on repeatedly learning a single rule (e.g., with a subgroup discoveryalgorithm). After a new rule has been learned, all examples that are covered bythis rule are removed. This is repeated until all examples are covered or a givenstopping criterion fires. A simple version of this so-called covering algorithm isshown in Figure 3, a survey of this family of algorithms can be found in [1].

1. Fu ̈rnkranz, J.: Separate-and-conquer rule learning. Artificial Intelligence Review

   13(1), 3–54 (1999)







### Explain the Local Prediction of the model 



##### LIME:  

------

An explanation model $g\in G$, where $G$ is a set of interpretable models, and the domain of g is $\{0,1\}^{d'}$ .

* $\Omega (g)$: the extent of g's interpretability  e.g., depth of a tree.

* $f:R^d\rightarrow R$: denotes the origin model.

* $\pi_x (z)$ : the distance between $z$ to $x$, the weight of sample $z$. 

* $\mathcal L(f,g,\pi_x )$: denotes how **unfaithful**  g approximates f, the smaller, the closer.

* Objective is 

  ​					 $     \xi (x)=\arg \min_{g\in G}\mathcal  L(f,g,\pi_x )+\Omega(g)$

  Example functions:

  ​					$\mathcal L(f,g,\pi_x )=\sum_{z,z'} \pi_x(z)(f(z)-g(z'))^2$   

  ​					$g(z')=\vec w_g^T z'$

  ​					$\Omega(g)=\infty\mathbb I(|\vec w_g|>K)$

  ​					$\pi_x(z)=\exp(\frac{-D(x,z)^2}{\sigma^2})$

  ​

**Prediction Explain  Steps**:

1. Drawn samples  $z=(x_1',x_2',\ldots,x_d')$ uniformly around $x=(x_1,x_2,\ldots,x_d)$.
2.  $z'\in \{0,1\}^{d'}$ denotes  a fraction of non-zero elements of $x'$. E.g.,  $z'=(x_1',x_5',x_8')$
3. Given data set $\mathcal Z$,    we optimize  $\xi(x)$ to get an explanation model.
4. We can choose different possibe models for $g$

**Pick Up Steps**:

1. Set budget B which denotes the number of samples to look at.

2. Given a set of instances $X ~ (|X|=n)$, construct a explain matrix $\mathcal W^{n\times d'}$, where $w_{i,j}$ denotes the importance of the  jth component in explaining instance i.

3. Let $I_j$ denotes the global importance of dimension j. For text mining,  

   ​		$I_j=\sqrt{\sum _i w_{i,j}}$,  and for images, it should be some features of super-pixels like histogram

4.  summary: 在有限的budget 下， 让$\max \sum I_j (\exist i~W_{i,j}>0)$ .

**Use in Model Selection**

1. Add noisy features.
2. Select classifiers with fewer untrustworthy predictions.

------

##### Anchors:  LIME Based On If-Then Rule

------

* A: a rule set of predicates, and $A(x)=1$ if all feature predicates are true for x. E.g., $A=\{not,bad\}$.
* Let $D(z|A)$  denotes the conditional distribution when A applies. E.g.,  $x_1='not',x_2='bad'$
* A is an anchor if :   $\mbox{prec}(A)=\mathbb E_{\mathcal D(z|A)}[f(x)=f(z)]$  and    $\mbox{prec}\geq\pi$ where z is a sample from  $D(z|A)$ where $A(z)=1$.
* If multiple anchors meet this criterion,  $P(\mbox{prec}(A)\geq\pi)\geq 1-\sigma$, then those has a larger input space are preferred (larger coverage)
* Coverage $\mbox{cov}(A)=\mathbb E_{D(z)} [A(z)]$
* **How to select ** from Huge number of possible anchors.?
  1. **Buttom-up Construction**:
     * ​	Initially we have an empty A which applies to any instance.
     *  In each iteration, a number of candidate rules are generated by a new feature, i.e. $A=\{A\land a_i, A\land a_{i+1},…\}$, and the one with the **highest** estimated precision $Ep$ is chosen 
     * This approaches tries to find a the shortest anchor (which may have a larger coverage)
     * How to compute $Ep$?
       * Formulated as multi-armed bandit problem and solve by KL-LUCB.

------

##### Falling Rule Lists [6]

Falling rule lists are classification models consisting of an ordered list of if-then rules,
where 

 	1. the order of rules determines which example should be classified by each rule, and
	2. the estimated probability of success decreases monotonically down the list.

A Bayesian framework for learning falling rule lists that does not rely on traditional greedy decision tree learning methods.



------

 [1]  :   Predictions are explained  with contributions of individual feature values

 [2] :  Gradient vector. The sign of each of its individual entries indicates whether the prediction would increase or decrease  when the corresponding feature of x0 is increased locally and each entry’s absolute value gives the  amount of influence in the change in prediction





###### Related Wroks

1. E. Strumbelj and I. Kononenko. An efficient explanation of individual classifications using game theory. Journal of  Machine Learning Research, 11, 2010.  (**Linear Model**)
2. D. Baehrens, T. Schroeter, S. Harmeling, M. Kawanabe, K. Hansen, and K.-R. Mu ̈ller. How to explain individual classification decisions. Journal of Machine Learning  Research, 11, 2010. (**Gradient vector**)
*  R. Caruana, Y. Lou, J. Gehrke, P. Koch, M. Sturm, and N. Elhadad. Intelligible models for healthcare: Predictingpneumonia risk and hospital 30-day readmission. InKnowledge Discovery and Data Mining (KDD), 2015. (**Additive Model**)

* B. Letham, C. Rudin, T. H. McCormick, and D. Madigan.  Interpretable classifiers using rules and bayesian analysis:Building a better stroke prediction model. Annals of AppliedStatistics, 2015.

*  B. Ustun and C. Rudin. Supersparse linear integer models for optimized medical scoring systems. Machine Learning,  2015.

* F. Wang and C. Rudin. Falling rule lists. In Artificial Intelligence and Statistics (AISTATS), 2015.

* I. Sanchez, T. Rocktaschel, S. Riedel, and S. Singh. Towards extracting faithful and descriptive representations of latent variable models. In AAAI Spring Syposium on Knowledge Representation and Reasoning (KRR): Integrating Symbolic xsand Neural Approaches, 2015.

  ​			
  ​		
  	

  ​			
  ​		
  ​	


​			
​		
​		
​		
​	
​	
​		
​			
​				






​			
​		
​	


​			
​		
​	
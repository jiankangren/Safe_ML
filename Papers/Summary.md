## Hybrid Methods

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



##  2. Bound Uncertainty

### 2.1 Model Average

Train several models initialized with different parameters and stduy their standard deviation $$\sigma_f (x)$$ .

> Disadvantage:  there is no guarantee that the uncertainly estimation is correct.

​		

## Interpretable Machine Learning

### Learn Rejection+ Mining Rarity

#### Density Estimation and deactivating the prediction in low-density regions.







### Global Interpretable Models 

#### Ensembles of Low-Dimensional Submodels

##### Hierarchical Ensemble Models

Given training set   $$\mathcal X=\{x^1,x^2…x^M\}$$ of N diemention ,i .e.,  $$x^k=(x_1^k,x_2^k…x_N^k)$$ and output $$y^k\in \{0,1\}$$.  We project the N dimension to an aribitray subspace $$\pi_{\beta} (\mathcal X)$$, where $$\beta\subset \{1,2,3..N\}$$.   

Submodel    is defined   and input dimension is limited to two for visulization i.e., $$|\beta_j|\leq 2$$

​			$$g_j:  g_j\left(\pi_{\beta_j}(x^k)\right)=\hat y^k_j$$

​	and the outout is based on votes of the nodes in the tree

​		$$f(x^k)=\frac{1}{|models|} \sum_{i\in models}\frac{1}{1+\exp(\hat y_j^k)}$$



​	In the tree-like model, each node split is based on a strong classifier like SVM.  The 			general steps are as follows.

	1. Set dimension limit $$d_{\mbox{limit}}$$
2. Solve $$\min_{\beta_j }|y- g_j\left(\pi_{\beta_j}(\mathcal X)\right)|$$
3. if new splited nodes are not pure, then build child trees.



##### Non-Hierarhical Ensemble Model

Need to set the default class that  $$c_{pref}$$ that must not be misclassified.     Thus if a   sub-model $   \forall x^k\mbox{ with } g_j\left(\pi_{\beta_j}(x^k)\right)=c_{pref},  $   then $y^k=\hat y^k=c_{pref}$.      The output is based on an OR logic 

​					$f(x^k)=\or_{j\in Models} g_j(\pi_{\beta_j}(x^k))$

###### Related Work

* Nusser,S.,Otte,C.,Hauptmann,W.:Interpretableensemblesoflocalmodelsforsafety-related applications. In: Proceedings of 16th European Symposium on Artificial NeuralNetworks (ESANN 2008), Brugge, Belgium, pp. 301–306 (2008)
* [13]  Nusser,S.,Otte,C.,Hauptmann,W.,Kruse,R.:Learningverifiableensemblesforclas-sification problems with high safety requirements. In: Wang, L.S.L., Hong, T.P. (eds.)Intelligent Soft Computation and Evolving Data Mining: Integrating Advanced Tech-nology, pp. 405–431. IGI Global (2009)



##### Multivariate Adaptive Regression Splines (MARS) (适合高纬度-线性回归的推广)

MARS Basic:

* $$h_{ij}(x)=(x_i-t_j)_+=\begin{split}x_i-t_j&\mbox{ if }x_i>t_j\\0&\mbox{ otherwise }\end{split}$$

* 对于每个输入变量 $$x_i$$, 将样本中的数值作为节点， i.e.,	

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



##### Rule Based Method/Fuzzy?





### Explain the Local Prediction of the model 











###### Related Wroks

* E. Strumbelj and I. Kononenko. An efficient explanation of individual classifications using game theory. Journal of  Machine Learning Research, 11, 2010.  (**Linear Model**)

* D. Baehrens, T. Schroeter, S. Harmeling, M. Kawanabe, K. Hansen, and K.-R. Mu ̈ller. How to explain individual classification decisions. Journal of Machine Learning  Research, 11, 2010. (**Gradient vector**)

*  R. Caruana, Y. Lou, J. Gehrke, P. Koch, M. Sturm, and N. Elhadad. Intelligible models for healthcare: Predictingpneumonia risk and hospital 30-day readmission. InKnowledge Discovery and Data Mining (KDD), 2015. (**Additive Model**)

* B. Letham, C. Rudin, T. H. McCormick, and D. Madigan.  Interpretable classifiers using rules and bayesian analysis:Building a better stroke prediction model. Annals of AppliedStatistics, 2015.

*  B. Ustun and C. Rudin. Supersparse linear integer models for optimized medical scoring systems. Machine Learning,  2015.

* F. Wang and C. Rudin. Falling rule lists. In Artificial Intelligence and Statistics (AISTATS), 2015.

* I. Sanchez, T. Rocktaschel, S. Riedel, and S. Singh. Towards extracting faithful and descriptive representations of latent variable models. In AAAI Spring Syposium on Knowledge Representation and Reasoning (KRR): Integrating Symbolic xsand Neural Approaches, 2015.


  ​			
  ​		
  ​	

* ​


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
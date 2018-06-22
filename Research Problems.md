## Possible  Research Direction



### Formal Verification of ML Component

* **Q1**: What properties are required for a ML component so that  formal verification like reachability analysis can be applied.  (**Perhaps we need a tutorial about this**)

  * **Q1.1**:Any analytical models  $f(\mathbb x)$?

  * **Q1.2**:Black box  models with a global linear  lower and upper bound ?  $ K_0||(\mathbb xx)||\leq f(\mathbb x)\leq K_1||(\mathbb xx)||$

  * **Q1.3**: Can black box  models  $f(\mathbb x)$ with  lower and upper bound for each input space partition  been verifed by formal method?    
    $$
    f(\mathbb x)=\max(l(x),\min(u(x),f(x)))\\
    l(x)=k \mbox{ if }x \in [a,b]
    $$


 If the answer to  **Q1.3**  is  **yes**, 

* **Q1.4**:  should the safe layer model from the training data or only approximate $f(\mathbb x)$ based on sampling 

  

* **Q2**: If  How to explore the tradeoff  between the  size of the bound and the accuracy of the safety layer  model?  

  * **The smaller bound, the better?**   Would it influence the accuracy of verification.  (In fact the model is safe, but become unsafe  because of the bound)

    * If we apply a larger bound,  what is the influence on verification process, while achieving higher approximation accuracy?
    * If we apply a smaller bound,  what is the influence onthe verification process, while achieving lower approximation accuracy?

  * We need a **new metric** to formualte the problem.  

    - For a required accuracy requirement, i.e., based on sampling   $P[f(\mathbb x)\in[l,u]]\geq 1-\epsilon$,  minimize the  weighted bound ?

    

* **Q3**:  we need an application scenario and compared with other approaches.

  * The metric to compare with other approaches?  
    * The time cost for  verification? 
    *  Would it influence the accuracy of verification.  (In fact the model is safe, but become unsafe  because of the bound)

    



#### Steps:

1. First of all, we need to answer  Q1 before any further action would be taken.





### ML with Reject Option



* **Q1**:  Are there more complicated methods for ML with rejection except the ensembles models based on votes ?

  * Thus, the first step is to literature review all the possible related works.

* **Q2**:  Based on Q1, we can list the possible limitations of these methods. For example, there are some obvious problems

  * The confidence based on decision boundary does not consider the space where are is no data
  * Cannot handle regression problems.

  To address the above problems, a potential strategy is cluster the empty space and space with dense data. Or we can start with the classification problem.  The key is to sampling the empty space.

* **Q3**:  How to evaluate the proposed mechanism?   We need to come up with a metric to compare with other  methods, and find an application scenario.



###How to use the fact the harzard event is rare to improve safety of ML.


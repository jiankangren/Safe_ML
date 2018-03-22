# On The Safety of Machine Learning
> Safety: 
>     1. The probability of 'expected' harms.
>     2. Possibility of 'unexpected' harms because of lack of knowledge.
> ###  Characterize Cost Serverity of Outcomes
##### Related Works
* H. Alemzadeh, J. Raman, N. Leveson, Z. Kalbarczyk, and R. K. Iyer, “Adverse events in robotic surgery: A retrospective study of 14 years of FDA data,” PLoS ONE, vol. 11, no. 4, pp. 1–20, 04 2016.
### Capture Uncertainty 
> Issues:
>     1.  Samples $$(X,Y)$$ are not drawn from the acutal distribution (perhaps unknown).
>     2.  Samples $$(X,Y)$$ are  drawn from from a known distribution but comprises only a small part of the $$\mathcal X\times \mathcal Y$$ space.



## Possible Strategies:
### 1. Inherently Safe Design 
Inherently safe design is the exclusion of a potential hazard from the system (instead of controlling the hazard).

#### 1.1 Interptable Models 
Interptable models exclude features that are not causally related to the outcome. Features or functions capturing quirks in the data can be noted and excluded, thereby avoiding related harm. Similarly, by carefully selecting variables that are causally related to the outcome, phenomena that are not a part of the true ‘physics’ of the system can be excluded, and associated harm be avoided. 

* Bayesian Case Model (BCM)   [BeenKimPhDThesis]

* Decision Tree

* Rule-based Method

* Multivariate adaptive regression splines

* Generalized unbiased interaction dection and estimation

* Symbolic regression

* Additive models

* Explain Locally 

  *  [“Why Should I Trust You? Explaining the Predictions of Any Classifier](http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) Ribeiro et al., KDD 2016

    > The general idea is simple.  Use a simple  model with (fewer but meaningfull features) to learn the function of the complicated model at a local point to explain that prediction.

  * Anchors: High-Precision Model-Agnostic Explanations AAAI

#### 1.2 Extra Regularization or Constraints on Cost Function
#####  Related Works

* Otte C. (2013) Safe and Interpretable Machine Learning: A Methodological Review. In: Moewes C., Nürnberger A. (eds) Computational Intelligence in Intelligent Data Analysis. Studies in Computational Intelligence, vol 445. Springer, Berlin, Heidelberg

  > Summary: ensemble of low-dimension interptable simple models and  use more complexity models  to learn the residual


* A. A. Freitas, “Comprehensible classification models – a position paper,” SIGKDD Explorations, vol. 15, no. 1, pp. 1–10, Jun. 2013.
* C. Rudin, “Algorithms for interpretable machine learning,” in Proc. ACM SIGKDD Conf. Knowl. Discov. Data Min., New York, NY, Aug. 2014, p. 1519.
* S. Athey and G. W. Imbens, “Machine learning methods for estimating heterogeneous causal effects,” http://arxiv.org/pdf/1504.01132.pdf, Jul. 2015.
* F. Wang and C. Rudin, “Causal falling rule lists,” http://arxiv.org/pdf/1510.05189.pdf, Oct. 2015.
* A. Chakarov, A. Nori, S. Rajamani, S. Sen, and D. Vijaykeerthy, “Debugging machine learning tasks,” http://arxiv.org/pdf/1603.07292.pdf, Mar. 2016.
* M. Petrik and R. Luss, “Interpretable policies for dynamic product recommendations,” in Proc. Conf. Uncertainty Artif.
  Intell., Jersey City, NJ, Jun. 2016, p. 74.

---
### 2. Safety Reserves 
* A safety factor is a ratio between the maximal load that does not lead to failure and the load for which the system was designed. 
* A safety margin is the difference between the two.

The uncertainty in the matching of training and test data distributions or in the instantiation of the test set can be parameterized  with the symbol $$\theta$$. Let $$R^*( \theta)$$ be the risk of the risk-optimal model if the  $$\theta$$ were known. Along the same lines as safety factors and safety margins, robust formulations find h while constraining or minimizing $$\max_{\theta}\frac{R^*(h,\theta)}{R^*(\theta)}$$ or   $$\max_{\theta}R^*(h,\theta)-R^*(\theta)$$



Such formulations can capture uncertainty in the class priors and uncertainty resulting from label noise in classification problems

#####  Related Works
* F. Provost and T. Fawcett, “Robust classification for imprecise environments,” Mach. Learn., vol. 42, no. 3, pp. 203–231, Mar. 2001.
* M. A. Davenport, R. G. Baraniuk, and C. D. Scott, “Tuning support vector machines for minimax and Neyman-Pearson classification,” vol. 32, no. 10, pp. 1888–1898, Oct. 2010

## 3. Safe Fail 

A system remains safe when it fails in its intended operation.
#### Reject Option
A technique used in machine learning when predictions cannot be given confidently is the reject option: the model reports that it cannot reliably give a prediction and does not attempt to do so, thereby failing safely.


#### Related Works
* K. R. Varshney, R. J. Prenger, T. L. Marlatt, B. Y. Chen, and W. G. Hanley, “Practical ensemble classification error bounds for different operating points,” IEEE Transactions on Knowledge and Data Engineering, vol. 25, no. 11, pp. 2590–2601, Nov. 2013. 

  Summary:

>   1. still require data representive
>
>   2. based on data, compute average accuracy (strength) and diverse->then bound the false prediction rate based on strength and correlation
>
>   3. $$\hat y(x)=\begin{split}-1&\mbox{ if }\phi(x) \leq -t\\\mbox{reject}&\mbox{ if }\phi(x)\in(-t,t)\\1 &\mbox{ if }\phi(x) \geq t\end{split}$$
>
>      ​
>
>   4. Reject with different threshold, i.e.,
>      $$ \hat y(x)=\begin{split}-1& \mbox{ if }\phi(x) \leq t_1\\\ \mbox{reject}& \mbox{ if }\phi(x)\in(t_1,t_2)\\1 &\mbox{ if }\phi(x) \geq t_2\end{split}$$
>
>      ​
>
>   5. Margin $$z=mr(x,y)=av_kI(h_k(x)=y)-\max_{j\neq y}av_k I(h_k(x)=j)$$  :  denotes the extent correct prediction over all classifiers, i.e., the larger margin, the more confidence
>
>   6. Probability Error $$P_E(t)=Pr_{X,Y}[z\in[-1,-t]]$$ : denotes the proportion of samples with margin is less than -t
>
>      * Probability Rejection $$ P_R(t)=Pr_{X,Y}[z\in [-t,t]]$$
>      * **Reject Option Risk** $$ L_c(t)=P_E(t)+c\times P_R(t)~c\in[0,0.5]$$
>
>   7. Correlation
>
> $$
> \bar p=\frac{2}{m(m-2)}\sum_{i\neq j} \mathbb E[\hat y_i(\pmb x)\hat y_j(\pmb x)]
> $$
>
> 9. Strength
>
> $$
> s=\mathbb E_z[z]=\mathbb E_{X,Y} [mr(X,Y)]
> $$
>
> 10. variance of z $$var(z)\leq \bar p(1-s^2)$$
> 11. Chebyshev inequality we get the probability  $$\Rightarrow Pr(y\neq \hat y)\leq \bar p(1-s^2)/s^2$$
> 12.  ##### Issue:  Implicitly assume the distance from the decision boundary  denotes the confidence. However,  if some part of input space has very low density, the boundary derived maybe biased.

* J. Attenberg, P. Ipeirotis, and F. Provost, “Beat the machine: Challenging humans to find a predictive model’s “unknown unknowns”,” ACM J. Data Inf. Qual., vol. 6, no. 1, p. 1, Mar. 2015.
*  G. M. Weiss, “Mining with rarity: A unifying framework,” SIGKDD Explor. Newsletter, vol. 6, no. 1, pp. 7–19, Jun. 2004.
*  Learning with Rejection Corinna Cortes1 , Giulia DeSalvo2 , and Mehryar Mohri2,1



## Example Application

## Surgical Robots

In autonomous robotic surgery, a machine learning enabled surgical robot continuously estimates the state of the environment (e.g., length or thickness of soft tissues under surgery) based on the measurements from sensors (e.g., image data or force signals) and generates a plan for executing actions (e.g., moving the robotic instruments along a trajectory). The mapping function from the perception of environment to the robotic actions is considered as a surgical skill which the robot learns, through evaluation of its own actions or from observing the actions of expert surgeons. The quality of the learned surgical skills can be assessed using cost functions that are either automatically learned or are manually defined by surgeons.

Given the uncertainty and large variability in the operator actions and behavior, organ/tissue movements and dynamics, and possibility of incidental failures in the robotic system and instruments, predicting all possible system states and outcomes and assessing their associated costs is very challenging.  For example, there have been ongoing reports of safety incidents during use of surgical robots that negatively impact patients by  causing procedure interruptions or minor injuries. These incidents happen despite existing safe
fail mechanisms included in the system and often result from a combination of different causal factors and unexpected conditions, including malfunctions of surgical instruments, actions taken  by the surgeon, and the patient’s medical history .



#### Possible Solutions

One solution for dealing with these uncertainties is to assess the robustness of the system in the presence of unwanted and rare hazardous events (e.g., failures in control system, noisy sensor measurements, or incorrect commands sent by novice operators) by simulating such events in virtual environments [7]  and quantifying the possibility of making safe decisions by the learning algorithm.



Another solution currently adopted in practice is through supervisory control of automated surgical tasks instead of fully autonomous surgery. For example, if the robot generates a geometrically optimized suture plan based on sensor data or surgeon input, it should still be tracked and updated in real time because of possible tissue motion and deformation during surgery [5].  This is an example of examining interpretable models to avoid possible harm.



 An example of adopting safety reserves (Section III-2) in robotic surgery is robust optimization of preoperative planning to minimize the uncertainty at the task level while maximizing the dexterity [8]

### Related Papers:

1. Y. Kassahun, B. Yu, A. T. Tibebu, D. Stoyanov, S. Giannarou, J. H. Metzen, and E. Vander Poorten, “Surgical robotics beyond enhanced dexterity instrumentation: a survey of machine learning techniques and their role in intelligent and autonomous surgical actions,” International Journal of Computer Assisted Radiology and Surgery, vol. 11, no. 4, pp. 553–568, 2016.

2. H. C. Lin, I. Shafran, T. E. Murphy, A. M. Okamura, D. D. Yuh, and G. D. Hager, Automatic Detection and Segmentation of Robot-Assisted Surgical Motions. Berlin, Heidelberg: Springer Berlin Heidelberg, 2005, pp. 802–810.

3. H. C. Lin, I. Shafran, D. Yuh, and G. D. Hager, “Towards automatic skill evaluation: Detection and segmentation of robot-assisted surgical motions,” Computer Aided Surgery, vol. 11, no. 5, pp. 220–230, 2006.

4.  C. E. Reiley, E. Plaku, and G. D. Hager, “Motion generation of robotic surgical tasks: Learning from expert demonstrations,”  in 2010 Annual International Conference of the IEEE Engineering in Medicine and Biology, Aug 2010, pp. 967–970.

5. A. Shademan, R. S. Decker, J. D. Opfermann, S. Leonard, A. Krieger, and P. C. W. Kim, “Supervised autonomous robotic soft tissue surgery,” Science Translational Medicine, vol. 8, no. 337, pp. 37ra64–337ra64, 2016.

6.  H. Alemzadeh, J. Raman, N. Leveson, Z. Kalbarczyk, and R. K. Iyer, “Adverse events in robotic surgery: A retrospective study of 14 years of FDA data,” PLoS ONE, vol. 11, no. 4, pp. 1–20, 04 2016.

7.  H. Alemzadeh, D. Chen, A. Lewis, Z. Kalbarczyk, J. Raman, N. Leveson, and R. K. Iyer, “Systems-theoretic safety assessment of robotic telesurgical systems,” in Proc. Int. Conf. Comput. Safety Reliability Secur., 2015, pp. 213–227.

8.  H. Azimian, M. D. Naish, B. Kiaii, and R. V. Patel, “A chance-constrained programming approach to preoperative planningof robotic cardiac surgery under task-level uncertainty,” IEEE Trans. Biomed. Health Inf., vol. 19, no. 2, pp. 612–1898,  Mar. 2015.

   ​

   ​

## Self-Driving Cars:

Self-driving cars are autonomous cyber-physical systems capable of making intelligent navigation decisions in real-time without any human input. They combine a range of sensor data from laser range finders and radars with video and GPS data to generate a detailed 3D map of the environment and estimate their position. The control system of the car uses this information to determine the optimal path to the destination and sends the relevant commands to actuators that control the steering, braking, and throttle

#### Several sources of uncertainty and failure 

*  Unreliable or noisy sensor signals (e.g., GPS data or video signals in bad weather conditions), limitations of computer vision systems, and 
* unexpected changes in the environment (e.g., unknown driving scenes or unexpected accidents on the road) can adversely affect the ability of control system in learning and understanding the environment and making safe decisions.



The importance of epistemic uncertainty or ”uncertainty on uncertainty” in these AI-assisted
systems has been recently recognized, and there are ongoing research efforts towards quantifying
the robustness of self-driving cars to events that are rare (e.g., distance to a bicycle running on
an expected trajectory) or not present in the training data (e.g., unexpected trajectories of moving
objects) [1].  **Systems that recognize such rare events trigger safe fail mechanisms.**

To the best of our knowledge, there is no self-driving car system with an inherently safe design that utilizes, e.g., interpretable models.  Fail-safe mechanisms that upon detection of failures or less confident predictions, stop the autonomous control software and switch to a  backup system or a degraded level of autonomy (e.g., full control by the driver) are considered for self-driving cars [2].

### Related Papers

1. J. Duchi, P. Glynn, and R. Johari, “Uncertainty on uncertainty, robustness, and simulation,” SAIL-Toyota Center for AI Research, Stanford University, Tech. Rep., Jan. 2016.
2.  P. Koopman and M. Wagner, “Challenges in autonomous vehicle testing and validation,” SAE International Journal of Transportation Safety, vol. 4, no. 2016-01-0128, pp. 15–24, 2016.




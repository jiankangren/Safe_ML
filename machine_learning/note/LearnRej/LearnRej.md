#ID2 Mining with Rarity: A Unifying Framework (Survey/Mainly for data science)

1. Rare classes/cases/ imbalanced data
2. evaluation metrics to hand rare classes and rare cases
3. generalize from specific examples requires an extra-evidentiary bias
4. noisy data influence more on rare data

### Methods to Improve
1. Evaluation Metrics take rarity into account
* ROC analysis
2. Non-Greedy Search Tech (e.g., DT use greedy)
3. Bias on specilization
4. Learn only rare class
5. **Segment data** to R1 and R2 and partition problem into sub-problems. So in R1 the percentage rare cases becomes large and R2 becomes even rarer (but acceptable)
6. add more weights (costs) for identifying positive rare class.
7. Sampling.
* under-sampling: eliminate majority class example
* over-sampling: add minority-class
* Two phase rule induction:
1. If high precision rules cannot be achieved, then low precision rules are acceptable as long as they have relatively high recall-the first phase focus on recall.
2. Second phase, optimize precision.

---

#ID3 Beat Machine
Use regard to encourage people to find rare sample that the model is confident but wrong.

---

#ID1 Practical Ensemble Classification Error Bounds for Different Operating Points （Ensemble models）
1. still require data representive
2. based on data, compute average accuracy (strength) and diverse->then bound the false prediction rate based on strength and correlation

#### Classify with Reject Option

* $$\hat y(x)=\begin{split}
-1&\mbox{ if }\phi(x) \leq -t\\
\mbox{reject}& \mbox{ if }\phi(x)\in(-t,t)\\
1 &\mbox{ if }\phi(x) \geq t
\end{split}
$$

* Reject with different threshold, i.e.,
$$\hat y(x)=\begin{split}
-1&\mbox{ if }\phi(x) \leq t_1\\
\mbox{reject}& \mbox{ if }\phi(x)\in(t_1,t_2)\\
1 &\mbox{ if }\phi(x) \geq t_2
\end{split}
$$
* Margin $$mr(X,Y)$$: average number of right prediction-(wrong+reject) over all classifiers
* probability density of margin: $$
\int_{X,Y} f_{mr}(X,Y) d_Xd_Y=mr(X,Y)\\
f(z)=f(mr(X,Y))=f_{mr}(X,Y)
$$

* Probability Error $$P_E(t)=Pr[z\in[-1,-t]]$$ :就是指考虑了那些被预测为negtaive 的样本的error
* Probability Rejection $$ P_R(t)=Pr[z\in [-t,t]]$$
* **Reject Option Risk** $$ L_c(t)=P_E(t)+c\times P_R(t)~c\in[0,0.5]$$

#### Strength & Correlation


* Correlation
$$
\bar p=\frac{2}{m(m-2)}\sum_{i\neq j} \mathbb E[\hat y_i(\pmb x)\hat y_j(\pmb x)]
$$

* Strength
$$
s=\mathbb E_z[z]=\mathbb E_{X,Y} [mr(X,Y)]
$$

* variance of z $$var(z)\leq \bar p(1-s^2)$$
* Chebyshev inequality $$\Rightarrow Pr(y\neq \hat y)\leq \bar p(1-s^2)/s^2$$



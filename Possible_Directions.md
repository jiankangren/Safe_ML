### Hierarchicaly Method: Low Height DT+High Complexity Black Box Model 

1.  Build a low hight tree and meet certian standard.
2.  Data in each leaf node can then modeled by more complex methods.



### Local Interpretability for a Group of Prediction

The model predictions for multiple instances can be explained by either using methods for global model interpretability (on a modular level) or single instance explanations. The global methods can be applied by taking the group of instances, pretending it’s the complete dataset, and using the global methods on this subset. The single explanation methods can be used on each instance and listed or aggregated afterwards for the whole group.





## Case Based Method?

If there is no good exemplar,  then reject prediction?  But such approach is suitable for decision making.











### Evaluating the Interpretability Quality:

* **Application level evaluation (real task)**: Put the explanation into the product and let the end user test it.
* **Human level evaluation (simple task)** is a simplified application level evaluation. The difference is that these experiments are not conducted with the domain experts, but with lay humans. This makes experiments less expensive (especially when the domain experts are radiologists) and it is easier to find more humans
* **Function level evaluation (proxy task)** does not require any humans. This works best when the class of models used was already evaluated by someone else in a human level evaluation. For example it might be known that the end users understand decision trees. In this case, a proxy for explanation quality might be the depth of the tree. Shorter trees would get a better explainability rating.



There are more dimensions to interpretability:

- Model sparsity: How many features are being used by the explanation?
- Monotonicity: Is there a monotonicity constraint? Monotonicity means that a feature has a monotonic relationship with the target. If the feature increases, the target either always increases or always decreases, but never switches between increasing and decreasing.
- Uncertainty: Is a measurement of uncertainty part of the explanation?
- Interactions: Is the explanation able to include interactions of features?
- Cognitive processing time: How long does it take to understand the explanation?
- Feature complexity: What features were used for the explanation? PCA components are harder to understand than word occurrences, for example.
- Description length of explanation.



### “Good” Explanation?



##### Contrastive explanations are easier to understand than complete explanations.

A doctor who wonders: “Why did the treatment not work on the patient?”, might ask for an explanation contrastive to a patient, where the treatment worked and who is similar to the non-responsive patient. Contrastive explanations are easier to understand than complete explanations. A complete explanation to the doctor’s why question (why does the treatment not work) might include: The patient has the disease already since 10 years, 11 genes are over-expressed making the disease more severe, the patients body is very fast in breaking down the medication into ineffective chemicals , etc.. The contrastive explanation, which answers the question compared to the other patient, for whom the drug worked, might be much simpler: 

The best explanation is the one that highlights the greatest difference between the object of interest and the reference object. **What it means for interpretable machine learning**: Humans don’t want a complete explanation for a prediction but rather compare what the difference were to another instance’s prediction (could also be an artificial one). 



**Explanations are selected**: 

People don’t expect explanations to cover the actual and complete list of causes of an event. We are used to selecting one or two causes from a huge number of possible causes as THE explanation.

**What it means for interpretable machine learning**: Make the explanation very short, give only 1 to 3 reasons, even if the world is more complex. The LIME method, described in Chapter [5.4](https://christophm.github.io/interpretable-ml-book/lime.html#lime) does a good job with this.



**Explanations focus on the abnormal**.

If one of the input features for a prediction was abnormal in any sense (like a rare category of a categorical feature) and the feature influenced the prediction, it should be included in an explanation, even if other ‘normal’ features have the same influence on the prediction as the abnormal one. An abnormal feature in our house price predictor example might be that a rather expensive house has three balconies. Even if some attribution method finds out that the three balconies contribute the same price difference as the above average house size, the good neighbourhood and the recent renovation, the abnormal feature “three balconies” might be the best explanation why the house is so expensive.
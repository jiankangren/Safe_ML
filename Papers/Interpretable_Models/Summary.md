
## Safe and Interpretable Machine Learning: A Methodological Review
### Existing Solutions
1. Model Averaging  with different initialization and study the deviation on unexplored space.
    *   Limitations: Models not fully independant.
2.  Estimate input space density and deactivate the model in lowdensity regions.
      *   Limitations: density estimation in high demension is hard
3. Constrain model using a-priori knowledge.
    * if y and x monotonical.
4. Limiting the output range.
### Ensemble of Low Dimension Models (AirBag)
* Input n-dimension x and binary y
1. train a model of two dimension and can classifiy all negative samples(stiuation that do not need  alarm)
2.  select best 2-d model with expert knowledge
3.  remove true positive samples
4.  go to step 2
### Use a interptable model+gaussian processor(learn the residuals)
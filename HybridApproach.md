
## Parallelized hybridization design

- Output of several existing implementations combined
- Least invasive design
- Some weigting or voting schema
    - Weights can be learned dynamically
    - Extreme case of dynamic weighting is switching
    
    
![](res/parallelized_hybridization_design_idea.PNG)
    
**Weighted**

- But how to drive weights?
    - Estimate, e.g. by empirical bootstrapping
    - Dynamic adjustment of weights

- Empirical bootstrapping
    - Historical data is needed
    - Compute different weightings
    - Decide which one does best
    
- Dynamic adjustment of weights
    - Start with for instance uniform weight distribution
    - For each user adapt weights to minimize error of prediction
    
- Minimize MAE

![](res/parallelized_hybridization_design_weighted_MAE.PNG)

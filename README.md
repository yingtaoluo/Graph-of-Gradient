# Graph-of-Gradient
Please see newly added experiments "New Experiment Table 4.png", which is a hyperparameter sensitivity study on "K" of the KNN algorithm. It shows that "K" generally does not affect the model's performance.

Please see newly added visualization "New Visualization Figure 3.png". It is visualizing the distribution of input features and last-layer gradients upon different race groups for MIMIC-III dataset (Purple: White, Yellow: 'BLACK', Green: 'HISPANIC', Others are discarded for the convenience of visualization and presentation). Figure 3 shows that while it is impossible to separate people by race based on input features, we can actually see that the least populated subpopulation group (colored by green) is distributed closely. This case study demonstrates that gradient is indeed a better representation of the true demographics, when the true demographics is unknown.

"Normal" folder contains Baselines (without fairness algorithms). 

"models" folder together with "train.py" is an example implementation of GoG.
